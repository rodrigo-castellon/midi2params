import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import numpy as np
import os
import pickle
import pretty_midi

import wavegenie.constants as constants

############
# Datasets #
############

class FlesDataset(Dataset):
    """
    Dataset class specifically suited for fles training. Assumes the following:
    - working with synthetic data
    - dataset path has folders modified_midis/ and synth_params/ but does not need clips/ or synths/
    - an index of the dataset is 
    """
    def __init__(self, dset_path, settype='train'):
        # dset_path: path to dataset, which contains paired training data, with:
        #   dset_path/{synth_params, modified_midis}
        #   and in each we have dset_path/[folder]/{train, val, test}
        # settype: should use training or val or test?

        self.params_folder_name = 'params'
        self.midi_folder_name = 'midi'
        self.len_clip = 5
        self.sample_rate = 16000
        self.frame_rate = 250

        f_ids = np.array([os.path.splitext(f)[0] for f in os.listdir(os.path.join(dset_path, self.midi_folder_name, settype))])
        # note that there may be f_ids with control parameters with <self.len_clip*self.frame_rate samples
        valids = (np.array([pickle.load(open(os.path.join(dset_path, self.params_folder_name, settype, f_id + '.p'), 'rb'))['f0_hz'].shape[0] for f_id in f_ids]) >= self.len_clip * self.frame_rate)
        f_ids = f_ids[np.where(valids)[0]]
        self.f_ids = sorted(f_ids)
        self.dset_path = dset_path
        self.settype = settype
    def __getitem__(self, idx):
        # get file id (name of file without the extension)
        f_id = self.f_ids[idx]

        ##############
        # PARAMETERS #
        ##############
        params = pickle.load(open(os.path.join(self.dset_path, self.params_folder_name, self.settype, f_id + '.p'), 'rb'))
        
        f0 = params['f0_hz']
        loudness_db = params['loudness_db']
        conf = params['f0_confidence']

        ########
        # MIDI #
        ########
        # get MIDI
        midi_path = os.path.join(self.dset_path, self.midi_folder_name, self.settype, f_id + '.midi')
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = midi.instruments[0].notes

        # get onsets/offsets from MIDI
        onsets = [int(self.frame_rate * n.start) for n in notes]
        offsets = [int(self.frame_rate * n.end) for n in notes]
        onset_arr = np.zeros((10 * self.frame_rate), dtype=np.float32)
        onset_arr[onsets] = 1  # embed pointwise onsets/offsets into zero array
        offset_arr = np.zeros((10 * self.frame_rate), dtype=np.float32)
        offset_arr[offsets] = 1  # embed pointwise onsets/offsets into zero array
        
        # get pitches
        pitches = notes2pitches(notes, 10 * self.frame_rate, NO_NOTE_VAL=0)
        
        # truncate to 4 seconds exactly
        f0 = f0[:self.len_clip * self.frame_rate]
        loudness_db = loudness_db[:self.len_clip * self.frame_rate]
        conf = conf[:self.len_clip * self.frame_rate]

        pitches = pitches[:self.len_clip * self.frame_rate]
        onset_arr = onset_arr[:self.len_clip * self.frame_rate]
        offset_arr = offset_arr[:self.len_clip * self.frame_rate]

        return f0, loudness_db, conf, pitches, onset_arr, offset_arr
    def __len__(self):
        return len(self.f_ids)


def load_dataset(set_type, config):
    # set_type is "train", "val", or "test"

    dset_fpath = config.dataset.fpath
    
    dataset = FlesDataset(dset_fpath, settype=set_type)
    
    return dataset



##########
# Models #
##########

class SeqModel(nn.Module):
    def __init__(self, hidden_size=30):
        super(SeqModel, self).__init__()
        self.rnn = nn.GRU(129, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 101 + 129)  # 101 cents + 129 loudness values
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out)
        cent_logits, ld_logits = out[..., :101], out[..., 101:]
        return cent_logits, ld_logits

model_dict = {
    'linear-1': nn.Linear(129, 1),
    'conv-1-5': nn.Conv1d(constants.NUM_MIDI_PITCHES + 1, 1, 5),
    'seq-1': SeqModel()
}

##################
# Loss Functions #
##################

# each loss function here expects that the predicted/ground truth
# output takes on the form of a tensor of shape N x seqlen

def regression_loss(pred, y):
    # naive loss function: add up all elementwise
    # l2 distances
    return nn.MSELoss()(pred, y)

loss_fn_dict = {
    'regression': regression_loss
}

##############
# Optimizers #
##############

def get_optimizer(model, config):
    # check each hyperparameter
    if 'learning_rate' in config.training:
        lr = config.training.learning_rate
    else:
        lr = 0.01

    if 'weight_decay' in config.training:
        weight_decay = config.training.weight_decay
    else:
        weight_decay = 0.0

    if config.training.optim == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    else:
        pass

    return optim

################################
# Miscellaneous/Data Wrangling #
################################

p2f = lambda d: 440 * 2**((d - 69) / 12)

def notes2pitches(notes, length, NO_NOTE_VAL=128, FRAME_RATE=250, transform=None):
    """
    Turn a list of monophonic notes into a list of pitches
    at 250Hz. In other words, turn [Note(), Note(), ...] into
    np.array [65, 65, 65, 65, 128, ...]
    """

    # NOTE: [0, 127] is reserved by MIDI, but we use
    # NO_NOTE_VAL as a flag value for "no note playing right now"
    pitches = np.full((length), NO_NOTE_VAL, dtype=np.float32)

    for note in notes:
        start_idx = int(note.start * FRAME_RATE)
        end_idx = int(note.end * FRAME_RATE)
        pitch = note.pitch
        if not(transform is None):
            pitch = transform(pitch)
        pitches[start_idx:end_idx] = pitch
    return pitches

def get_loader(dataset, config, set_type):
    # get the dataloader from the dataset object

    # (potentially) don't shuffle during evaluation
    if set_type in ['val', 'eval', 'test']:
        shuffle = config.loader.shuffle_during_eval
    else:
        shuffle = True

    dataloader = DataLoader(dataset,
                            batch_size=config.loader.batch_size,
                            shuffle=shuffle,
                            num_workers=config.loader.num_workers,
                            pin_memory=config.loader.pin_memory)
    
    return dataloader

