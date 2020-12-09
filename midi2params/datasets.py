import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import os
import pickle
import pretty_midi
from scipy import stats
from scipy.io.wavfile import read as wavread
import matplotlib.pyplot as plt
import time
import copy
from addict import Dict

from utils.util import p2f, to_numpy

############
# Datasets #
############

class MIDIParamsDataset(Dataset):
    """
    Dataset class specifically suited for MIDI -> params training. The majority of the code
    can be used for params -> MIDI training as well, but there are small parts that require editing.
    Assumes the following:
    - dataset path has folders self.midi_folder_name/ and self.params_folder_name/
    - Each of the folders above is split into folder/{train, val, test}, and each subfolder therein
      contains just files
    
    We also designate two *distinct* lengths of a clip. We first have a length of the inputted clip, which should
    be (usually) 5 seconds. Then, we have the length of the cropped clip (usually 4 seconds) to help account
    for any f0/loudness signal offsets.
    """
    def __init__(self, config=None,
                 settype: str = 'train'):
        # --PARAMETERS--
        # config: configuration dict as read in in trainscript.py
        # settype: should use training or val or test?

        if config is None:
            raise 'Must provide a config.'

        self.dset_path = config.dataset.dset_path
        self.params_folder_name = config.dataset.params_folder_name
        self.midi_folder_name = config.dataset.midi_folder_name
        # NOTE: here we account for dataset discrepancies where a newer dataset
        # may have MIDI files already processed into 'pitches', 'onset_arr', 'offset_arr'
        # in order to prevent trailing notes
        first_midi_file = os.listdir(os.path.join(self.dset_path, self.midi_folder_name, 'train'))[0]
        if os.path.splitext(first_midi_file)[1] in ['.mid', '.midi']:
            self.midi_type = 'raw'
        elif os.path.splitext(first_midi_file)[1] == '.p':
            self.midi_type = 'processed'
        else:
            raise 'Unsupported file extension for MIDI files.'

        if type(config.dataset.audio_folder_name) == type(''):
            print('USING AUDIO')
            self.audio = True
            self.audio_folder_name = config.dataset.audio_folder_name
        else:
            print('NOT USING AUDIO') 
            self.audio = False
        self.settype = settype
        self.config = config
        self.len_clip = config.preprocessing.len_clip
        self.len_subclip = config.preprocessing.len_subclip
        self.offset_f0 = config.preprocessing.offset_f0
        self.offset_ld = config.preprocessing.offset_ld
        self.frame_rate = config.frame_rate
        self.sample_rate = config.sample_rate
        self.ld_bins = config.preprocessing.ld_bins
        self.cent_bins = config.preprocessing.cent_bins

        f_ids = np.array([os.path.splitext(f)[0] for f in os.listdir(os.path.join(self.dset_path, self.midi_folder_name, settype))])
        # NOTE: there may be f_ids with control parameters with <self.len_clip*self.frame_rate samples
        # this is a preventative measure, just to prevent any bugs in __getitem__
        # TODO: refactor this tremendous expression into multiple lines
        valids = (np.array([pickle.load(open(os.path.join(self.dset_path, self.params_folder_name, self.settype, f_id + '.p'), 'rb'))['f0_hz'].shape[0] for f_id in f_ids]) >= self.len_clip * self.frame_rate)
        f_ids = f_ids[np.where(valids)[0]]
        if config.dataset.truncate_to != Dict() and config.dataset.truncate_to >= 1 and settype == 'train':
            f_ids = f_ids[:config.dataset.truncate_to]

        self.f_ids = sorted(f_ids)

    def format_labels(self, batch):
        """
        Format raw data into actual ground truth outputs for our model. Batch
        gives raw (N, 1250) shape signals and we transform these into:
        - (potentially) Gauss-ified cents and loudness
        - one-hot indexed cents and loudness, and
        """

        # here, we take what the dataloader spits out and format it into
        # ground truth tensors for f0 and loudness and teacher forcing
        # labels to use as input during training

        pitches = copy.deepcopy(batch['pitches'])
        f0 = copy.deepcopy(batch['f0'])
        #np.save('notebooks/pitches.npy', pitches)
        #np.save('notebooks/f0.npy', f0)
        cents = 1200 * torch.log2(f0 / p2f(pitches))  # compute cents
        cents[np.where(cents <= -50)] = -50  # clip negative values
        cents[np.where(cents >= 50)] = 50  # clip positive values
        cents[np.where(torch.isnan(cents))] = 0
        # now compute ground truth for f0
        f0_gt_discrete = (cents + 50).long()
        assert (f0_gt_discrete >= 0).all(), f0_gt_discrete[np.where(f0_gt_discrete < 0)[0]]
        if self.config.training.gaussian_during_train:
            f0_gt_gauss = cents.long() + 50
            f0_gt_gauss = to_gauss(f0_gt_gauss,
                             101,
                             scale=self.config.training.gaussian_during_train_std).float()

        # compute ground truth for loudness
        ld_gt_discrete = (batch['loudness_db'] + 120).long()
        if self.config.training.gaussian_during_train:
            ld_gt_gauss = (batch['loudness_db'] + 120).long()
            ld_gt_gauss = to_gauss(ld_gt_gauss,
                             121,
                             scale=self.config.training.gaussian_during_train_std).float()

        # update in-place
        batch.update({
            'f0_gt_discrete': f0_gt_discrete,
            'ld_gt_discrete': ld_gt_discrete
        })
        if self.config.training.gaussian_during_train:
            batch.update({
                'f0_gt_gauss': f0_gt_gauss,
                'ld_gt_gauss': ld_gt_gauss
            })

        return batch

    def load_params(self, f_id):
        params = pickle.load(open(os.path.join(self.dset_path, self.params_folder_name, self.settype, f_id + '.p'), 'rb'))

        f0 = params['f0_hz']
        loudness_db = params['loudness_db']
        conf = params['f0_confidence']
        return f0, loudness_db, conf

    def load_midi(self, f_id):
        if self.midi_type == 'raw':
            # get MIDI notes
            midi_path = os.path.join(self.dset_path, self.midi_folder_name, self.settype, f_id + '.mid')
            # handle the possibility of either .mid or .midi files
            try:
                midi = pretty_midi.PrettyMIDI(midi_path)
            except FileNotFoundError:
                midi = pretty_midi.PrettyMIDI(midi_path + 'i')
            print(f_id)
            print(midi.instruments)
            notes = midi.instruments[0].notes

            # get onsets/offsets from MIDI
            onsets = [int(self.frame_rate * n.start) for n in notes]
            offsets = [int(self.frame_rate * n.end) for n in notes]
            # NOTE: make extra long arrays just in case the MIDI notes go on for longer
            onset_arr = np.zeros((3 * self.len_clip * self.frame_rate), dtype=np.float32)
            onset_arr[onsets] = 1  # embed pointwise onsets/offsets into zero array
            offset_arr = np.zeros((3 * self.len_clip * self.frame_rate), dtype=np.float32)
            offset_arr[offsets] = 1  # embed pointwise onsets/offsets into zero array

            # get pitches
            pitches = notes2pitches(notes, 3 * self.len_clip * self.frame_rate, NO_NOTE_VAL=0)
        else: # self.midi_type == 'processed'
            fpath = os.path.join(self.dset_path, self.midi_folder_name, self.settype, f_id + '.p')
            arrs = pickle.load(open(fpath, 'rb'))
            pitches, onset_arr, offset_arr = arrs['pitches'].astype(np.float32), arrs['onset_arr'].astype(np.float32), arrs['offset_arr'].astype(np.float32)
        
        return pitches, onset_arr, offset_arr

    def random_truncate(self, batch):
        """
        Randomly truncate our batch from length self.len_clip to length self.len_subclip.
        """
        # NOTE: we take into account the f0 and ld offsets
        # by taking the min/max of any offset:
        # * offset_f0
        # * offset_f0 - 1 (teacher forcing)
        # * offset_ld,
        # * offset_ld - 1 (teacher forcing)
        # * 0
        max_offset = max(self.offset_f0, self.offset_ld, 0)  # will be non-negative
        min_offset = min(self.offset_f0 - 1, self.offset_ld - 1, 0)  # will be non-positive
        max_tstep = (self.len_clip - self.len_subclip) * self.frame_rate - max_offset
        min_tstep = -min_offset
        tstep = np.random.randint(min_tstep, max_tstep)
        
        len_subsequence = int(self.len_subclip * self.frame_rate)
        for k, arr in batch.items():
            offset = 0
            if 'ld' in k or 'loudness' in k:
                offset = self.offset_ld
            elif 'f0' in k:
                offset = self.offset_f0
            
            if 'teacher_forcing' in k:
                # we want the labels to be shifted back by one
                offset -= 1

            batch[k] = arr[offset + tstep:offset + tstep + len_subsequence]
        return batch

    def format_input(self, batch):
        """
        Take 'pitches' and teacher forcing arrays and concatenate.
        """
        # convert input pitches to one-hot
        x = F.one_hot(batch['pitches'].long(), 129).float()
        # concatenate pitch one-hots to onsets and offsets

        x = torch.cat((x,
                       batch['onset_arr'].unsqueeze(-1),
                       batch['offset_arr'].unsqueeze(-1)), dim=-1)

        # concatenate teacher-forcing labels onto x
        autoregressive_type = self.config.model.constructor_args.autoregressive_type
        if autoregressive_type == 'onehot':
            x = torch.cat((x,
                          batch['teacher_forcing_f0'],
                          batch['teacher_forcing_ld']), dim=-1)
        elif autoregressive_type == 'scalar':
            x = torch.cat((x,
                           batch['teacher_forcing_f0'].unsqueeze(-1),
                           batch['teacher_forcing_ld'].unsqueeze(-1)), dim=-1)
        
        batch['x'] = x
        
        return batch

    def load_audio(self, f_id):
        audio_path = os.path.join(self.dset_path, self.audio_folder_name, self.settype, f_id + '.wav')
        _, audio = wavread(audio_path)

        return audio
        
    def get_item(self, idx):
        """
        Does all of the deterministic heavy duty I/O and preprocessing in __getitem__.
        """

        # get file id (name of file without the extension)
        f_id = self.f_ids[idx]

        ##############
        # PARAMETERS #
        ##############

        f0, loudness_db, conf = self.load_params(f_id)

        assert f0.shape == loudness_db.shape == conf.shape
        assert (-120 <= loudness_db).all() and (loudness_db <= 0).all()
        assert (0 <= conf).all() and (conf <= 1).all()
        # exclude last 20 because f0 is zeros there (frames and stuff)
        # 30 because viterbi decoder has some wiggle room (???)
        assert (30 <= f0[:-20]).all() and (f0[:-20] <= 1975.5).all(), np.save('f0-for-{}.npy'.format(f_id), f0)

        ########
        # MIDI #
        ########

        pitches, onset_arr, offset_arr = self.load_midi(f_id)

        assert (0 <= pitches).all() and (pitches <= 128).all()
        assert (0 <= onset_arr).all() and (onset_arr <= 1).all()
        assert (0 <= offset_arr).all() and  (offset_arr <= 1).all()

        #########
        # AUDIO #
        #########

        if self.audio:
            audio = self.load_audio(f_id)

        #############################
        # TRUNCATE TO self.len_clip #
        #############################

        batch = {
            'f0': f0,
            'loudness_db': loudness_db,
            'conf': conf,
            'pitches': pitches,
            'onset_arr': onset_arr,
            'offset_arr': offset_arr
        }

        batch = {k: torch.Tensor(batch[k][:self.len_clip * self.frame_rate]) for k in batch.keys()}


        if self.audio:
            batch['audio'] = audio[:self.len_clip * self.sample_rate]

        ###########################
        # GET GROUND TRUTH LABELS #
        ###########################

        batch = self.format_labels(batch)

        return batch

    def __getitem__(self, idx):
        ########################################
        # GET BATCH OF LENGTH self.len_subclip #
        ########################################

        # NOTE: this compartmentalization was originally intended to
        # use lru_cache
        batch = self.get_item(idx)

        return batch

    def __len__(self):
        return len(self.f_ids)

def to_gauss(onehot_indices, num_classes, scale=5):
    """
    Gauss-ify a long tensor of shape (N_1,...,N_k) with integers in the interval [0, num_classes) as
    seen in https://arxiv.org/pdf/1802.06182.pdf. Output will be of shape (N_1,...,N_k,num_classes).
    """
    
    tensor_shape = onehot_indices.shape
    
    onehot_indices = onehot_indices.flatten()  # flatten into (N_1 + ... + N_k,)
    
    onehot_indices = to_numpy(onehot_indices).astype('int')
    # TODO: add some caching to improve efficiency in loading this gauss_arr (use tempfile library)
    gauss_arr = np.array([stats.norm(loc=i, scale=scale).pdf(np.arange(num_classes)) for i in range(num_classes)])
    gaussed = gauss_arr[onehot_indices]
    normalized = gaussed / gaussed.sum(axis=1)[..., np.newaxis]
    return torch.Tensor(normalized).view(*tensor_shape, num_classes)

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

def normalize(x, min_, max_):
    return (x - min_) / (max_ - min_)
