"""
Reproduce midi2params by doing a forward pass on a given example from the test set.
Specifically, we get the last batch of the test set and choose the 7th example, which
has paired <midi,audio>, which allows us to do several listening tests at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import yaml
from addict import Dict
from datetime import datetime
import copy
from scipy.io.wavfile import write as wavwrite

from train_utils import *

args = parse_arguments()

args.config = '/work/midi2params/midi2params/configs/midi2params-test.yml'

# get config
print('getting config')
config = load_config(args.config)

# override if we just don't have a GPU
if not(torch.cuda.is_available()) and config.device == 'cuda':
    config.device = 'cpu'

seed_everything(config.seed)

# set up a unique identifier for this session
sess_dt = str(datetime.now()).split('.')[0]

if config.device == 'cuda':
    dtype_float = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
elif config.device == 'cpu':
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
else:
    raise Exception('Invalid device type')

# get dataset here
print('getting dataset')
set_types = ['train', 'val', 'test']
dset_path = config.dataset.constructor_args.dset_path
print('loading in dataset fully')
sets = []
for set_type in set_types:
    print('getting {}'.format(set_type))
    sets.append(load_dataset(set_type, config))
    print('{} is of length {}'.format(set_type, len(sets[-1])))

trainset, valset, testset = sets

# make dataloaders
print('making dataloaders')
loaders = []
for dset, set_type in zip([trainset, valset, testset], set_types):
    loaders.append(get_loader(dset, config, set_type))
train_loader, val_loader, test_loader = loaders

# load model
print('loading model')
model = load_model(config)
if config.device == 'cuda':
    model = model.cuda()

# define loss function
print('defining loss function')
loss_fn_discrete = loss_fn_dict['cross-entropy-1']
loss_fn_soft = loss_fn_dict['cross-entropy-2']

# define optimizer
print('defining optimizer')
optimizer = get_optimizer(model, config)


# get the last batch from test_loader

for batch in test_loader:
    pass


i = 7
print(batch.keys())

audio = to_numpy(batch['audio'][i])[..., np.newaxis]

# extract the f0/loudness features/parameters with DDSP
print('extracting f0/loudness parameters with DDSP...')
from utils.util import extract_ddsp_synthesis_parameters

audio_parameters = extract_ddsp_synthesis_parameters(audio)


# now load the DDSP model
print('loading DDSP model...')

from utils.util import load_ddsp_model

ckpt_path = '/work/midi2params/checkpoints/CustomViolinCheckpoint'
model = load_ddsp_model(ckpt_path)

# now resynthesize the same audio, should sound similar
print('resythesizing...')

from utils.util import synthesize_ddsp_audio

resynth = synthesize_ddsp_audio(model, audio_parameters)

#print(resynth)
#print(resynth.shape)
wavwrite('test1.wav', 16000, resynth)

# now we take the MIDI for this example and heuristically generate
# reasonable f0/loudness curves via heuristics

def generate_loud(beats, length=1250, decay=True):
    """
    Generate a loudness envelope for each note, decaying over time.
    """
    arrs = []
    length = 2500
    base = -30
    decay_rate = -0.01 # decays -1 per timestep/index
    #notelength = 0.7
    ld_arr = np.full((length), -120)
    for i, beat in enumerate(beats):
        if i == len(beats) - 1:
            next_beat = length
        else:
            next_beat = beats[i + 1]
        ld_arr[beat:next_beat] = np.linspace(base, base + decay_rate * (next_beat - beat), next_beat - beat)

    return ld_arr


def gen_heuristic(batch, i=0):
    """
    Take a batch containing 'pitches', 'onset_arr', and 'offset_arr' and
    turn them into f0 and loudness heuristically.
    """
    onsets = np.where(batch['onset_arr'][i] == 1)[0]
    if len([i for i in onsets if i < 30]) == 0:
        onsets = np.concatenate(([10], onsets))

    ld = generate_loud(onsets)
    pitches = copy.deepcopy(batch['pitches'][i])
    f0 = np.array(p2f(pitches))
    return f0, ld

print('generating heuristic parameters...')
f0_h, ld_h = gen_heuristic(batch, i=i)
heuristic_parameters = {
    'f0_hz': f0_h.astype(np.float32),
    'loudness_db': ld_h.astype(np.float32)
}

# now resynthesize into the audio. this should sound more different.
print('resynthesizing...')

resynth = synthesize_ddsp_audio(model, heuristic_parameters)
wavwrite('test2.wav', 16000, resynth)

# now we take the MIDI for this example and instead of heuristically
# generating f0/loudness curves, we generate them with our best learned
# midi2params model

model_path = '/work/midi2params/model/best_model_cpu_120.pt'
for batch in test_loader:
    break

# load the model
print('loading midi2params model...')
best_model = load_best_model(config, model_path)

# generate the parameters
print('generating the parameters...')
f0_pred, ld_pred = midi2params(best_model, batch)

# now resynthesize with DDSP
for k, arr in batch.items():
    batch[k] = to_numpy(arr)

train_params = {
    'f0_hz': f0_pred[i],
    'loudness_db': ld_pred[i]
}

print('resynthesizing...')
new_model_resynth = synthesize_ddsp_audio(model, train_params)

wavwrite('test3.wav', 16000, new_model_resynth)








