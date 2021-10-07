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

from train_utils import *

args = parse_arguments()

args.config = '/workspace/midi2params/midi2params/configs/midi2params-test.yml'

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


# extract the f0/loudness features/parameters with DDSP

# now load the DDSP model

# now resynthesize the same audio, should sound similar

# now we take the MIDI for this example and heuristically generate
# reasonable f0/loudness curves via heuristics


# now resynthesize into the audio. this should sound more different.




# now we take the MIDI for this example and instead of heuristically
# generating f0/loudness curves, we generate them with our best learned
# midi2params model

# load the model

# convert to CUDA if possible

# generate the parameters


# now resynthesize with DDSP













