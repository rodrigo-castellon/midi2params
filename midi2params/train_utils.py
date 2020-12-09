"""
This file contains training utils for our midi->params trainscript. This file is split up between:
- Datasets
- Models
- Loss Functions
- Optimizers
- Logging and Visualization
- General/Miscellaneous Functions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import random
import os
import matplotlib.pyplot as plt
import argparse
import yaml
import sys
from addict import Dict

from utils.util import to_numpy, p2f, sample_from

import datasets
import models

########
# Data #
########

def load_dataset(set_type, config):
    # set_type is "train", "val", or "test"

    dataset = datasets.MIDIParamsDataset(settype=set_type, config=config)
    
    return dataset

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

def trim_and_preprocess(batch, config):
    """
    Pre-process the batch by (1) producing the teacher forcing labels for input, (2)
    trimming the clips to the shorter length, and (3) standardizing f0 and loudness.
        - 'teacher_forcing_f0': could be (N, seq_len) or (N, seq_len, 101)
        - 'teacher_forcing_ld': could be (N, seq_len) or (N, seq_len, 121)
        - 'x': (N, seq_len, {121, 121 + 2, 121 + 101 + 121})
    """

    # TODO: standardize f0 and loudness if scalar teacher forcing is used.

    # create teacher forcing arrays, which are just copies of the labels at this point.
    # we'll adjust them back one index later.
    if config.training.gaussian_during_train:
        teacher_forcing_f0 = batch['f0_gt_gauss']
        teacher_forcing_ld = batch['ld_gt_gauss']
    else:
        teacher_forcing_f0 = F.one_hot(batch['f0_gt_discrete'], 101).float()
        teacher_forcing_ld = F.one_hot(batch['ld_gt_discrete'], 121).float()

    batch['teacher_forcing_f0'] = teacher_forcing_f0
    batch['teacher_forcing_ld'] = teacher_forcing_ld

    # now trim
    # we'll exclude 0.2 seconds (50 frames) from each side of the 5 second clip
    # to prevent any wonky stuff
    final_idx = config.frame_rate * config.preprocessing.len_clip
    len_subclip = config.frame_rate * config.preprocessing.len_subclip
    subclip_start = np.random.randint(50, final_idx - 50 - len_subclip)
    for k, arr in batch.items():
        if k == 'audio':
            continue
        if 'ld' in k or 'loudness' in k:
            offset = config.preprocessing.offset_ld
        elif 'f0' in k:
            offset = config.preprocessing.offset_f0

        if 'teacher_forcing' in k:
            # we want the labels to be shifted back by one
            offset -= 1

        batch[k] = arr[:, subclip_start + offset:subclip_start + offset + len_subclip]

    x = F.one_hot(batch['pitches'].long(), 129).float()

    # concatenate pitch one-hots to onsets and offsets

    x = torch.cat((x,
                   batch['onset_arr'].unsqueeze(-1),
                   batch['offset_arr'].unsqueeze(-1)), dim=-1)

    # concatenate teacher-forcing labels onto x
    autoregressive_type = config.model.autoregressive_type
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

##########
# Models #
##########

def load_model(config):
    # construct model according to config-specified arguments
    model = models.model_dict[config.model.id](config=config)

    return model

def load_best_model(config, fpath=None):
    # load in the best model
    if fpath is None:
        fpath = config.model.best_path

    model = torch.load(fpath)
    
    if config.device == 'cuda':
        model = model.cuda()

    return model

def midi2params(model, batch):
    """
    Take a batch which contains many MIDI files and turn them into
    DDSP parameters using the inputted model.
    """
    cent_logits, ld_logits, cent_out, ld_out = model.generate(batch)
    
    pitches = batch['pitches']
    
    f0_pred, ld_pred = get_predictions_from_model_outputs(cent_out, ld_out, pitches)
    
    return to_numpy(f0_pred), to_numpy(ld_pred)

##################
# Loss Functions #
##################

# each loss function here expects that the predicted/ground truth
# output takes on the form of a tensor of shape N x seqlen

def regression_loss(pred, y):
    # naive loss function: add up all elementwise
    # l2 distances
    return nn.MSELoss()(pred, y)

def cross_entropy_loss(pred, y):
    """
    Expects two tensors of shape (N, ..., C) and computes cross entropy along
    the last dimension, reducing to a scalar by taking the mean.
    """
    return torch.mean(torch.sum(-nn.LogSoftmax(-1)(pred) * y, dim=-1))

loss_fn_dict = {
    'regression': regression_loss,
    'cross-entropy-1': nn.CrossEntropyLoss(),  # discrete
    'cross-entropy-2': cross_entropy_loss  # soft
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

#############################
# Logging and Visualization #
#############################

def compile_metrics(metrics):
    """
    Compile a metrics dictionary in a way specific to wavegenie. Specifically,
    take the mean of any list and any other object is left the way it is.
    """
    compiled = {}
    for k, obj in metrics.items():
        if type(obj) == type([]):
            compiled[k] = np.array(obj).mean()
        else:  # is matplotlib plot object
            compiled[k] = obj
    metrics = compiled
    return metrics

def get_predictions_from_model_outputs(cent_out, ld_out, pitches):
    """
    Turn the one-hot outputs cent_out and ld_out into continuous scalar signals in time.
    """
    # get a (1250,) array of cents outputted by the model
    if cent_out.is_cuda:
        cent_out = cent_out.cpu()
    if ld_out.is_cuda:
        ld_out = ld_out.cpu()
    if pitches.is_cuda:
        pitches = pitches.cpu()
    
    cent_pred, ld_pred = cent_out.argmax(-1).float() - 50, ld_out.argmax(-1).float() - 120
    
    f0_pred = p2f(pitches.float()) * 2**(cent_pred / 1200)
    
    return to_numpy(f0_pred), to_numpy(ld_pred)  # ensure that we return numpy arrays

def get_predictions(cent_logits, ld_logits, pitches):
    """
    Get raw model predictions as continuous signals for easy comparison and evaluation.
    """
    
    # TODO: adapt for the multi-output model with absolute f0 prediction
    
    # get a (1250,) array of cents outputted by the model
    if cent_logits.is_cuda:
        cent_logits = cent_logits.cpu()
    if ld_logits.is_cuda:
        ld_logits = ld_logits.cpu()
    if pitches.is_cuda:
        pitches = pitches.cpu()

    cents_pred = F.softmax(cent_logits, dim=-1) @ torch.arange(-50, 51).float()
    # compute predicted f0 from these predicted cents
    f0_pred = p2f(pitches.float()) * 2**(cents_pred / 1200)
    
    # compute loudness from our logits
    loudness_pred = F.softmax(ld_logits, dim=-1) @ torch.arange(-120, 1).float()
    
    return to_numpy(f0_pred), to_numpy(loudness_pred)  # ensure that we return numpy arrays

def plot_predictions_against_groundtruths(f0_pred, loudness_pred, f0, loudness):
    """
    Take model predictions and overlay ground truths with them in two vertically-aligned plots.
    Return `plt` to allow this to be sent to W&B.
    """
    # convert to numpy arrays if torch tensors
    f0_pred, loudness_pred, f0, loudness = to_numpy(f0_pred), to_numpy(loudness_pred), to_numpy(f0), to_numpy(loudness)

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(20, 15)

    axs[0].plot(f0, linewidth=1, label='F0 (ground truth)')
    axs[0].plot(f0_pred, linewidth=1, label='F0 (predicted)')
    axs[0].set_title('F0 Comparison')
    axs[0].set_xlim(0, 1000)
    axs[0].legend()

    axs[1].plot(loudness, label='Loudness (ground truth)')
    axs[1].plot(loudness_pred, label='Loudness (predicted)')
    axs[1].set_xlim(0, 1000)
    axs[1].set_title('Loudness Comparison')
    axs[1].legend()
    
    #np.save('f0_pred.npy', f0_pred)
    #np.save('f0.npy', f0)
    #fig.savefig('example_fig.png')
    
    return fig

def plot_probdist_prediction(ld_logits):
    ld_probs = F.softmax(ld_logits, dim=-1)
    fig, ax = plt.subplots()
    ax.imshow(to_numpy(ld_probs).T, origin='lower')
    ax.set_xlim(0, 1000)
    ax.set_title('Loudness Probability Distribution')
    
    return fig

###################################
# General/Miscellaneous Functions #
###################################

def seed_everything(seed):
    # seed torch, numpy, etc.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model.')
    # put in the config path
    DEFAULT_CONFIG_PATH = 'midi2params/configs/midi2params.yml'
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the train config file')

    args = parser.parse_args()

    return args

def parse_custom_arguments():
    args = [arg[2:].split('=') for arg in sys.argv[1:]]
    return args

def load_config(cfg_path):
    """
    Load a nested configuration a la hydra.
    """

    with open(cfg_path) as file:
        config = Dict(yaml.load(file, Loader=yaml.FullLoader))

    # now go through each subconfig and add it into our main config
    cfg_folder = os.path.split(cfg_path)[0]
    for subconfig_dict in config.subconfigs:
        folder, subconfig_name = [i for i in subconfig_dict.items()][0]
        subfolder = os.path.join(cfg_folder, folder)
        matches = [f for f in os.listdir(subfolder) if subconfig_name in f]
        if len(matches) >= 1:
            subconfig_path = os.path.join(subfolder, matches[0])
        with open(subconfig_path) as file:
            subconfig = Dict(yaml.load(file, Loader=yaml.FullLoader))
        config[folder] = subconfig

    return config
    
