"""
This file contains training utils for our midi->params inferscript. This file is split up between:
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
from wavegenie.util import to_numpy, p2f, sample_from

import datasets
import models

########
# Data #
########

def load_dataset(set_type, config):
    # set_type is "train", "val", or "test"

    constructor_args = config.dataset.constructor_args
    constructor_args['settype'] = set_type
    constructor_args['config'] = config

    dataset = datasets.MIDIDataset(**constructor_args)
    
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
    Pre-process the batch by (1) producing the teacher forcing labels for input and (2)
    trimming the clips to the shorter length.
        - 'teacher_forcing_f0': could be (N, seq_len) or (N, seq_len, 101)
        - 'teacher_forcing_ld': could be (N, seq_len) or (N, seq_len, 121)
        - 'x': (N, seq_len, {121, 121 + 2, 121 + 101 + 121})
    """

    # now trim
    # we'll exclude 0.2 seconds (50 frames) from each side of the 5 second clip
    # to prevent any wonky stuff
    final_idx = config.frame_rate * config.preprocessing.len_clip
    len_subclip = config.frame_rate * config.preprocessing.len_subclip
    subclip_start = np.random.randint(50, final_idx - 50 - len_subclip)
    for k, arr in batch.items():
        batch[k] = arr[:, subclip_start:subclip_start + len_subclip]
    
    return batch

##########
# Models #
##########

def load_model(config):
    # construct model according to config-specified arguments
    model = models.model_dict[config.model.id](**config.model.constructor_args)
    
    model.load_state_dict(torch.load(config.model.best_path))

    return model


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

def get_predictions(cent_out, ld_out, pitches):
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

def plot_predictions_against_inputs(f0_pred, loudness_pred, pitches, onset_arr, offset_arr):
    """
    Plot model predictions against model inputs in two vertically-aligned plots.
    Return `fig` to allow this to be sent to W&B.
    """
    # convert to numpy arrays if torch tensors
    f0_pred, loudness_pred, pitches, onset_arr, offset_arr = to_numpy(f0_pred), to_numpy(loudness_pred), to_numpy(pitches), to_numpy(onset_arr), to_numpy(offset_arr)

    # fit to the graph
    offset_arr[np.where(offset_arr == 1)] = loudness_pred.max()
    onset_arr[np.where(onset_arr == 1)] = loudness_pred.max()
    offset_arr[np.where(offset_arr == 0)] = loudness_pred.min()
    onset_arr[np.where(onset_arr == 0)] = loudness_pred.min()
    f0_midi = p2f(pitches)

    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches(20, 15)

    axs[0].plot(f0_midi, linewidth=1, label='F0 (MIDI input)')
    axs[0].plot(f0_pred, linewidth=1, label='F0 (predicted)')
    axs[0].set_title('F0 Comparison')
    axs[0].set_xlim(0, 1000)
    axs[0].legend()

    axs[1].plot(onset_arr, label='Onsets')
    axs[1].plot(offset_arr, label='Offsets')
    axs[1].plot(loudness_pred, label='Loudness (predicted)')
    axs[1].set_xlim(0, 1000)
    axs[1].set_title('Loudness Comparison')
    axs[1].legend()
    
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
    # am I missing something, potentially?
    # doesn't reproduce the same results every time
    # TODO: random.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform model inference.')
    DEFAULT_CONFIG_PATH = 'configs/midi2params-infer.yml'
    print('using', DEFAULT_CONFIG_PATH)
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the inference config file')

    args = parser.parse_args()

    return args