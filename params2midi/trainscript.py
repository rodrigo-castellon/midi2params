"""
Script to train neural net. Here, we aim to learn the mapping
control parameters -> MIDI (fles).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import yaml
import argparse
import os
import pickle
from scipy.io.wavfile import read as wavread, write as wavwrite
from addict import Dict
from datetime import datetime

from train_utils import *
import wavegenie.constants as constants

import wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model.')
    # put in the config path
    DEFAULT_CONFIG_PATH = 'configs/default.yml'
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the train config file')
    parser.add_argument('--cached', action='store_true', help='Use cached dataset instead of loading from the full dataset')

    args = parser.parse_args()

    return args

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(verbose=True):
    args = parse_arguments()

    # get config
    print('getting config')
    with open(args.config) as file:
        config = Dict(yaml.load(file, Loader=yaml.FullLoader))

    # set up wandb config
    wandb.init(project="fles", config=config)

    seed_everything(config.seed)

    # set up a unique identifier for this session
    sess_dt = str(datetime.now()).split('.')[0]
    
    if config.device == 'cuda':
        dtype_flt = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    elif config.device == 'cpu':
        dtype_flt = torch.FloatTensor
        dtype_long = torch.LongTensor
    else:
        raise Exception('Invalid device type')

    # get dataset here
    print('getting dataset')
    set_types = ['train', 'val', 'test']
    if args.cached:
        print('using cached dataset')
        trainset = torch.load(os.path.join(config.dataset.fpath, 'train-cache.pt'))
        valset = torch.load(os.path.join(config.dataset.fpath, 'val-cache.pt'))
        testset = torch.load(os.path.join(config.dataset.fpath, 'test-cache.pt'))
    else:
        print('loading in dataset fully')
        sets = []
        for set_type in set_types:
            print('getting {}'.format(set_type))
            sets.append(load_dataset(set_type, config))
            torch.save(sets[-1], os.path.join(config.dataset.fpath, set_type + '-cache.pt'))
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
    model = model_dict[config.model.id]
    if config.device == 'cuda':
        model = model.cuda()
    # use wandb to watch model
    wandb.watch(model)
    
    # define loss function
    print('defining loss function')
    loss_fn = loss_fn_dict[config.loss_fn]
    
    # define optimizer
    print('defining optimizer')
    optimizer = get_optimizer(model, config)
    
    # set up logging
    os.mkdir(os.path.join(config.log_dir, sess_dt))
    config_save_fpath = os.path.join(config.log_dir, sess_dt, 'config.yml')
    with open(config_save_fpath, 'w') as f:
        yaml.dump(config, f)
    train_loss_fpath = os.path.join(config.log_dir, sess_dt, 'train_loss.txt')
    val_loss_fpath = os.path.join(config.log_dir, sess_dt, 'val_loss.txt')
    best_model_path = os.path.join(config.log_dir, sess_dt, 'best_model.pt')
    
    # training loop
    best_loss = float('inf')
    step = 0
    for epoch in range(config.num_epochs):
        print('Epoch #{}'.format(epoch + 1))

        for i, (f0, loudness_db, conf, pitches, onset_arr, offset_arr) in enumerate(train_loader):
            # process into x and y
            print(f0.shape)
            x, y = torch.LongTensor(x).type(dtype_long), torch.Tensor(y).type(dtype_flt)
            
            x = F.one_hot(x, constants.NUM_MIDI_PITCHES + 1).type(dtype_flt)
            
            x = x.permute(0, 2, 1)
            
            # compare only with the last element of the sequence of y, since
            # we're not parallelizing context-dependent models for now
            y = y[:, -1]
            
            # zero out gradients
            optimizer.zero_grad()

            # compute loss function
            preds = model(x)
            loss = loss_fn(preds, y)
            # backward pass and gradient update
            loss.backward()
            optimizer.step()
            step += 1
            
            # save loss
            wandb.log({'Training loss': loss}, step=step)
            with open(train_loss_fpath, 'a') as f:
                f.write('{:.3g}\n'.format(loss))
            
            if i % 300 == 0:
                print('Iteration #{}'.format(i + 1))
                print('Loss: {}'.format(loss))

        # average validation loss over entire valset
        val_losses = []
        for x, y in val_loader:
            # process x and y
            x, y = torch.LongTensor(x).type(dtype_long), torch.Tensor(y).type(dtype_flt)
            
            x = F.one_hot(x, constants.NUM_MIDI_PITCHES + 1).type(dtype_flt)
            
            x = x.permute(0, 2, 1)
            
            # compare only with the last element of the sequence of y, since
            # we're not parallelizing context-dependent models for now
            y = y[:, -1]

            # compute loss function
            preds = model(x)
            loss = loss_fn(preds, y)
            val_losses.append(loss)

        loss = np.array([float(l) for l in val_losses]).mean()
        # save loss
        wandb.log({'Validation Loss': loss}, step=step)
        with open(val_loss_fpath, 'a') as f:
            f.write('Epoch #{}: {:.3g}\n'.format(epoch, loss))

        if loss < best_loss:
            # save best model
            best_loss = loss
            wandb.run.summary['best_loss'] = best_loss
            torch.save(model.state_dict(), best_model_path)

        print('Validation loss: {:.3g}'.format(loss))

    # TODO: use test set somehow

if __name__ == '__main__':
    main()
