"""
Script to train neural net. Here, we aim to learn the mapping
MIDI -> control parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import yaml
import pickle
from addict import Dict
from datetime import datetime
import wandb
import time
import copy

from train_utils import *

DEFAULT_CONFIG_PATH = 'midi2params/configs/midi2params-best.yml'

def main(verbose=True):
    args = parse_custom_arguments()

    # get config
    print('getting config')
    config = load_config(DEFAULT_CONFIG_PATH)
    
    # override stuff
    for key, value in args:
        # key will be something like model.autoregressive_type
        # value will be some kind of primitive type
        value = value.strip().replace('\'', '').replace('"','')

        if value in ['True', 'False', 'true', 'false']:
            value = True if value == 'True' else False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:  # value is a string
                pass

        subconfig = config
        for keyval in key.split('.')[:-1]:
            subconfig = subconfig[keyval]
        subconfig[key.split('.')[-1]] = value

    if not(torch.cuda.is_available()) and config.device == 'cuda':
        config.device = 'cpu'

    # set up wandb config
    wandb.init(project='midi2params', config=config)

    seed_everything(config.seed)

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

    # start watching the model with wandb
    wandb.watch(model)
    
    # define loss function
    print('defining loss function')
    loss_fn_discrete = loss_fn_dict['cross-entropy-1']
    loss_fn_soft = loss_fn_dict['cross-entropy-2']
    
    # define optimizer
    print('defining optimizer')
    optimizer = get_optimizer(model, config)

    # set up logging
    log_dir = wandb.run.dir
    print('logging everything to', log_dir)
    config_save_fpath = os.path.join(log_dir, 'config.yml')
    with open(config_save_fpath, 'w') as f:
        yaml.dump(config, f)
    train_loss_fpath = os.path.join(log_dir, 'train_loss.txt')
    val_loss_fpath = os.path.join(log_dir, 'val_loss.txt')
    best_model_path = os.path.join(log_dir, 'best_model.pt')
    metrics_to_track = ['loudness_loss_discrete', 'cents_loss_discrete',
                        'total_loss_discrete', 'infer_time']

    if config.training.gaussian_during_train:
        metrics_to_trick += ['loudness_loss_gauss', 'cents_loss_gauss', 'total_loss_gauss']
    
    if config.logging.log_perplexity:
        metrics_to_track += ['loudness_perp_discrete', 'cents_perp_discrete',
                             'loudness_perp_gauss', 'cents_perp_gauss']
    
    # training loop
    best_loss = float('inf')
    step = 0
    halt_training = False
    for epoch in range(config.training.num_epochs):
        print('Epoch #{}'.format(epoch + 1))

        train_metrics = {'train_' + k: [] for k in metrics_to_track}
        train_metrics['train_backward_time'] = []

        model.train()  # doesn't make a difference but good practice
        train_start = time.time()
        for i, batch in enumerate(train_loader):
            ########################
            # MOVE BATCH TO DEVICE #
            ########################

            for k, arr in batch.items():
                if arr.type() == 'torch.FloatTensor':
                    dtype = dtype_float
                elif arr.type() == 'torch.LongTensor':
                    dtype = dtype_long

                batch[k] = arr.type(dtype)

            #######################
            # TRIM AND PREPROCESS #
            #######################

            batch = trim_and_preprocess(batch, config)

            #####################
            # GET MODEL OUTPUTS #
            #####################

            infer_start = time.time()
            cent_logits, ld_logits = model(batch)
            infer_time = time.time() - infer_start

            ##############
            # CENTS LOSS #
            ##############

            y_pred = cent_logits

            if config.training.gaussian_during_train:
                cents_loss_gauss = loss_fn_soft(y_pred, batch['f0_gt_gauss'])

            y_pred = y_pred.view(-1, 101)
            batch['f0_gt_discrete'] = batch['f0_gt_discrete'].reshape(-1)

            cents_loss_discrete = loss_fn_discrete(y_pred, batch['f0_gt_discrete'])

            #################
            # LOUDNESS LOSS #
            #################

            y_pred = ld_logits

            if config.training.gaussian_during_train:
                ld_loss_gauss = loss_fn_soft(y_pred, batch['ld_gt_gauss'])

            y_pred = y_pred.view(-1, 121)
            batch['ld_gt_discrete'] = batch['ld_gt_discrete'].reshape(-1)

            ld_loss_discrete = loss_fn_discrete(y_pred, batch['ld_gt_discrete'])

            ###################################
            # COMPUTE TOTAL LOSS AND BACKPROP #
            ###################################

            loss_discrete = cents_loss_discrete + ld_loss_discrete
            if config.training.gaussian_during_train:
                loss_gauss = cents_loss_gauss + ld_loss_gauss
                loss = loss_gauss
            else:
                loss = loss_discrete

            # halt train if we get a NaN loss
            if torch.isnan(loss) or loss > 1e2:
                halt_training = True
                break

            optimizer.zero_grad()
            backward_start = time.time()
            if config.training.train_only_on_loudness:
                if config.training.gaussian_during_train:
                    loss = ld_loss_gauss
                else:
                    loss = ld_loss_discrete

            loss.backward()
            backward_time = time.time() - backward_start
            optimizer.step()
            step += 1

            ##################
            # HANDLE METRICS #
            ##################
            ld_perp_discrete = np.exp(float(ld_loss_discrete))  # perplexities
            cents_perp_discrete = np.exp(float(cents_loss_discrete))
            if config.training.gaussian_during_train:
                ld_perp_gauss = np.exp(float(ld_loss_gauss))
                cents_perp_gauss = np.exp(float(cents_loss_gauss))

            # losses
            train_metrics['train_loudness_loss_discrete'].append(float(ld_loss_discrete))
            train_metrics['train_cents_loss_discrete'].append(float(cents_loss_discrete))
            train_metrics['train_total_loss_discrete'].append(float(loss_discrete))
            if config.training.gaussian_during_train:
                train_metrics['train_loudness_loss_gauss'].append(float(ld_loss_gauss))
                train_metrics['train_cents_loss_gauss'].append(float(cents_loss_gauss))
                train_metrics['train_total_loss_gauss'].append(float(loss_gauss))

            # perplexities
            if config.logging.log_perplexity:
                train_metrics['train_loudness_perp_discrete'].append(ld_perp_discrete)
                train_metrics['train_cents_perp_discrete'].append(cents_perp_discrete)
                if config.training.gaussian_during_train:
                    train_metrics['train_loudness_perp_gauss'].append(ld_perp_gauss)
                    train_metrics['train_cents_perp_gauss'].append(cents_perp_gauss)

            # timing
            train_metrics['train_backward_time'].append(backward_time)
            train_metrics['train_infer_time'].append(infer_time)

        if halt_training:
            break
        # compile metrics
        train_time = time.time() - train_start
        train_metrics = compile_metrics(train_metrics)
        train_metrics['epoch'] = epoch
        train_metrics['train_total_time'] = train_time
        # save loss
        wandb.log(train_metrics, step=step)
        with open(train_loss_fpath, 'a') as f:
            f.write('{:.3g}\n'.format(train_metrics['train_total_loss_discrete']))

        print('Train Loss: {:.3g}'.format(train_metrics['train_total_loss_discrete']))

        # average validation loss over entire valset
        val_metrics = {'val_' + k: [] for k in metrics_to_track}
        model.eval()
        for i, batch in enumerate(val_loader):
            ########################
            # MOVE BATCH TO DEVICE #
            ########################

            for k, arr in batch.items():
                if arr.type() == 'torch.FloatTensor':
                    dtype = dtype_float
                elif arr.type() == 'torch.LongTensor':
                    dtype = dtype_long
                
                batch[k] = arr.type(dtype)

            #######################
            # TRIM AND PREPROCESS #
            #######################

            batch = trim_and_preprocess(batch, config)

            #for k, arr in batch.items():
            #    batch[k] = arr[1:]

            #####################
            # GET MODEL OUTPUTS #
            #####################

            infer_start = time.time()
            cent_logits, ld_logits = model(batch)
            infer_time = time.time() - infer_start

            ##############
            # CENTS LOSS #
            ##############

            y_pred = cent_logits

            if config.training.gaussian_during_train:
                cents_loss_gauss = loss_fn_soft(y_pred, batch['f0_gt_gauss'])

            y_pred = y_pred.view(-1, 101)
            batch['f0_gt_discrete'] = batch['f0_gt_discrete'].reshape(-1)

            cents_loss_discrete = loss_fn_discrete(y_pred, batch['f0_gt_discrete'])

            #################
            # LOUDNESS LOSS #
            #################

            y_pred = ld_logits

            if config.training.gaussian_during_train:
                ld_loss_gauss = loss_fn_soft(y_pred, batch['ld_gt_gauss'])

            y_pred = y_pred.view(-1, 121)
            batch['ld_gt_discrete'] = batch['ld_gt_discrete'].reshape(-1)

            ld_loss_discrete = loss_fn_discrete(y_pred, batch['ld_gt_discrete'])

            ###################################
            # COMPUTE TOTAL LOSS AND BACKPROP #
            ###################################

            loss_discrete = cents_loss_discrete + ld_loss_discrete
            if config.training.gaussian_during_train:
                loss_gauss = cents_loss_gauss + ld_loss_gauss
                loss = loss_gauss
            else:
                loss = loss_discrete
            
            ##################
            # HANDLE METRICS #
            ##################
            ld_perp_discrete = np.exp(float(ld_loss_discrete))  # perplexities
            cents_perp_discrete = np.exp(float(cents_loss_discrete))
            if config.training.gaussian_during_train:
                ld_perp_gauss = np.exp(float(ld_loss_gauss))
                cents_perp_gauss = np.exp(float(cents_loss_gauss))

            # losses
            val_metrics['val_loudness_loss_discrete'].append(float(ld_loss_discrete))
            val_metrics['val_cents_loss_discrete'].append(float(cents_loss_discrete))
            val_metrics['val_total_loss_discrete'].append(float(loss_discrete))
            if config.training.gaussian_during_train:
                val_metrics['val_loudness_loss_gauss'].append(float(ld_loss_gauss))
                val_metrics['val_cents_loss_gauss'].append(float(cents_loss_gauss))
                val_metrics['val_total_loss_gauss'].append(float(loss_gauss))
            
            # perplexities
            if config.log_perplexity:
                val_metrics['val_loudness_perp_discrete'].append(ld_perp_discrete)
                val_metrics['val_cents_perp_discrete'].append(cents_perp_discrete)
                if config.training.gaussian_during_train:
                    val_metrics['val_loudness_perp_gauss'].append(ld_perp_gauss)
                    val_metrics['val_cents_perp_gauss'].append(cents_perp_gauss)
            
            # timing
            val_metrics['val_infer_time'].append(infer_time)

            # save n prediction examples if we're on the right epoch
            if epoch % config.logging.example_saving.save_every == 0 and i == 0:
                idx_start = config.logging.example_saving.i
                for idx in range(idx_start, idx_start + config.logging.example_saving.num_examples):
                    f0, loudness_db, pitches = batch['f0'], batch['loudness_db'], batch['pitches']
                    pitches = pitches[idx].float()
                    cent_logits_ = cent_logits[idx]
                    ld_logits_ = ld_logits[idx]
                    f0_pred, loudness_pred = get_predictions(cent_logits_, ld_logits_, pitches)
                    # now plot these
                    fig = plot_predictions_against_groundtruths(f0_pred, loudness_pred, f0[idx], loudness_db[idx])
                    fig2 = plot_probdist_prediction(ld_logits_)
                    # give plt to wandb
                    val_metrics['val_pred_plot {}'.format(idx)] = fig
                    val_metrics['val_pred_plot_ld {}'.format(idx)] = fig2


        # compile metrics
        val_metrics_compiled = {}
        for k, obj in val_metrics.items():
            if type(obj) == type([]):
                val_metrics_compiled[k] = np.array(obj).mean()
            else:  # is matplotlib plot object
                val_metrics_compiled[k] = obj
        val_metrics_compiled['epoch'] = epoch
        val_metrics = val_metrics_compiled
        # save loss
        wandb.log(val_metrics, step=step)
        with open(val_loss_fpath, 'a') as f:
            f.write('Epoch #{}: {:.3g}\n'.format(epoch, val_metrics['val_total_loss_discrete']))

        if loss < best_loss:
            # save best model
            best_loss = val_metrics['val_total_loss_discrete']
            wandb.run.summary['best_val_total_loss_discrete'] = best_loss
            torch.save(model, best_model_path)

        print('Validation loss: {:.3g}'.format(loss))

    # TODO: use test set somehow

if __name__ == '__main__':
    main()
