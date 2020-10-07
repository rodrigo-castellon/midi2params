"""
Script to perform inference (generation) given a trained midi2params model.
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

from infer_utils import *

def main(verbose=True):
    args = parse_arguments()

    # get config
    print('getting config')
    with open(args.config) as file:
        config = Dict(yaml.load(file, Loader=yaml.FullLoader))

    # override if we just don't have a GPU
    if not(torch.cuda.is_available()) and config.device == 'cuda':
        config.device = 'cpu'

    # set up wandb config
    wandb.init(project='midi2params-inference', config=config)

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

    # set up logging
    os.mkdir(os.path.join(config.logging.log_dir, sess_dt))
    config_save_fpath = os.path.join(config.logging.log_dir, sess_dt, 'config.yml')
    with open(config_save_fpath, 'w') as f:
        yaml.dump(config, f)
    metrics = {}

    model.eval()
    example_save_count = 0
    for i, batch in enumerate(val_loader):
        print('batch #{}'.format(i))
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

        cent_logits, ld_logits, cent_out, ld_out = model.generate(batch)
        
        ##################
        # HANDLE METRICS #
        ##################

        # save all examples
        num_examples = 5
        for idx in range(num_examples):
            print('plotting plot #{}'.format(example_save_count))
            pitches, onset_arr, offset_arr = batch['pitches'], batch['onset_arr'], batch['offset_arr']
            pitches, onset_arr, offset_arr = pitches[idx].float(), onset_arr[idx].float(), offset_arr[idx].float()
            cent_logits_ = cent_logits[idx]
            ld_logits_ = ld_logits[idx]
            f0_pred, loudness_pred = get_predictions(cent_logits_, ld_logits_, pitches)
            # now plot these
            fig = plot_predictions_against_inputs(f0_pred, loudness_pred, pitches, onset_arr, offset_arr)
            fig2 = plot_probdist_prediction(ld_logits_)
            # give plt to wandb
            print('saving to {}'.format(os.path.join(config.logging.log_dir, sess_dt)))
            fig.savefig(os.path.join(config.logging.log_dir, sess_dt, 'pred_plot_f0_{}.png'.format(example_save_count)))
            fig2.savefig(os.path.join(config.logging.log_dir, sess_dt, 'pred_plot_ld_{}.png'.format(example_save_count)))
            np.save(os.path.join(config.logging.log_dir, sess_dt, 'f0_pred_{}.npy'.format(example_save_count)),
                    f0_pred)
            np.save(os.path.join(config.logging.log_dir, sess_dt, 'ld_pred_{}.npy'.format(example_save_count)),
                    loudness_pred)
            metrics['pred_plot {}'.format(example_save_count)] = fig#wandb.Image(fig)
            metrics['pred_plot_ld {}'.format(example_save_count)] = fig2#wandb.Image(fig2)
            example_save_count += 1

            if example_save_count % 5 == 0:
                print('logging to wandb')
                wandb.log(metrics, step=1)
                metrics = {}


if __name__ == '__main__':
    main()