"""
Say at `dset_path` we have subfolders `dset_path/folder1`, `dset_path/folder2`, etc. full with paired training data. We want to do a train/val/test split in parallel in these folders.
"""

import os
import shutil
import numpy as np
import random
import sys
import copy
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Do parallel train/val/test split.')
    parser.add_argument('--path', required=True, type=str, help='dataset path')
    parser.add_argument('--train', default=0.8, type=float, help='train split (float)')
    parser.add_argument('--val', default=0.1, type=float, help='val split (float)')
    parser.add_argument('--fles', action='store_true', help='split based on program number (third identifier in files)')
    parser.add_argument('--dry', action='store_true', help='dry run')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose output')
    
    args = parser.parse_args()
    
    return args

def get_fids(path):
    # get the file ids for a folder path `path`
    return [os.path.splitext(f)[0] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def parallel_split(dset_path, train=0.8, val=0.1, fles=False, dry=False, verbose=False):
    """
    Perform a train/val/test split. We assume that dset_path contains
    paired files in dset_path/folder1, dset_path/folder2, etc. Each subfolder
    contains identical file names but with potentially different extensions.
    """
    assert train + val <= 1, 'invalid split'
    # get parallel folders first
    
    folders = [f for f in os.listdir(dset_path) if os.path.isdir(os.path.join(dset_path, f))]

    # ensure that the folders have the same exact file ids in them
    list_of_file_ids = [get_fids(os.path.join(dset_path, folder)) for folder in folders]
    fileid_hashes = [hash(str(sorted(ids))) for ids in list_of_file_ids]
    # converting to set removes duplicates; all of them should be
    assert len(set(fileid_hashes)) == 1, 'Ensure that folders in this directory are structured identically'
    
    exts = [os.path.splitext(os.listdir(os.path.join(dset_path, folder))[0])[1] for folder in folders]
    
    # get a definitive list of file ids and perform train/val/test split on them
    file_ids = list_of_file_ids[0]
    
    # get sizes of each split
    len_train = int(train * len(file_ids))
    len_val = int(val * len(file_ids))
    len_test = len(file_ids) - len_train - len_val

    if fles:
        # shuffle around the file ids such that we have homogeneous program numbers in train, val, and test splits
        program_num_list = [int(f.split('-')[-1]) for f in file_ids]
        program_nums = list(set(program_num_list))
        counts = {num: program_num_list.count(num) for num in program_nums}
        cumulative_count = 0
        # collect program numbers for each split
        train_nums, val_nums, test_nums = [], [], []
        for num in sorted(program_nums, key=lambda x:np.random.rand()):
            cumulative_count += counts[num]
            if cumulative_count < len_train:
                train_nums.append(num)
            elif cumulative_count < len_train + len_val:
                val_nums.append(num)
            else:
                test_nums.append(num)
        
        # now split up file_ids based on program numbers
        train = [f for f in file_ids if int(f.split('-')[-1]) in train_nums]
        val = [f for f in file_ids if int(f.split('-')[-1]) in val_nums]
        test = [f for f in file_ids if int(f.split('-')[-1]) in test_nums]
    else:
        np.random.shuffle(file_ids)
        train, val, test = file_ids[:len_train], file_ids[len_train:len_train + len_val], file_ids[-len_test:]
    

    # now move all of the files to their respective split folders, for each folder
    print('computed split')
    print('moving files around now')
    for folder, ext in zip(folders, exts):
        folder_path = os.path.join(dset_path, folder)
        for split in ['train', 'val', 'test']:
            out_str = 'creating directory ' + os.path.join(folder_path, split)
            print(out_str)
            if not(dry):
                os.mkdir(os.path.join(folder_path, split))
        for train_id in train:
            if verbose:
                out_str = 'moving {} to {}'.format(os.path.join(folder_path, train_id + ext),
                                                         os.path.join(folder_path, 'train', train_id + ext))
                print(out_str)
            if not(dry):
                shutil.move(os.path.join(folder_path, train_id + ext),
                            os.path.join(folder_path, 'train', train_id + ext))
        for val_id in val:
            if verbose:
                out_str = 'moving {} to {}'.format(os.path.join(folder_path, val_id + ext),
                                                   os.path.join(folder_path, 'val', val_id + ext))
                if dry:
                    out_str = '(dry) ' + out_str
                print(out_str)
            if not(dry):
                shutil.move(os.path.join(folder_path, val_id + ext),
                            os.path.join(folder_path, 'val', val_id + ext))
        for test_id in test:
            if verbose:
                out_str = 'moving {} to {}'.format(os.path.join(folder_path, test_id + ext),
                                                   os.path.join(folder_path, 'test', test_id + ext))
                print(out_str)
            if not(dry):
                shutil.move(os.path.join(folder_path, test_id + ext),
                            os.path.join(folder_path, 'test', test_id + ext))


if __name__ == '__main__':
    args = parse_arguments()
    print('running parallel split')
    parallel_split(args.path, train=args.train, val=args.val, fles=args.fles, dry=args.dry, verbose=args.verbose)