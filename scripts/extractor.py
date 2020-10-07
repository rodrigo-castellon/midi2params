"""
Extract parameters from all WAV files in a folder and place those pickled audio parameters
in an adjacent folder. The source folder can contain subfolders.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pickle
import tqdm
import multiprocessing
import argparse
from scipy.io.wavfile import read as wavread
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress annoying TF warnings

from wavegenie.audio_io import load_audio, save_wav
from wavegenie.util import preview_audio, DDSP_DEFAULT_FS_AUDIO
from wavegenie.util import extract_ddsp_synthesis_parameters

# name of parallel folder holding the parameters we are extracting
BASE_FOLDER_NAME = 'params'

def parse_arguments():
    desc = 'Extract parameters from all WAV files in a folder and place those pickled audio parameters in an adjacent folder.'
    parser = argparse.ArgumentParser(description=desc)
    # put in the config path
    parser.add_argument('--worker', '-w', type=int, default=0, help='Worker ID.')
    parser.add_argument('--total', '-t', type=int, default=1, help='Total number of workers.')
    parser.add_argument('--path', '-p', type=str, default='data/DDSPViolin-clipped2/wav', help='Path to folder; can contain subfolders as long as every file (no matter how deep) is a WAV file.')

    args = parser.parse_args()

    return args

def extract(fpath, basepath, force=False):
    # given an explicit file path and an explicit base folder path (that points to the
    # base folder where fpath resides in), save parameters to equivalent
    # parameters file path
    
    # Parameters:
    # fpath: explicit path to file
    # basepath: explicit path to base folder
    # force: should force extraction if the file has already been extracted in the past?
    # verbose: verbose

    pure_fpath = os.path.splitext(fpath)[0]
    relpath = os.path.relpath(pure_fpath, basepath)  # how to get from basepath to the specific file
    # switch basepath from synths folder to the parameters folder
    basepath = os.path.join(os.path.split(basepath)[0], BASE_FOLDER_NAME)
    parameter_path = os.path.join(basepath, relpath) + '.p'

    if os.path.exists(parameter_path) and not(force):
        return
    
    # load audio
    _, audio = wavread(fpath)
    start = time.time()
    audio_parameters = extract_ddsp_synthesis_parameters(audio[np.newaxis, ...])
    print('took {:.3g} seconds'.format(time.time() - start))

    # build up the file tree up to this point, if it doesn't exist yet
    deepest_folder = os.path.split(parameter_path)[0]
    print('writing to', parameter_path)
    try:
        os.makedirs(deepest_folder)
    except FileExistsError:
        pass

    pickle.dump(audio_parameters, open(parameter_path, 'wb'))
        
if __name__ == '__main__':
    print('starting')
    args = parse_arguments()
    i, skip_every, FOLDER_PATH = args.worker, args.total, args.path
    
    # get all files that we need to convert
    allfiles = []
    for (dirpath, dirnames, filenames) in os.walk(FOLDER_PATH):
        allfiles += [os.path.join(dirpath, file) for file in filenames]
    fl = sorted(allfiles)
    
    fl = fl[i::skip_every]
    for idx, file in enumerate(fl):
        print('worker #{}: {} / {}'.format(i, idx, len(fl)))
        print(file, FOLDER_PATH)
        # NOTE: The extract function is stateful but here we want stateless behavior and
        # multiprocessing.Process is a quick way of achieving this
        p = multiprocessing.Process(target=extract, args=(file, FOLDER_PATH))
        p.start()
        p.join()
