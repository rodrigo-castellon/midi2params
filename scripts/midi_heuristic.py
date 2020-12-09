"""
This file performs heuristic MIDI conversion from audio (or from parameters, if specified)
following the methodology of the Swedish study.

We allow for certain hyperparameter configurations as specified in the arguments to the program.
"""

import numpy as np
import argparse
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from utils.audio_io import load_audio, save_wav
from utils.util import preview_audio, DDSP_DEFAULT_FS_AUDIO
from utils.util import extract_ddsp_synthesis_parameters
from utils.util import load_ddsp_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Heuristic MIDI conversion program.')
    parser.add_argument('--path', '-p', type=str, help='Path to specific MIDI file to convert.')
    parser.add_argument('--start', '-s', type=float, help='Start time of audio clip.')
    parser.add_argument('--end', '-e', type=float, help='End time of audio clip.')
    
    args = parser.parse_args()
    return args

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def enforce_monophonicity(onsets, offsets):
    """
    Enforce monophonicity given onsets and offsets by
    offsetting a note if it is interrupted by another note.
    We assume that onsets and offsets are already sorted.
    """

    # get rid of an offset that is potentially before the first onset
    if offsets[0] < onsets[0]:
        offsets = offsets[1:]
    
    # get rid of an onset that is potentially at the end
    if onsets[-1] > offsets[-1]:
        onsets = onsets[:-1]
    
    onsets, offsets = sorted(list(onsets)), sorted(list(offsets))
    # if there are two offsets in a row, then we place down an onset directly after
    # the first offset
    for initial, final in zip(offsets[:-1], offsets[1:]):
        # if there are no onset such that initial < onset < final, then we execute
        execute = True
        for onset in onsets:
            if initial < onset < final:
                execute = False
                break

        if execute:
            onsets.append(initial + 1)
            onsets = sorted(onsets)
    
    # if there are two onsets in a row, then we place down an offset directly before
    # the second onset
    for initial, final in zip(onsets[:-1], onsets[1:]):
        # if there are no onset such that initial < onset < final, then we execute
        execute = True
        for offset in offsets:
            if initial < offset < final:
                execute = False
                break

        if execute:
            offsets.append(final - 1)
            offsets = sorted(offsets)

    return np.array(onsets), np.array(offsets)

def convert_heuristic(params, FRAME_RATE=250, transform=None):
    """
    Heuristically convert f0/loudness/confidence parameters into
    MIDI (described by discretized f0 and onset/offset arrays). Adapted
    from Jonason et al. (2020; unpublished); main difference is that for
    equality x=0.5 we look instead for 0.4 < x < 0.6, and we also do not
    track whether pitch has shifted at least a semitone for an onset.
    
    Parameters:
    params: control parameters extracted from DDSP
    FRAME_RATE: frame rate for the control parameters
    transform: an element-wise function to apply to the f0_midi array
    """
    
    # construct onset and offset arrays
    f0_conf = np.array(params['f0_confidence']).flatten()
    f0 = np.array(params['f0_hz']).flatten()
    N = f0.shape[0]  # length of parameters
    
    onsets = contiguous_regions((np.gradient(f0_conf) > 0) * (f0_conf < 0.6) * (f0_conf > 0.4))[:,0]
    offsets = contiguous_regions((np.gradient(f0_conf) < 0) * (f0_conf < 0.6) * (f0_conf > 0.4))[:,0]
    
    # enforce monophonicity in the onset and offset arrays
    onsets, offsets = enforce_monophonicity(onsets, offsets)
    
    onset_arr = np.zeros((2 * N), dtype=np.float32)
    onset_arr[onsets] = 1
    offset_arr = np.zeros((2 * N), dtype=np.float32)
    offset_arr[offsets] = 1

    onset_arr = onset_arr[:N]
    offset_arr = offset_arr[:N]
    
    # construct the MIDI-fied f0 signal
    pitches = []
    f0_midi = np.zeros((2 * N), dtype=np.float32)
    for i, onset in enumerate(onsets):
        for offset in offsets:
            if offset > onset:
                break
        else:
            offset = -1
        avg_pitch = f0[onset:offset].mean()
        pitches.append(avg_pitch)
        if i + 1 == len(onsets):
            f0_midi[onset:] = avg_pitch
        else:
            f0_midi[onset:onsets[i+1]] = avg_pitch

    f0_midi = f0_midi[:N]
    pitches = np.array(pitches)
    
    if not(transform is None):
        # apply a transform to our pitches
        pitches = transform(pitches)
    
    hmidi = {
        'f0_midi': f0_midi,
        'onset_arr': onset_arr,
        'offset_arr': offset_arr,
        'onsets': onsets,
        'offsets': offsets,
        'pitches': pitches
    }

    return hmidi


def convert(fpath, start=None, end=None):
    extension = os.path.splitext(args.path)[1]
    if extension == '.wav':
        # load the wav file as audio
        if start is None:
            start = 0
        if end is None:
            audio, fs = load_audio(
                fpath,
                DDSP_DEFAULT_FS_AUDIO,
                num_channels=1,
                normalize=True,
                start_time_seconds=start)
        else:
            audio, fs = load_audio(
                fpath,
                DDSP_DEFAULT_FS_AUDIO,
                num_channels=1,
                normalize=True,
                start_time_seconds=start,
                end_time_seconds=end)

        # now convert the audio into DDSP parameters
        audio_parameters = extract_ddsp_synthesis_parameters(audio)
    elif extension == '.p':
        audio_parameters = pickle.load(open(fpath, 'rb'))
    else:
        raise 'Invalid file.'
    
    print('loaded audio file')
    # now convert these parameters into MIDI!
    print('converting to MIDI')
    f2p = lambda f: 69 + 12 * np.log2(f / 440)
    hmidi = convert_heuristic(audio_parameters, FRAME_RATE=DDSP_DEFAULT_FS_AUDIO, transform=f2p)
    return hmidi
    

def main(args):
    # carry out the conversion
    hmidi = convert(args.path, start=args.start, end=args.end)
    
    # save to parallel folder
    fname = os.path.splitext(os.path.split(args.path)[1])[0] + '.p'
    hmidi_folder = os.path.join(os.path.split(os.path.split(args.path)[0])[0], 'heuristic_midis')
    hmidi_fpath = os.path.join(hmidi_folder, fname)
    
    if not(os.path.exists(hmidi_folder)):
        os.makedirs(hmidi_folder)
    
    pickle.dump(hmidi, open(hmidi_fpath, 'wb'))
    

if __name__ == '__main__':
    args = parse_arguments()
    
    main(args)
    
    
