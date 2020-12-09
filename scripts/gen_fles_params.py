"""
Run gen_fles_midis.py BEFORE this script. This takes all MIDIs in the midi folder under
the target directory, synthesizes them, extracts their DDSP parameters, and places
those parameters in a parallel folder under the target directory.
"""

print('importing dependencies')
import itertools
import os
import pretty_midi
import random
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import copy
import argparse
import multiprocessing
print('finished importing standard libraries')

#print('importing wavegenie code')
from utils.util import DDSP_DEFAULT_FS_AUDIO
from utils.util import extract_ddsp_synthesis_parameters
#print('finished importing wavegenie code')

# some global variables
lock = multiprocessing.Lock()

pretty_midi.pretty_midi.MAX_TICK = 1e16
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MOD_MIDI_PATH, PARAMS_PATH = 'modified_midis', 'synth_params'

LEN_CLIP = 5
VALID_PROGRAM_NUMBERS = list(range(128))
INVALID_PROGRAM_NUMBERS = [8, 9, 10, 13, 14, 16, 17, 18, 19, 21, 31,
                           47, 48, 49, 55, 78, 86, 87, 96, 97, 98, 101,
                           112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                           122, 123, 124, 125, 126, 127]
for invalid in INVALID_PROGRAM_NUMBERS:
    VALID_PROGRAM_NUMBERS.remove(invalid)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate fles training dataset randomly (or add to a pre-existing one).')
    parser.add_argument('--force', '-f', action='store_true', help='Process a MIDI file even if it has already been processed.')
    parser.add_argument('--worker', '-w', type=int, default=0, help='Worker ID.')
    parser.add_argument('--total', '-t', type=int, default=1, help='Total number of workers running.')
    parser.add_argument('--targetpath', '-p', type=str, default='fles_dataset', help='Path to the target dataset path')

    args = parser.parse_args()

    return args

def instrument_is_monophonic(ins):
    # Ensure sorted
    ins.notes = sorted(ins.notes, key=lambda x: x.start)
    notes = ins.notes
    last_note_start = -1
    for n in notes:
        assert n.start >= last_note_start
        last_note_start = n.start

    monophonic = True
    for i in range(len(notes) - 1):
        n0 = notes[i]
        n1 = notes[i + 1]
        if n0.end > n1.start:
            monophonic = False
            break
    return monophonic

def process_and_save_clip(clipmidi_path, tgtpath):
    #try:
    global lock, VALID_PROGRAM_NUMBERS

    print('PROCESSING {}'.format(clipmidi_path))

    # load in the clip
    clipmidi = pretty_midi.PrettyMIDI(clipmidi_path)

    assert len(clipmidi.instruments) == 1  # only one instrument
    assert instrument_is_monophonic(clipmidi.instruments[0])  # monophonic
    assert clipmidi.instruments[0].program in VALID_PROGRAM_NUMBERS  # in our valid program numbers
    assert len(clipmidi.instruments[0].notes) >= 1  # has at least one note
    assert len([n.pitch for n in clipmidi.instruments[0].notes if n.pitch > 95 or n.pitch < 24]) == 0  # within CREPE range
    assert not(clipmidi.instruments[0].is_drum)  # not a drum

    # synthesize and extract parameters
    print('synthesizing...')
    synthed = clipmidi.fluidsynth(fs=16000)[..., np.newaxis].astype('float32')
    print(synthed.shape)
    lock.acquire()
    print('extracting DDSP parameters...')
    synth_params = extract_ddsp_synthesis_parameters(synthed)
    lock.release()
    print('done extracting synthesis parameters')

    # clipmidi_path has an extension and unnecessary folders. Remove them
    file_id = os.path.split(os.path.splitext(clipmidi_path)[0])[1]
    # file_id is now something like 'lakh_hash-44-0'
    # Now we append the extension

    basepath = PARAMS_PATH
    ext = '.p'
    fname = file_id + ext
    out_path = os.path.join(tgtpath, basepath, fname)

    deepest_folder = os.path.join(tgtpath, basepath)
    if not(os.path.exists(deepest_folder)):
        os.makedirs(deepest_folder)

    pickle.dump(synth_params, open(out_path, 'wb'))
    #except Exception as e:
    #    print('ERROR')
    #    print(e)


if __name__ == '__main__':
    print('parsing arguments')
    args = parse_arguments()

    if not(os.path.exists(args.targetpath)):
        os.makedirs(args.targetpath)

    # get MIDI file paths
    print('getting MIDI file paths')
    midi_fps = sorted(glob.glob(os.path.join(args.targetpath, MOD_MIDI_PATH, '*.mid*')))

    # split up based on worker ID
    midi_fps = midi_fps[args.worker::args.total]
    for i, midi_fp in enumerate(midi_fps):
        print('midi file #{}/{}'.format(i, len(midi_fps)))
        p = multiprocessing.Process(target=process_and_save_clip, args=(midi_fp, args.targetpath))
        p.start()
        p.join()
