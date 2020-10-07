"""
This script is intended to generate the MIDIs for fles training from a source Lakh dataset. Stage 1 of 2 for generating fles dataset.
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

pretty_midi.pretty_midi.MAX_TICK = 1e16
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MOD_MIDI_PATH, PARAMS_PATH = 'modified_midis', 'synth_params'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate fles training dataset randomly (or add to a pre-existing one).')
    parser.add_argument('--num', '-n', type=int, default=10, help='Number of files to use')
    parser.add_argument('--sourcepath', '-s', type=str, default='dataset', help='Path to the source Lakh dataset (as is after extracting tar file)')
    parser.add_argument('--targetpath', '-t', type=str, default='fles_dataset', help='Path to the target dataset path')

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


def check_midi(
        midi_fp,
        min_num_instruments=1,
        filter_mid_len_below_seconds=5.,
        filter_mid_len_above_seconds=600.,
        filter_mid_bad_times=True,
        filter_ins_max_below=24,
        filter_ins_min_above=95,
        filter_ins_duplicate=True):

    if min_num_instruments <= 0:
        raise ValueError()

    # Ignore unusually large MIDI files (only ~25 of these in the dataset)
    if os.path.getsize(midi_fp) > (512 * 1024): #512K
        print('too large')
        return False, None

    try:
        midi = pretty_midi.PrettyMIDI(midi_fp)
    except:
        print('improper prettymidi load')
        return False, None
    
    #########################
    # Filtering Instruments #
    #########################

    # Filter out drum instruments
    midi.instruments = [i for i in midi.instruments if not i.is_drum]

    # Filter out duplicate instruments
    if filter_ins_duplicate:
        uniques = set()
        instruments_unique = []
        for ins in midi.instruments:
            pitches = ','.join(['{}:{:.1f}'.format(str(n.pitch), n.start) for n in ins.notes])
            if pitches not in uniques:
                instruments_unique.append(ins)
                uniques.add(pitches)
                midi.instruments = instruments_unique
        if len(midi.instruments) < min_num_instruments:
            return False, None

    # Filter out instruments with bizarre ranges
    instruments_normal_range = []
    for ins in midi.instruments:
        pitches = [n.pitch for n in ins.notes]
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        if max_pitch >= filter_ins_max_below and min_pitch <= filter_ins_min_above:
            instruments_normal_range.append(ins)
    midi.instruments = instruments_normal_range
    
    if len(midi.instruments) < min_num_instruments:
        return False, None

    # Filter out polyphonic instruments
    midi.instruments = [i for i in midi.instruments if instrument_is_monophonic(i)]

    if len(midi.instruments) < min_num_instruments:
        return False, None

    # Filter out instruments with no velocity variation
    instruments_with_variation = []
    for ins in midi.instruments:
        vs = set([note.velocity for note in ins.notes])
        if len(vs) > 1:
            instruments_with_variation.append(ins)
    midi.instruments = instruments_with_variation

    if len(midi.instruments) < min_num_instruments:
        return False, None

    ########################
    # Filtering MIDI Files #
    ########################

    # Filter MIDIs with extreme length
    midi_len = midi.get_end_time()
    if midi_len < filter_mid_len_below_seconds or midi_len > filter_mid_len_above_seconds:
        return False, None

    # Filter out negative times and quantize to audio samples
    for ins in midi.instruments:
        for n in ins.notes:
            if filter_mid_bad_times:
                if n.start < 0 or n.end < 0 or n.end < n.start:
                    return False
            n.start = round(n.start * 44100.) / 44100.
            n.end = round(n.end * 44100.) / 44100.

    return True, midi

def process_and_save_clip(clipmidi, tgtpath, relpath, ins_num, ins, start_time, VALID_PROGRAM_NUMBERS):
    #try:
    clipmidi.instruments = [ins]
    if len(clipmidi.instruments[0].notes) == 0:
        return

    clipmidi.instruments[0].notes = sorted(clipmidi.instruments[0].notes, key=lambda x: x.start)
    # create new midi object where we just transfer over the notes
    # we do this to remove pitch bending or other unwanted MIDI artifacts
    print('creating reduced clipmidi')
    clipmidi_reduced = pretty_midi.PrettyMIDI()
    ins_reduced = pretty_midi.Instrument(program=np.random.choice(VALID_PROGRAM_NUMBERS))
    for note in clipmidi.instruments[0].notes:
        note_reduced = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=note.start, end=note.end)
        ins_reduced.notes.append(note_reduced)
    clipmidi_reduced.instruments.append(ins_reduced)

    assert len(clipmidi.instruments) == len(clipmidi_reduced.instruments) == 1
    assert len(clipmidi.instruments[0].notes) == len(clipmidi_reduced.instruments[0].notes)
    assert hash(tuple([n.pitch for n in clipmidi.instruments[0].notes])) == hash(tuple([n.pitch for n in clipmidi_reduced.instruments[0].notes]))
    assert hash(tuple([n.velocity for n in clipmidi.instruments[0].notes])) == hash(tuple([n.velocity for n in clipmidi_reduced.instruments[0].notes]))
    assert hash(tuple([n.start for n in clipmidi.instruments[0].notes])) == hash(tuple([n.start for n in clipmidi_reduced.instruments[0].notes]))
    print('passed assert tests')

    clipmidi = clipmidi_reduced

    # Randomize the instrument program number
    clipmidi.instruments[0].program = np.random.choice(VALID_PROGRAM_NUMBERS)
    print('USING PROGRAM #{}'.format(clipmidi.instruments[0].program))

    # now save this midi file
    out_paths = []
    # relpath points to a file with an extension. Remove it.
    file_id = os.path.splitext(relpath)[0]
    # file_id is now something like 'lakh_hash'
    basepath = MOD_MIDI_PATH
    ext = '.midi'
    # Now we append more specific identifiers
    fname = file_id + '-{}-{}'.format(start_time, clipmidi.instruments[0].program) + ext

    out_path = os.path.join(tgtpath, basepath, fname)
    out_paths.append(out_path)

    deepest_folder = os.path.join(tgtpath, basepath)
    if not(os.path.exists(deepest_folder)):
        os.makedirs(deepest_folder)

    print('saving MIDI to')
    print(out_paths[0])
    assert len(clipmidi.instruments) == 1  # only one instrument
    assert instrument_is_monophonic(clipmidi.instruments[0])  # monophonic
    assert clipmidi.instruments[0].program in VALID_PROGRAM_NUMBERS  # in our valid program numbers
    assert len(clipmidi.instruments[0].notes) >= 1  # has at least one note
    if len([n.pitch for n in clipmidi.instruments[0].notes if n.pitch > 95 or n.pitch < 24]) != 0:  # within CREPE range
        return
    assert not(clipmidi.instruments[0].is_drum)  # not a drum
    clipmidi.write(out_paths[0])
    #except Exception as e:
    #    print('ERROR')
    #    print(e)

def process_and_save(midi, tgtpath, relpath):
    """
    Process the given MIDI file (split up, synthesize, and extract parameters) and save it
    to the dataset path.

    midi: pretty_midi object
    tgtpath: target path to folder to save entire dataset
    relpath: relative path that builds from tgtpath to the specific file
    """
    LEN_CLIP = 5
    VALID_PROGRAM_NUMBERS = list(range(128))
    INVALID_PROGRAM_NUMBERS = [8, 9, 10, 13, 14, 16, 17, 18, 19, 21, 31,
                               47, 48, 49, 55, 78, 86, 87, 96, 97, 98, 101,
                               112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                               122, 123, 124, 125, 126, 127]
    for invalid in INVALID_PROGRAM_NUMBERS:
        VALID_PROGRAM_NUMBERS.remove(invalid)

    # We now split this up into LEN_CLIP-second snippets, and only take the snippets with at least one note
    last_note_time = midi.get_end_time()
    num_clips = int(last_note_time // LEN_CLIP)
    print('last_note_time', last_note_time)

    for i in range(num_clips):
        start_time = LEN_CLIP * i
        end_time = LEN_CLIP * (i + 1)
        print('start_time', start_time)
        clipmidi = copy.deepcopy(midi)
        clipmidi.adjust_times(np.array([start_time, end_time]), np.array([0., LEN_CLIP]))
        orig_instruments = clipmidi.instruments

        # For each instrument: synthesize, extract, and save
        for ins_num, ins in enumerate(clipmidi.instruments):
            process_and_save_clip(clipmidi, tgtpath, relpath, ins_num, ins, start_time, VALID_PROGRAM_NUMBERS)


def main(midi_fps, srcpath, tgtpath):
    # Keep randomly selecting a file from the Lakh MIDI dataset until it passes our filter
    while True:
        filepath = np.random.choice(midi_fps)

        # Make sure it passes through our whole-file filters
        result, midi = check_midi(filepath)
        if result:
            break

    # and save both the MIDI and the parameters
    relpath = os.path.split(filepath)[1]  # we only care about the tail file, since the superfolder does not add any information
    print('processing and saving')
    process_and_save(midi, tgtpath, relpath)


if __name__ == '__main__':
    print('parsing arguments')
    args = parse_arguments()

    if not(os.path.exists(args.sourcepath)):
        raise "Invalid source Lakh MIDI dataset path"

    if not(os.path.exists(args.targetpath)):
        os.makedirs(args.targetpath)

    # get MIDI file paths
    print('getting MIDI file paths')
    midi_fps = glob.glob(os.path.join(args.sourcepath, '2/*.mid*'))

    for i in range(args.num):
        print(i)
        # # NOTE: The extract function is stateful but here we want stateless behavior and
        # multiprocessing.Process is a quick way of achieving this
        main(midi_fps, args.sourcepath, args.targetpath)
