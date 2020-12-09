"""
This script splits n long MIDI files into LEN_CLIP-long segments and saves these MIDI clips to a new target directory. The big difference between this script and split_midis.py is that this saves each clip as a dictionary {'pitches': [...], 'onset_arr': [...], 'offset_arr': [...]} serialized as a pickle file instead.
"""

import pretty_midi
import os
import argparse
import copy
import numpy as np
import pickle

from utils.util import notes2pitches

FRAME_RATE = 250

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate fles training dataset randomly (or add to a pre-existing one).')
    parser.add_argument('--length', '-l', type=float, default=5, help='Length of the splits in seconds.')
    parser.add_argument('--sourcepath', '-s', type=str, default='data/DDSPViolin-clipped3/midi/test', help='Path to the source Lakh dataset (as is after extracting tar file)')
    parser.add_argument('--targetpath', '-t', type=str, default='data/DDSPViolin-clipped3/midi/test/clips', help='Path to the target dataset path')

    args = parser.parse_args()

    return args


def main(args):
    """
    Go through the MIDI files in the directory, split them up, and save them to the target directory.
    """

    global FRAME_RATE
    LEN_CLIP = args.length
    if not(os.path.exists(args.targetpath)):
        os.makedirs(args.targetpath)

    fpaths = [os.path.join(args.sourcepath, f) for f in os.listdir(args.sourcepath) if os.path.splitext(f)[1] in ['.mid', '.midi']]

    for fpath in fpaths:
        print('splitting up {}'.format(fpath))
        fname = os.path.split(fpath)[1]
        f_id = os.path.splitext(fname)[0]
        # split up the file
        midi = pretty_midi.PrettyMIDI(fpath)
        # We now split this up into LEN_CLIP-second snippets, and only take the snippets with at least one note
        last_note_time = midi.get_end_time()
        num_clips = int(last_note_time // LEN_CLIP)
        print('last_note_time', last_note_time)

        notes = midi.instruments[0].notes

        for i in range(num_clips):
            start_time = LEN_CLIP * i
            end_time = LEN_CLIP * (i + 1)
            print('start_time', start_time)
            valid_notes = [n for n in copy.deepcopy(notes) if n.end > start_time and n.start < end_time]
            for note in valid_notes:
                note.start -= start_time
                note.end -= start_time

            pitches = notes2pitches(valid_notes, LEN_CLIP * FRAME_RATE,
                                    NO_NOTE_VAL=0, FRAME_RATE=FRAME_RATE)

            # create onset_arr and offset_arr
            onsets = [int(FRAME_RATE * n.start) for n in valid_notes]
            offsets = [int(FRAME_RATE * n.end) for n in valid_notes]
            # filter out bad onsets and bad offsets
            onsets = [x for x in onsets if x >= 0]
            offsets = [x for x in offsets if x <= LEN_CLIP * FRAME_RATE]
            # NOTE: make extra long arrays just in case the MIDI notes go on for longer
            onset_arr = np.zeros((10 * LEN_CLIP * FRAME_RATE), dtype=np.float32)
            onset_arr[onsets] = 1  # embed pointwise onsets/offsets into zero array
            offset_arr = np.zeros((10 * LEN_CLIP * FRAME_RATE), dtype=np.float32)
            offset_arr[offsets] = 1  # embed pointwise onsets/offsets into zero array

            savepath = os.path.join(args.targetpath, f_id + '-{}.p'.format(start_time))

            arrs = {
                'pitches': pitches,
                'onset_arr': onset_arr,
                'offset_arr': offset_arr
            }

            pickle.dump(arrs, open(savepath, 'wb'))

if __name__ == '__main__':
    args = parse_arguments()

    main(args)
