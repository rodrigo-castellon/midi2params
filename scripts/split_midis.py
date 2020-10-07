"""
This script splits n long MIDI files into LEN_CLIP-long segments and saves these MIDI clips to a new target directory.
"""

import pretty_midi
import os
import argparse
import copy
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate fles training dataset randomly (or add to a pre-existing one).')
    parser.add_argument('--length', '-l', type=float, default=5, help='Length of the splits.')
    parser.add_argument('--sourcepath', '-s', type=str, default='data/DDSPViolin/midi/train', help='Path to the source Lakh dataset (as is after extracting tar file)')
    parser.add_argument('--targetpath', '-t', type=str, default='data/DDSPViolin/midi/train/clips', help='Path to the target dataset path')

    args = parser.parse_args()

    return args


def main(args):
    """
    Go through the MIDI files in the directory, split them up, and save them to the target directory.
    """

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

        for i in range(num_clips):
            start_time = LEN_CLIP * i
            end_time = LEN_CLIP * (i + 1)
            print('start_time', start_time)
            clipmidi = copy.deepcopy(midi)
            clipmidi.adjust_times(np.array([start_time, end_time]), np.array([0., LEN_CLIP]))

            clipmidi.write(os.path.join(args.targetpath, f_id + '-{}.midi'.format(start_time)))


if __name__ == '__main__':
    args = parse_arguments()

    main(args)