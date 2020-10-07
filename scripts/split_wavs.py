"""
This script splits n long wav files into LEN_CLIP-long segments and saves these wav clips to a new target directory.
"""

from scipy.io.wavfile import read as wavread, write as wavwrite
import librosa
import os
import argparse

SAMPLE_RATE = 16000

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate fles training dataset randomly (or add to a pre-existing one).')
    parser.add_argument('--length', '-l', type=float, default=5, help='Length of the splits.')
    parser.add_argument('--sourcepath', '-s', type=str, default='data/DDSPViolin/wav/train', help='Path to the source Lakh dataset (as is after extracting tar file)')
    parser.add_argument('--targetpath', '-t', type=str, default='data/DDSPViolin/wav/train/clips', help='Path to the target dataset path')

    args = parser.parse_args()

    return args

def main(args):
    """
    Go through the WAV files in the directory, split them up, and save them to the target directory.
    """

    LEN_CLIP = args.length * SAMPLE_RATE
    if not(os.path.exists(args.targetpath)):
        os.makedirs(args.targetpath)

    fpaths = [os.path.join(args.sourcepath, f) for f in os.listdir(args.sourcepath) if os.path.splitext(f)[1] == '.wav']

    for fpath in fpaths:
        print('splitting up {}'.format(fpath))
        fname = os.path.split(fpath)[1]
        f_id = os.path.splitext(fname)[0]
        # split up the file
        try:
            audio = wavread(fpath)[1]
        except ValueError:  # if the file is 24-bit
            # fall back to librosa
            audio = librosa.load(fpath, sr=SAMPLE_RATE)[0].flatten()
        # We now split this up into LEN_CLIP-second snippets, and only take the snippets with at least one note
        last_note_time = audio.shape[0]
        num_clips = int(last_note_time // LEN_CLIP)
        print('last_note_time', last_note_time)

        for i in range(num_clips):
            start_time = LEN_CLIP * i
            end_time = LEN_CLIP * (i + 1)
            print('start_time', i * args.length)
            write_fpath = os.path.join(args.targetpath, f_id + '-{}.wav'.format(i * args.length))
            wavwrite(write_fpath, SAMPLE_RATE, audio[start_time:end_time])




if __name__ == '__main__':
    args = parse_arguments()

    main(args)