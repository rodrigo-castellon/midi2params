"""
Reproduce midi2params by doing a forward pass on a given example from the test set.
Specifically, we get the last batch of the test set and choose the 7th example, which
has paired <midi,audio>, which allows us to do several listening tests at once.
"""

import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import yaml
from addict import Dict
from datetime import datetime
import copy
import librosa as lr
from scipy.io.wavfile import read as wavread, write as wavwrite
import pretty_midi

from train_utils import *

def parse_arguments():
    # here the user can provide what they want to reproduce for
    # this dataset
    # you can either reproduce:
    # (1) DDSP resynthesis
    # (2) midi2params inference + DDSP resynthesis
    # (3) heuristic inference + DDSP resynthesis

    # for (1) you can provide an audio file to extract parameters from
    # for (2) and (3) you can provide a MIDI file

    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument('--resynthesis', type=str)
    parser.add_argument('--midi2params', type=str)
    parser.add_argument('--heuristic', type=str)

    args = parser.parse_args()

    return args

def midi2batch(midi_path):
    """
    Take a PrettyMIDI object as input and return batch, a dict with 'pitches' and
    'onset_arr' and 'offset_arr'.
    """

    frame_rate = 250
    len_clip = 10

    # handle the possibility of either .mid or .midi files
    midi = pretty_midi.PrettyMIDI(midi_path)
    print(midi.instruments)
    notes = midi.instruments[0].notes

    # get onsets/offsets from MIDI
    onsets = [int(frame_rate * n.start) for n in notes]
    offsets = [int(frame_rate * n.end) for n in notes]
    # NOTE: make extra long arrays just in case the MIDI notes go on for longer
    onset_arr = np.zeros((3 * len_clip * frame_rate), dtype=np.float32)
    onset_arr[onsets] = 1  # embed pointwise onsets/offsets into zero array
    offset_arr = np.zeros((3 * len_clip * frame_rate), dtype=np.float32)
    offset_arr[offsets] = 1  # embed pointwise onsets/offsets into zero array

    # get pitches
    pitches = notes2pitches(notes, 3 * len_clip * frame_rate, NO_NOTE_VAL=0)

    batch = {
        'pitches': pitches,
        'onset_arr': onset_arr,
        'offset_arr': offset_arr
    }

    batch = {k: t.Tensor(batch[k][:len_clip * frame_rate]) for k in batch.keys()}

    return batch

args = parse_arguments()

config_path = '/work/midi2params/midi2params/configs/midi2params-test.yml'

# get config
print('getting config')
config = load_config(config_path)

# override if we just don't have a GPU
if not(torch.cuda.is_available()) and config.device == 'cuda':
    config.device = 'cpu'

# now load the DDSP model
print('loading DDSP model...')

from utils.util import load_ddsp_model
from utils.util import synthesize_ddsp_audio

ckpt_path = '/work/midi2params/checkpoints/CustomViolinCheckpoint'
model = load_ddsp_model(ckpt_path)

if args.resynthesis:
    audio = lr.load(args.resynthesis, mono=True, sr=16000)[0][..., np.newaxis]

    # extract the f0/loudness features/parameters with DDSP
    print('extracting f0/loudness parameters with DDSP...')
    from utils.util import extract_ddsp_synthesis_parameters

    audio_parameters = extract_ddsp_synthesis_parameters(audio)

    # now resynthesize the same audio, should sound similar
    print('resythesizing...')

    resynth = synthesize_ddsp_audio(model, audio_parameters)

    wavwrite('out.wav', 16000, resynth)

elif args.midi2params:
    # now we take the MIDI for this example and instead of heuristically
    # generating f0/loudness curves, we generate them with our best learned
    # midi2params model

    # transform MIDI file into a batch, which is a dict with 'pitches', 'onset_arr',
    # and 'offset_arr'
    batch = midi2batch(args.midi2params)

    model_path = '/work/midi2params/model/best_model_cpu_120.pt'

    # load the model
    print('loading midi2params model...')
    best_model = load_best_model(config, model_path)

    # generate the parameters
    print('generating the parameters...')
    f0_pred, ld_pred = midi2params(best_model, batch)

    # now resynthesize with DDSP
    for k, arr in batch.items():
        batch[k] = to_numpy(arr)

    train_params = {
        'f0_hz': f0_pred[i],
        'loudness_db': ld_pred[i]
    }

    print('resynthesizing...')
    new_model_resynth = synthesize_ddsp_audio(model, train_params)

    wavwrite('test3.wav', 16000, new_model_resynth)

elif args.heuristic:
    # now we take the MIDI for this example and heuristically generate
    # reasonable f0/loudness curves via heuristics

    def generate_loud(beats, length=1250, decay=True):
        """
        Generate a loudness envelope for each note, decaying over time.
        """
        arrs = []
        length = 2500
        base = -30
        decay_rate = -0.01 # decays -1 per timestep/index
        #notelength = 0.7
        ld_arr = np.full((length), -120)
        for i, beat in enumerate(beats):
            if i == len(beats) - 1:
                next_beat = length
            else:
                next_beat = beats[i + 1]
            ld_arr[beat:next_beat] = np.linspace(base, base + decay_rate * (next_beat - beat), next_beat - beat)

        return ld_arr


    def gen_heuristic(batch, i=0):
        """
        Take a batch containing 'pitches', 'onset_arr', and 'offset_arr' and
        turn them into f0 and loudness heuristically.
        """
        onsets = np.where(batch['onset_arr'][i] == 1)[0]
        if len([i for i in onsets if i < 30]) == 0:
            onsets = np.concatenate(([10], onsets))

        ld = generate_loud(onsets)
        pitches = copy.deepcopy(batch['pitches'][i])
        f0 = np.array(p2f(pitches))
        return f0, ld

    print('generating heuristic parameters...')
    f0_h, ld_h = gen_heuristic(batch, i=i)
    heuristic_parameters = {
        'f0_hz': f0_h.astype(np.float32),
        'loudness_db': ld_h.astype(np.float32)
    }

    # now resynthesize into the audio. this should sound more different.
    print('resynthesizing...')

    resynth = synthesize_ddsp_audio(model, heuristic_parameters)
    wavwrite('test2.wav', 16000, resynth)
else:
   print('you didn\'t pick anything!')
