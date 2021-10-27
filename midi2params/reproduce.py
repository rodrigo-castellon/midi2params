"""
Reproduce midi2params in one of three ways, as specified by the user. Either pass in a path to a MIDI file
and run DDSP(midi2params(MIDI)) or DDSP(Heuristic(MIDI)), or you pass in a path to an audio file, in which case
we extract audio parameters from that audio file and resynthesize with DDSP. All models used here are trained
on the custom violin dataset.
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
from pathlib import Path
import pickle
import os

from utils.util import load_ddsp_model
from utils.util import synthesize_ddsp_audio
from utils.util import extract_ddsp_synthesis_parameters
from utils.util import notes2pitches

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

    parser = argparse.ArgumentParser(description='Run inference with the best model.')

    parser.add_argument('--resynthesis', type=str)
    parser.add_argument('--midi2params', type=str)
    parser.add_argument('--heuristic', type=str)
    parser.add_argument('--out', type=str, default='out.wav')

    args = parser.parse_args()

    return args

def midi2batch(midi_path):
    """
    Take a path to a MIDI file as input and return batch,
    a dict with 'pitches' and 'onset_arr' and 'offset_arr'.

    Note that this accepts both .mid* files and "processed"
    pickle files (see MIDI2ParamsDataset.load_midi for more
    info).
    """

    midi_path = Path(midi_path)
    frame_rate = 250
    len_clip = 10

    if midi_path.suffix[:4] == '.mid':
        # handle the possibility of either .mid or .midi files
        midi = pretty_midi.PrettyMIDI(str(midi_path))
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

    elif midi_path.suffix == '.p':
        arrs = pickle.load(open(midi_path, 'rb'))
        pitches, onset_arr, offset_arr = arrs['pitches'].astype(np.float32), arrs['onset_arr'].astype(np.float32), arrs['offset_arr'].astype(np.float32)

    batch = {
	'pitches': pitches,
	'onset_arr': onset_arr,
	'offset_arr': offset_arr
    }

    batch = {k: t.Tensor(batch[k][np.newaxis, :len_clip * frame_rate]) for k in batch.keys()}

    return batch

def resynthesize(audio_path, model):
    """
    Take a path to an audio file, extract f0/loudness parameters, and
    resynthesize using DDSP. Then, synthesize with DDSP.
    """

    audio = lr.load(audio_path, mono=True, sr=16000)[0][..., np.newaxis]

    # extract the f0/loudness features/parameters with DDSP
    print('extracting f0/loudness parameters with DDSP...')

    audio_parameters = extract_ddsp_synthesis_parameters(audio)

    # now resynthesize the same audio, should sound similar
    print('resythesizing...')

    resynth = synthesize_ddsp_audio(model, audio_parameters)

    return resynth

def convert_midi_to_parameters(midi_path, config, model):
    """
    Take a path to a MIDI file and use the learned model to generate f0/loudness
    parameters using the best learned midi2params model. Then, synthesize with
    DDSP.
    """

    # transform MIDI file into a batch, which is a dict with 'pitches', 'onset_arr',
    # and 'offset_arr'
    batch = midi2batch(midi_path)

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
        'f0_hz': f0_pred[0],
        'loudness_db': ld_pred[0]
    }

    print('resynthesizing...')
    new_model_resynth = synthesize_ddsp_audio(model, train_params)

    return new_model_resynth

def heuristic(midi_path, model):
    """
    Take a path to a MIDI file and heuristically generate reasonable
    f0/loudness curves via heuristics. Then, synthesize with DDSP.
    """

    # transform MIDI file into a batch, which is a dict with 'pitches', 'onset_arr',
    # and 'offset_arr'
    batch = midi2batch(midi_path)

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
    f0_h, ld_h = gen_heuristic(batch, i=0)
    heuristic_parameters = {
        'f0_hz': f0_h.astype(np.float32),
        'loudness_db': ld_h.astype(np.float32)
    }

    # now resynthesize into the audio. this should sound more different.
    print('resynthesizing...')
    
    resynth = synthesize_ddsp_audio(model, heuristic_parameters)

    return resynth

def main():
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

    ckpt_path = '/work/midi2params/checkpoints/CustomViolinCheckpoint'
    model = load_ddsp_model(ckpt_path)

    # perform computation
    if args.resynthesis:
        output = resynthesize(args.resynthesis, model)
    elif args.midi2params:
        output = convert_midi_to_parameters(args.midi2params, config, model)
    elif args.heuristic:
        output = heuristic(args.heuristic, model)
    else:
       print('you didn\'t pick anything!')
       return

    # save
    os.makedirs(Path(args.out).parent, exist_ok=True)
    wavwrite(args.out, 16000, output)


if __name__ == '__main__':
    main()
