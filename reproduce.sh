#!/bin/bash

# do midi2params on a custom MIDI file, in this case it is twinkle twinkle little star
python3 midi2params/reproduce.py --midi2params data/custom_midi.mid --out data/outputs/custom_midi_out.wav

# reproduce DDSP(midi2params(midi)), DDSP(heuristic(midi)), and DDSP using a test example from the recorded dataset
python3 midi2params/reproduce.py --midi2params data/CustomViolin_16k_train_aligned/midi/test/schindler-55.p --out data/outputs/midi2params_out.wav
python3 midi2params/reproduce.py --heuristic data/CustomViolin_16k_train_aligned/midi/test/schindler-55.p --out data/outputs/heuristic_out.wav
python3 midi2params/reproduce.py --resynth data/CustomViolin_16k_train_aligned/wav/test/schindler-55.wav --out data/outputs/resynth_out.wav

