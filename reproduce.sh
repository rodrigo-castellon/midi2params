#!/bin/bash

python3 midi2params/reproduce.py --midi2params data/CustomViolin_16k_train_aligned/midi/test/schindler-55.p --out midi2params_out.wav
python3 midi2params/reproduce.py --heuristic data/CustomViolin_16k_train_aligned/midi/test/schindler-55.p --out heuristic_out.wav
python3 midi2params/reproduce.py --resynth data/CustomViolin_16k_train_aligned/wav/test/schindler-55.wav --out resynth_out.wav
