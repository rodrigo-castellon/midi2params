# Getting Started

Below are some commands you might want to run to get this repo set up and running.

```sh
# set up a virtual environment
virtualenv -p python3 --no-site-packages midi2params_env
cd midi2params_env
source bin/activate

# install some python packages
pip install ddsp==0.7.0
pip install pyyaml
pip install addict
pip install torch==1.2.0
pip install pretty_midi
python -m pip install ipykernel
git clone https://github.com/rodrigo-castellon/midi2params.git
cd midi2params
pip install -e .
# install the kernel into jupyter, so that it can be used
# in the notebook
python -m ipykernel install --user --name=testenv3

# get necessary data
./get_data.sh
./get_checkpoint.sh
./get_model.sh
```

To test the model out, `notebooks/midi2params-results.ipynb` is a demo notebook.

Also, check out the paper, currently hosted [here](https://cs.stanford.edu/~rjcaste/research/realistic_midi.pdf).

# File Structure Convention

Some of these may be mainly relevant if you're interested in training models.

## `configs/`

A folder that contains YAML config files for training models (MIDI -> parameters, parameters -> MIDI, etc.)

## `data/`

A folder containing the datasets to train on. Each folder under `data/` represents a distinct dataset. Each dataset `data/dataset` contains subfolders, such as `synth_params/` and `modified_midis/`. Then, under each of those is `train/`, `val/`, and `test/`. Under each of those are raw files---WAV files, MIDIs, or pickle files---that make up the meat of the dataset. The file names (excluding the extension) should be identical between the subfolders of a given dataset.

## `logs/`

Contains logs from runs training models. Each run is a folder under `logs/` that contains the best model so far, as well as losses over time, etc.

## `midi2params/`

Training script/utilities for our midi2param models.

## `notebooks/`

Catch-all for one-off and long-term Jupyter notebooks.

## `params/`

Model checkpoints from DDSP.

## `params2midi/`

Training script/utilities for our params2midi models.

## `scripts/`

Catch-all for one-off and long-term reusable scripts for data wrangling/manipulation.

## `utils/`

Contains utility and convenience functions.
