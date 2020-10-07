# Setup

```sh
virtualenv -p python3 --no-site-packages wavegenie
cd wavegenie
source bin/activate
git clone https://github.com/magenta/ddsp.git
cd ddsp
pip install -e .
cd ..
git clone git@github.com:chrisdonahue/wavegenie.git
cd wavegenie
pip install -e .
pip install -r requirements.txt
```

# File Structure Convention

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

## `wavegenie/`

Contains all `wavegenie` package functionality, which primarily includes extra convenience functions taken from the official DDSP demo colab notebook.

## `get_data.sh`

Download and extract the raw indomain data used for DDSP.

## `get_params.sh`

Download and extract the DDSP model checkpoints.

## `get_test.sh`

Download test data.


