# Getting Started

## Reproducing inference

Inference can be reproduced with Docker containers. Follow the below instructions to do so:

1. Make sure you have Docker and Git installed on your machine.
2. Create a new file called `Dockerfile` and copy-paste the below in it:
```Dockerfile
FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN python3 -m pip install ddsp==0.7.0
RUN python3 -m pip install pyyaml
RUN python3 -m pip install addict
RUN python3 -m pip install pretty_midi
RUN python3 -m pip install pandas
RUN python3 -m pip install --upgrade setuptools

# install some system packages
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get install -y libsndfile1

RUN echo "rm -rf /work/midi2params" >> /workspace/startup.sh
RUN echo "cd /work" >> /workspace/startup.sh
RUN echo "git clone https://github.com/rodrigo-castellon/midi2params.git" >> /workspace/startup.sh
RUN echo "cd midi2params && git checkout reproduce" >> /workspace/startup.sh
RUN echo "python3 -m pip install -e ." >> /workspace/startup.sh
RUN echo "./get_data.sh" >> /workspace/startup.sh
RUN echo "./get_checkpoint.sh" >> /workspace/startup.sh
RUN echo "./get_model.sh" >> /workspace/startup.sh

ENTRYPOINT /bin/bash /workspace/startup.sh && /bin/bash
```
3. Run the command `docker build -t midi2params . && docker run -it --rm -v $(pwd):/work midi2params`
    - This will build the Docker image from the above Dockerfile and run it as a container, with bash as the shell.
4. Once inside the Docker container, go to the `/work/midi2params` directory.
5. Run `bash reproduce.sh` to reproduce DDSP(midi2params(MIDI)), DDSP(Heuristic(MIDI)), and DDSP, as seen in the demo page [here](https://rodrigo-castellon.github.io/midi2params/).

## Reproducing Training

Reproducing training runs with Docker containers is forthcoming.

## Paper

Check out the paper, currently hosted [here](https://cs.stanford.edu/~rjcaste/research/realistic_midi.pdf).

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
