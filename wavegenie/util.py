import os
import pickle
import re

import ddsp
import ddsp.training
import gin
import numpy as np
import copy
import torch
import torch.nn.functional as F

from .paths import PARAMS_DIR

DDSP_DEFAULT_FS_AUDIO = 16000
DDSP_DEFAULT_FS_PARAM = 250
DDSP_OFFICIAL_MODELS = [
    'Flute',
    'Flute2',
    'Tenor_Saxophone',
    'Trumpet',
    'Violin'
]


def extract_ddsp_synthesis_parameters(
    audio,
    param_fs=DDSP_DEFAULT_FS_PARAM,
    audio_fs=DDSP_DEFAULT_FS_AUDIO):
  ddsp.spectral_ops.reset_crepe()
  audio_features = ddsp.training.metrics.compute_audio_features(
      audio.swapaxes(0, 1),
      sample_rate=audio_fs,
      frame_rate=param_fs)
  audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
  del audio_features['audio']
  return audio_features


def load_ddsp_model(model_tag_or_dir):
  if model_tag_or_dir in DDSP_OFFICIAL_MODELS:
    model_dir = os.path.join(PARAMS_DIR, 'ddsp_official', model_tag_or_dir)
    if not os.path.exists(model_dir):
      raise Exception('Please run ./get_params.sh')
  else:
    model_dir = model_tag_or_dir

  # NOTE: This is only used to heuristically modify params
  #with open(os.path.join(model_dir, 'dataset_statistics.pkl'), 'rb') as f:
  #  dataset_statistics = pickle.load(f)

  # Load config
  cfg_fp = os.path.join(model_dir, 'operative_config-0.gin')
  gin.clear_config()
  with gin.unlock_config():
    gin.parse_config_file(cfg_fp, skip_unknown=True)

  # Find latest checkpoint
  ckpt_fns = list(set([f.split('.')[0] for f in os.listdir(model_dir) if f.startswith('ckpt-')]))
  ckpt_fns = sorted(ckpt_fns, key=lambda x: int(re.search(r'ckpt-(\d+)', x).group(1)))
  ckpt_fn = ckpt_fns[-1]
  ckpt_fp = os.path.join(model_dir, ckpt_fn)

  return (model_dir, cfg_fp, ckpt_fp)


def synthesize_ddsp_audio(model, synthesis_parameters):
  model_dir, cfg_fp, ckpt_fp = model

  # Parse config
  with gin.unlock_config():
    gin.clear_config()
    gin.parse_config_file(cfg_fp, skip_unknown=True)

  # Make sure we're using default hop size
  ddsp_default_hop_size = int(DDSP_DEFAULT_FS_AUDIO / DDSP_DEFAULT_FS_PARAM)
  model_time_steps_train = gin.query_parameter('DefaultPreprocessor.time_steps')
  model_n_samples_train = gin.query_parameter('Additive.n_samples')
  model_hop_size = int(model_n_samples_train / model_time_steps_train)
  if model_hop_size != ddsp_default_hop_size:
    # TODO: Support different hop sizes
    raise ValueError('Model uses non-standard hop size')
  
  # Update params
  time_steps = synthesis_parameters['f0_hz'].shape[0]
  num_samples = time_steps * ddsp_default_hop_size
  gin_params = [
    'Additive.n_samples = {}'.format(num_samples),
    'FilteredNoise.n_samples = {}'.format(num_samples),
    'DefaultPreprocessor.time_steps = {}'.format(time_steps),
  ]

  with gin.unlock_config():
    # NOTE: add default input keys if it's not already specified, since
    # DDSP will default to f0, ld, *and* z.
    if not('RnnFcDecoder.input_keys = ' in gin.config_str()):
      gin_params.append('RnnFcDecoder.input_keys = ("f0_scaled", "ld_scaled")')
    gin.parse_config(gin_params)

  # Load model  
  # NOTE: We have to load the model every time because the DDSP codebase is incredible
  model = ddsp.training.models.Autoencoder()
  model.restore(ckpt_fp)

  # Synthesize
  audio = model(synthesis_parameters, training=False)
  audio = audio.numpy().swapaxes(0, 1)

  return audio


def preview_audio(audio, fs=DDSP_DEFAULT_FS_AUDIO):
  from IPython.display import display, Audio
  display(Audio(audio.swapaxes(0, 1), rate=fs))

p2f_ = lambda p: 440 * 2**((p - 69) / 12)  # MIDI pitch to frequency
f2p = lambda f: 69 + 12 * np.log2(f / 440)  # frequency to MIDI pitch

# fixed for no note indices
def p2f(p):
  p = copy.deepcopy(p)
  p[np.where(p > 0)] = p2f_(p[np.where(p > 0)])
  return p

def to_numpy(tensor):
  """
  Convert any potential torch tensor to numpy array.
  """
  if type(tensor) == type(np.array([])):
    return tensor
  elif torch.is_tensor(tensor):
    if tensor.is_cuda:
      tensor = tensor.cpu()
    if tensor.requires_grad:
      tensor = tensor.detach()
    return np.array(tensor)
  else:
    raise 'Invalid object.'

def sample_from(probs, onehot=True):
  """
  Sample from a probs tensor of shape (N_1,...,N_k,num_classes) across the -1th
  dimension and return a one-hot tensor of shape (N_1,...,N_k,num_classes). If
  onehot is False, then return a tensor of shape (N_1,...,N_k) of indices for
  the aforementioned one-hot tensor.
  
  We're essentially probabilistically collapsing along the last dimension.
  """

  orig_shape = probs.shape
  num_classes = probs.shape[-1]
  probs = probs.view(-1, num_classes)  # take one sample

  sample = torch.multinomial(probs, 1)
  if onehot:
    sample = F.one_hot(sample, num_classes=num_classes)
    return sample.view(orig_shape)
  else:
    return sample.view(orig_shape[:-1])

def notes2pitches(notes, length, NO_NOTE_VAL=128, FRAME_RATE=250, transform=None):
  """
  Turn a list of monophonic notes into a list of pitches
  at 250Hz. In other words, turn [Note(), Note(), ...] into
  np.array [65, 65, 65, 65, 128, ...]
  """

  # NOTE: [0, 127] is reserved by MIDI, but we use
  # NO_NOTE_VAL as a flag value for "no note playing right now"
  pitches = np.full((length), NO_NOTE_VAL, dtype=np.float32)

  for note in notes:
    start_idx = int(note.start * FRAME_RATE)
    end_idx = int(note.end * FRAME_RATE)
    pitch = note.pitch
    if not(transform is None):
      pitch = transform(pitch)
    pitches[start_idx:end_idx] = pitch
  return pitches

normalize = lambda x: (x - x.min()) / (x.max() - x.min())