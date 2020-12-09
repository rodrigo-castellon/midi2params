import base64
import json
import os
import subprocess
import tempfile

import numpy as np
from scipy.io.wavfile import read as wavread, write as wavwrite

"""
YouTubeDownloaderException
  RequestException
    FilesizeException
    DurationException
    UnsupportedProviderException
    NoPermissionException
    RegionLockedException
    InvalidResourceException
  CommunicationException
    ThrottledException
    (retry)
      ProviderException
      TimeoutException
  RetryException
  UnknownException
"""

class YouTubeDownloaderException(Exception):
  pass


class RequestException(YouTubeDownloaderException):
  pass


class CommunicationException(YouTubeDownloaderException):
  pass


class RetryException(YouTubeDownloaderException):
  pass


class UnknownException(YouTubeDownloaderException):
  pass


class FilesizeException(RequestException):
  pass


class DurationException(RequestException):
  pass


class UnsupportedProviderException(RequestException):
  pass


class NoPermissionException(RequestException):
  pass


class RegionLockedException(RequestException):
  pass


class InvalidResourceException(RequestException):
  pass


class ThrottledException(CommunicationException):
  pass


class ProviderException(CommunicationException, RetryException):
  pass


class TimeoutException(CommunicationException, RetryException):
  pass


def _run_yt_cmd(cmd, stdout_fn, timeout=None, wait=False):
  p = subprocess.Popen(
      cmd.split(),
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE)
  try:
    p_res = p.communicate(timeout=timeout)
  except subprocess.TimeoutExpired:
    p.kill()
    if wait:
      p.wait()
    raise TimeoutException()

  stdout, stderr = [s.decode('utf-8').strip() for s in p_res]

  exception = None
  try:
    result = stdout_fn(stdout)
    assert result != False
  except Exception as e:
    exception = e

  p.kill()

  if exception is not None:
    if 'File is larger than max' in stdout:
      raise FilesizeException()

    if 'in your country' in stderr:
      raise RegionLockedException()
    if 'only available to Music Premium members' in stderr:
      raise NoPermissionException()
    if 'Incomplete YouTube ID' in stderr:
      raise InvalidResourceException()
    if 'Unsupported URL' in stderr:
      raise UnsupportedProviderException()
    if 'HTTP Error 404' in stderr:
      raise InvalidResourceException()
    if 'HTTP Error 403' in stderr:
      raise NoPermissionException()
    if 'HTTP Error 429' in stderr:
      raise ThrottledException()
    if 'HTTP Error 503' in stderr:
      raise ProviderException()
    if '--password' in stderr:
      raise NoPermissionException()
    if 'Unable to extract options' in stderr:
      raise RequestException()
    if 'is not a valid URL' in stderr:
      raise RequestException()
    if 'not available' in stderr:
      raise RequestException()
    if 'unavailable' in stderr:
      raise RequestException()
    if 'DRM protected' in stderr:
      raise NoPermissionException()
    if 'Unable to download' in stderr:
      raise RequestException()
    raise UnknownException(stdout + ';;;' + stderr)

  return result


def _run_yt_cmd_with_retries(*args, max_retries=1, **kwargs):
  for i in range(max_retries):
    try:
      return _run_yt_cmd(*args, **kwargs)
    except Exception as e:
      if isinstance(e, RetryException) and i + 1 < max_retries:
        continue
      else:
        raise e


_CMD_METADATA_TEMPLATE = """
youtube-dl \
    --dump-json \
    --no-cache-dir \
    --no-continue \
    --no-playlist \
    --format bestaudio/best \
    --extract-audio \
    --audio-format wav \
    --max-downloads 1 \
    {force_ipv4} \
    {url}
""".strip()
def load_youtube_metadata(url, timeout=None, max_retries=1, force_ipv4=False):
  cmd = _CMD_METADATA_TEMPLATE.format(
      force_ipv4='--force-ipv4' if force_ipv4 else '',
      url=url)
  metadata = _run_yt_cmd_with_retries(
      cmd,
      lambda x: json.loads(x),
      max_retries=max_retries,
      timeout=timeout)
  return metadata


_CMD_AUDIO_TEMPLATE = """
youtube-dl \
    --no-cache-dir \
    --no-continue \
    --no-playlist \
    --format bestaudio/best \
    --extract-audio \
    --audio-format wav \
    --max-downloads 1 \
    --write-info-json \
    {max_filesize} \
    --output {output_fp} \
    {force_ipv4} \
    {url}
""".strip()
def _load_youtube_audio(
    url,
    timeout=None,
    max_duration_seconds=None,
    max_filesize_mb=None,
    apply_start_end_trim=False,
    max_retries=1,
    force_ipv4=False,
    *args,
    **kwargs):
  """
  try:
    metadata = load_youtube_metadata(url, timeout=timeout, max_retries=max_retries)
  except Exception as e:
    if str(e) == 'Timeout':
      raise Exception('Timeout')
    else:
      logging.error(str(e))
      raise Exception('Could not retrieve metadata for URL')

  if max_duration_seconds is not None:
    try:
      duration_seconds = float(metadata['duration'])
    except:
      raise Exception('Duration not in metadata')
    if duration_seconds > max_duration_seconds:
      raise ValueError('Video too long')

  if max_filesize_mb is not None:
    max_filesize = float(max_filesize_mb) * 1024 * 1024
    try:
      filesize = float(metadata['filesize'])
    except:
      logging.warning('Filesize not in metadata')
      filesize = None

    if filesize is not None and filesize > max_filesize:
      raise ValueError('Video file too large')
  """

  with tempfile.NamedTemporaryFile(suffix='.wav') as f:
    # Retrieve video
    cmd = _CMD_AUDIO_TEMPLATE.format(
        max_filesize='--max-filesize {}m'.format(max_filesize_mb) if max_filesize_mb is not None else '',
        output_fp=f.name,
        force_ipv4='--force-ipv4' if force_ipv4 else '',
        url=url)
    _run_yt_cmd_with_retries(
        cmd,
        lambda x: os.path.getsize(f.name) > 0,
        max_retries=max_retries,
        timeout=timeout)

    # Parse metadata
    try:
      with open(f.name + '.info.json', 'r') as m:
        metadata = json.loads(m.read())
    except:
      metadata = {}
    if max_duration_seconds is not None and metadata.get('duration', 0.) > max_duration_seconds:
      raise DurationException()

    # Load audio into memory
    audio, fs = _load_audio(f.name, *args, **kwargs)

  # Trim start/end of youtube video
  if apply_start_end_trim:
    start = 0
    end = audio.shape[0]
    if metadata['start_time'] is not None and metadata['start_time'] > 0:
      start = int(metadata['start_time'] * float(fs))
    if metadata['end_time'] is not None and metadata['end_time'] > 0:
      end = int(metadata['end_time'] * float(fs))
    audio = audio[start:end]

  # Check duration once more
  if max_duration_seconds is not None:
    actual_duration = audio.shape[0] / float(fs)
    if actual_duration > max_duration_seconds:
      raise DurationException()

  return (audio, fs), metadata


def _load_audio(
    fp,
    fs=None,
    num_channels=None,
    normalize=False):
  # Try to use scipy if possible
  try:
    ext = os.path.splitext(fp)[1]
  except:
    ext = None

  wav = None
  if ext == '.wav':
    try:
      read_fs, wav = wavread(fp)
      if wav.dtype == np.int16:
        wav = int16_to_float32(wav)
      elif wav.dtype == np.float32:
        pass
      else:
        raise ValueError()
    except:
      wav = None

  # Fall back to librosa
  if wav is None:
    import librosa
    wav, read_fs = librosa.core.load(fp, sr=fs, mono=False)
    if wav.ndim == 2:
      wav = np.swapaxes(wav, 0, 1)

  assert wav.dtype == np.float32

  # At this point, _wav is np.float32 either [nsamps,] or [nsamps, nch].
  # We want [nsamps, 1, nch] to mimic 2D shape of spectral feats.
  if wav.ndim == 1:
    nsamps = wav.shape[0]
    nch = 1
  else:
    nsamps, nch = wav.shape
  wav = np.reshape(wav, [nsamps, nch])

  # Resample if not done already
  if fs is not None and fs != read_fs:
    import librosa
    resampled_chs = []
    for i in range(nch):
      ch = wav[:, i]
      ch = librosa.resample(np.asfortranarray(ch), read_fs, fs)
      resampled_chs.append(ch)
    wav = np.stack(resampled_chs, axis=1)
    read_fs = fs

  # Ensure number of channels is correct
  if num_channels is not None and nch != num_channels:
    # Average (mono) or expand (stereo) channels
    if num_channels == 1:
      wav = np.mean(wav, 1, keepdims=True)
    elif nch == 1 and num_channels == 2:
      wav = np.concatenate([wav, wav], axis=1)
    else:
      raise ValueError('Number of audio channels not equal to num specified')

  # Normalize
  if normalize:
    wav = normalize_audio(wav)

  return wav, read_fs


def normalize_audio(wav):
  # Normalize
  factor = np.max(np.abs(wav))
  if factor > 0:
    wav = wav / factor  # hack to avoid the read-only error
    #wav /= factor
  return wav


def save_wav(fp, audio, fs):
  if audio.dtype not in [np.int16, np.float32]:
    raise ValueError('Audio must be either int16 or float32')
  if audio.ndim != 2:
    raise ValueError('Audio must have two dimensions')
  wavwrite(fp, fs, audio)


def float32_to_int16(x):
  if x.dtype != np.float32:
    raise ValueError('Input is not float32')
  x = np.copy(x)
  x *= np.iinfo(np.int16).max
  x = np.clip(x, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
  x = x.astype(np.int16)
  return x


def int16_to_float32(x):
  if x.dtype != np.int16:
    raise ValueError('Input is not int16')
  x = x.astype(np.float32) / np.iinfo(np.int16).max
  return x


def load_audio(
    fp_or_url,
    fs=None,
    num_channels=None,
    normalize=False,
    timeout=None,
    start_time_seconds=None,
    end_time_seconds=None):
  if 'http' in fp_or_url:
    # Cloud file
    (audio, fs), _ = _load_youtube_audio(
        fp_or_url,
        timeout=timeout,
        fs=fs,
        num_channels=num_channels,
        normalize=False)
  else:
    # Load local file
    audio, fs = _load_audio(
        fp_or_url,
        fs=fs,
        num_channels=num_channels,
        normalize=False)

  # Trim start/end of youtube video
  start = 0
  if start_time_seconds is not None:
    start = int(start_time_seconds * float(fs))
  end = audio.shape[0]
  if end_time_seconds is not None:
    end = int(end_time_seconds * float(fs))
  audio = audio[start:end]

  # Renormalize
  if normalize:
    audio = normalize_audio(audio)

  return audio, fs
