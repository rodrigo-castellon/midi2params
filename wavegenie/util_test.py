import os
import unittest

from scipy.io.wavfile import read as wavread

from wavegenie.paths import TEST_DATA_DIR
from wavegenie.util import *

class TestUtil(unittest.TestCase):
  def test_reproduce_colab(self):
    _, audio = wavread(os.path.join(TEST_DATA_DIR, 'colab_input.wav'))
    _, expected = wavread(os.path.join(TEST_DATA_DIR, 'colab_resynth.wav'))

    audio_parameters = extract_ddsp_synthesis_parameters(audio[:, np.newaxis])
    model = load_ddsp_model('Violin')
    resynth = synthesize_ddsp_audio(model, audio_parameters)[:, 0]

    error_rms = np.sqrt(np.mean(np.square(resynth - expected)))
    self.assertTrue(error_rms < 0.05)

if __name__ == '__main__':
  unittest.main()
