#!/bin/bash
rm -rf test_data
mkdir test_data
pushd test_data
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QUEuGQO8E5ZPfESN6G_KQNQGKCYRth93' -O colab_input.wav
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kUWA3434oKW4PfiQRMybhK8j_VhY02Nl' -O colab_resynth.wav
sha256sum *.wav
popd
