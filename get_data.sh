#!/bin/bash

# download the file ID

if ! [ -d "data" ]; then
    mkdir data
fi

echo "Use this script to download the tar'ed folder containing training data into our local copy of \`data\` and untar it."
echo


cd data

wget http://cs.stanford.edu/~rjcaste/research/CustomViolin_16k_train_aligned.tar
wget http://cs.stanford.edu/~rjcaste/research/CustomViolin_16k_train_transcribed.tar

tar -xvf CustomViolin_16k_train_aligned.tar
tar -xvf CustomViolin_16k_train_transcribed.tar

# get 10s one because that's the one we use to test
wget http://cs.stanford.edu/~rjcaste/research/CustomViolin_16k_10s.tar
tar -xvf CustomViolin_16k_10s.tar
