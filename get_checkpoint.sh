#!/bin/bash


# create checkpoints folder if it's not already there
if ! [ -d "checkpoints" ]; then
    mkdir checkpoints
fi

echo "Downloading the tar'ed folder containing the Custom Violin training checkpoint into our local copy of \`checkpoints\` and untar-ing it."
echo

cd checkpoints

# download the checkpoint folder
wget http://cs.stanford.edu/~rjcaste/research/CustomViolinCheckpoint.tgz

tar -xvf CustomViolinCheckpoint.tgz
