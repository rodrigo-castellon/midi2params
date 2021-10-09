#!/bin/bash


echo "Downloading the tar'ed folder containing the best midi2params model into our local repo and untar-ing it."
echo

# download the model tar file
#wget http://cs.stanford.edu/~rjcaste/research/model.tar

#tar -xvf model.tar
#rm model.tar

wget http://cs.stanford.edu/~rjcaste/research/best_model.tar

tar -xvf best_model.tar
rm best_model.tar
