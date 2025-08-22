#!/bin/zsh

# Create the dwpc_rnn_clean conda environment from environment.yml

## for alpine
module load anaconda

conda env create -f environment.yml

echo "Activate the environment with: conda activate dwpc_rnn"
