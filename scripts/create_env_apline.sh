#!/bin/sh

#SBATCH --job-name=create_env
#SBATCH --account=amc-general
#SBATCH --output=./logs/create_env.log
#SBATCH --error=./logs/error_create_env.log
#SBATCH --time=00:30:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1 

# Exit if any command fails
set -e

# Create the dwpc_rnn conda environment from environment.yml

## for alpine
module load anaconda

# deactivate any loaded environment
conda deactivate

# remove environments with the same name
conda env remove --name dwpc_rnn

conda env create -f environment.yml

echo "dwpc_rnn environment created"
