#!/bin/sh

#SBATCH --job-name=create_hetmat
#SBATCH --account=amc-general
#SBATCH --output=./logs/output_create_hetmat.log
#SBATCH --error=./logs/error_create_hetmat.log
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1 

# Exit if any command fails
set -e

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Define relative data and output paths
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"

# Conda Environment:
module load anaconda
conda deactivate
conda activate dwpc_rnn

##########################################################################################################
##########################################################################################################

echo "****** permuting graphs ******"

# Loop through permutation numbers 0-10 to create 11 permuted graphs
for permutation_num in {0..10}; do
    echo "Creating permutation ${permutation_num}..."
    
    # Define input and output notebook paths
    input_notebook=${notebooks_path}/1_generate-permutations.ipynb
    output_notebook=${notebooks_path}/1_generate-permutations_output_${permutation_num}.ipynb
    
    # Run papermill with the permutation_number parameter
    papermill "$input_notebook" "$output_notebook" -p permutation_number "$permutation_num"
    
    echo "Completed permutation ${permutation_num}"
done

echo "All permutations (0-10) completed successfully!" 


# Deactivate the virtual environment w/ conda
conda deactivate