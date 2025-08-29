#!/bin/bash

# Master script to generate individual SLURM job scripts for each permutation
# This allows for distributed processing across multiple HPC nodes

# Exit if any command fails
set -e

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SCRIPT_DIR/..")

# Define paths
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"
jobs_dir="${SCRIPT_DIR}/permutation_jobs"
logs_dir="${BASE_DIR}/logs"

# Create directories if they don't exist
mkdir -p "$jobs_dir"
mkdir -p "$logs_dir"

echo "****** Generating SLURM job scripts for permutations ******"

# Function to create individual SLURM job script
create_permutation_job() {
    local perm_num=$1
    local job_script="${jobs_dir}/permutation_${perm_num}.sh"
    
    cat > "$job_script" << EOF
#!/bin/bash

#SBATCH --job-name=perm_${perm_num}
#SBATCH --account=amc-general
#SBATCH --output=${logs_dir}/output_permutation_${perm_num}.log
#SBATCH --error=${logs_dir}/error_permutation_${perm_num}.log
#SBATCH --time=01:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=8G

# Exit if any command fails
set -e

echo "Starting permutation ${perm_num} at \$(date)"
echo "Running on node: \$SLURM_NODELIST"
echo "Job ID: \$SLURM_JOB_ID"

# Load conda environment
module load anaconda
conda deactivate
conda activate dwpc_rnn

# Define paths
notebooks_path="${notebooks_path}"
input_notebook="\${notebooks_path}/1_generate-permutations.ipynb"
output_notebook="\${notebooks_path}/1_generate-permutations_output_${perm_num}.ipynb"

echo "Input notebook: \$input_notebook"
echo "Output notebook: \$output_notebook"
echo "Permutation number: ${perm_num}"

# Run papermill with the permutation_number parameter
echo "Starting papermill execution..."
papermill "\$input_notebook" "\$output_notebook" -p permutation_number ${perm_num}

echo "Permutation ${perm_num} completed successfully at \$(date)"
EOF

    # Make the job script executable
    chmod +x "$job_script"
    echo "Created job script: $job_script"
}

# Generate individual job scripts for permutations 0-10
for permutation_num in {0..10}; do
    create_permutation_job $permutation_num
done



echo ""
echo "****** Auto-submitting all permutation jobs ******"

# Automatically submit all jobs
echo "Submitting all permutation jobs..."
JOB_IDS=()

for permutation_num in {0..10}; do
    job_script="${jobs_dir}/permutation_${permutation_num}.sh"
    echo "Submitting permutation ${permutation_num}..."
    job_id=$(sbatch --parsable "$job_script")
    JOB_IDS+=($job_id)
    echo "  Job ID: $job_id"
done

echo ""
echo "All jobs submitted! Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor job status with: squeue -u \$USER"
echo "Cancel all jobs with: scancel ${JOB_IDS[@]}"


# Deactivate the virtual environment w/ conda
conda deactivate