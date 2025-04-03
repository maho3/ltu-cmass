#!/bin/bash
#SBATCH --job-name=globus_sync   # Job name
#SBATCH --array=0-18             # Job array range for subfolders
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=2              # Number of tasks
#SBATCH --time=12:00:00          # Time limit
#SBATCH --partition=shared       # Partition name
#SBATCH --account=phy240043      # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file

# SLURM_ARRAY_TASK_ID=10
# for i in {0..17}; do export SLURM_ARRAY_TASK_ID=$i; sh jobs/check_globus_sync.sh; done
# for i in {0..17}; do 
#     export SLURM_ARRAY_TASK_ID=$i
#     sh jobs/check_globus_sync.sh & 
# done
# wait  # Ensures the script waits for all background jobs to finish

# Define absolute path to subdirectory list
SUBDIR_FILE="/anvil/scratch/x-mho1/globus/subdirs.txt"

# Get subdirectory for this array task
SUBDIR=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SUBDIR_FILE")

# Define paths
# SOURCE_ENDPOINT="c42f0096-2d87-42f9-8e6a-edd08f2e1834"
# DEST_ENDPOINT="d9e522d3-c51e-4037-b375-55ffd155c715"
SOURCE_PATH="/anvil/scratch/x-mho1/cmass-ili/$SUBDIR"
# DEST_PATH="/ocean/projects/phy240015p/mho1/cmass-ili/$SUBDIR"
RECORD_PATH="/anvil/scratch/x-mho1/globus/$SUBDIR"

if [ ! -d "$SOURCE_PATH" ]; then
    echo "Source path $SOURCE_PATH is not a directory. Exiting."
    exit 1
fi

mkdir -p "$RECORD_PATH"  # Ensure RECORD_PATH exists

START_TIME=$(date +%s)

# Retrieve file lists
echo "Retrieving file lists for $SUBDIR..."
ls -lR $SOURCE_PATH > $RECORD_PATH/anvil_files.txt

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Time taken for $SUBDIR: $ELAPSED_TIME seconds"
