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

# SLURM_ARRAY_TASK_ID=7

# Define absolute path to subdirectory list
SUBDIR_FILE="/anvil/scratch/x-mho1/globus/subdirs.txt"

# Get subdirectory for this array task
SUBDIR=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SUBDIR_FILE")

# Define paths
SOURCE_ENDPOINT="c42f0096-2d87-42f9-8e6a-edd08f2e1834"
DEST_ENDPOINT="d9e522d3-c51e-4037-b375-55ffd155c715"
SOURCE_PATH="/anvil/scratch/x-mho1/cmass-ili/$SUBDIR"
DEST_PATH="/ocean/projects/phy240015p/mho1/cmass-ili/$SUBDIR"
RECORD_PATH="/anvil/scratch/x-mho1/globus/$SUBDIR"

if [ ! -d "$SOURCE_PATH" ]; then
    echo "Source path $SOURCE_PATH is not a directory. Exiting."
    exit 1
fi

mkdir -p "$RECORD_PATH"  # Ensure RECORD_PATH exists

START_TIME=$(date +%s)

# Retrieve file lists
echo "Retrieving file lists for $SUBDIR..."
globus ls --long --recursive --recursive-depth-limit 10 $SOURCE_ENDPOINT:$SOURCE_PATH > "$RECORD_PATH/anvil_files.txt"
globus ls --long --recursive --recursive-depth-limit 10 $DEST_ENDPOINT:$DEST_PATH > "$RECORD_PATH/bridges_files.txt"

# # Extract modification time and filename
# awk -F '|' '$6 ~ /file/ {gsub(/^ +| +$/, "", $5); gsub(/^ +| +$/, "", $7); print $5, $7}' "$RECORD_PATH/source_files.txt" | sort > "$RECORD_PATH/source_sorted.txt"
# awk -F '|' '$6 ~ /file/ {gsub(/^ +| +$/, "", $5); gsub(/^ +| +$/, "", $7); print $5, $7}' "$RECORD_PATH/dest_files.txt" | sort > "$RECORD_PATH/dest_sorted.txt"

# # Identify files to transfer or delete
# comm -23 "$RECORD_PATH/source_sorted.txt" "$RECORD_PATH/dest_sorted.txt" > "$RECORD_PATH/to_transfer.txt"
# comm -13 "$RECORD_PATH/source_sorted.txt" "$RECORD_PATH/dest_sorted.txt" > "$RECORD_PATH/to_delete.txt"

# # Display results
# echo "Files to be transferred ($SUBDIR):"
# cat "$RECORD_PATH/to_transfer.txt"
# echo "-----------------------------------"
# echo "Files to be deleted ($SUBDIR):"
# cat "$RECORD_PATH/to_delete.txt"
# echo "-----------------------------------"

# # Remove superfluous files
# rm "$RECORD_PATH/source_files.txt" "$RECORD_PATH/dest_files.txt" "$RECORD_PATH/source_sorted.txt" "$RECORD_PATH/dest_sorted.txt"

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Time taken for $SUBDIR: $ELAPSED_TIME seconds"
