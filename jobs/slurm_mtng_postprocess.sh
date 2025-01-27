#!/bin/bash
#SBATCH --job-name=mtnglike_postprocess   # Job name
#SBATCH --time=04:00:00         # Time limit
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=128            # Number of tasks
#SBATCH --partition=wholenode      # Partition name

# SLURM_ARRAY_TASK_ID=101
offset=2000

module restore cmass
conda activate cmassrun
lhid=$((SLURM_ARRAY_TASK_ID + offset))

python -m cmass.nbody.postprocess_fastpm nbody=mtnglike nbody.lhid=$lhid
