#!/bin/bash
#SBATCH --job-name=abacuslike   # Job name
#SBATCH --array=4-1000           # Job array range for lhid (0 to 10)
#SBATCH --nodes=2               # Number of nodes
#SBATCH --ntasks=256            # Number of tasks
#SBATCH --time=00:45:00         # Time limit
#SBATCH --partition=wholenode      # Partition name
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

module restore cmass
source /anvil/projects/x-phy240043/x-mho1/anaconda3/bin/activate
conda activate cmass
lhid=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass
python -m cmass.nbody.fastpm nbody=abacuslike nbody.lhid=$lhid
