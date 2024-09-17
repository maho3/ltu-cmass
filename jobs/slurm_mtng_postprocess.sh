#!/bin/bash
#SBATCH --job-name=mtnglike_postprocess   # Job name
#SBATCH --time=02:00:00         # Time limit
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=64            # Number of tasks
#SBATCH --partition=shared      # Partition name

module restore cmass
module load openmpi
source /anvil/projects/x-phy240043/x-mho1/anaconda3/bin/activate
conda activate cmassrun

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run
# lhid=$SLURM_ARRAY_TASK_ID

python -m cmass.nbody.postprocess_fastpm nbody=mtnglike nbody.lhid=$lhid
