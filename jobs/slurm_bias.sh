#!/bin/bash
#SBATCH --job-name=new_charm   # Job name
#SBATCH --array=0-999         # Job array range for lhid
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=64            # Number of tasks
#SBATCH --time=01:00:00         # Time limit
#SBATCH --partition=shared      # Partition name
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

module restore cmass
source /anvil/projects/x-phy240043/x-mho1/anaconda3/bin/activate
conda activate cmass
lhid=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass
# 0-1000
python -m cmass.bias.rho_to_halo nbody=abacuslike sim=fastpm nbody.lhid=$lhid

# 1000-2000
lhid=$((lhid+1000))
python -m cmass.bias.rho_to_halo nbody=abacuslike sim=fastpm nbody.lhid=$lhid
