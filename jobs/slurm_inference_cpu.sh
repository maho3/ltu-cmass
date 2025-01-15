#!/bin/bash
#SBATCH --job-name=galinference  # Job name
#SBATCH --array=0-23  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=32            # Number of tasks
#SBATCH --time=1:00:00         # Time limit
#SBATCH --partition=shared  # Partition name
#SBATCH --account=phy240043  # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=null
export TQDM_DISABLE=0

module restore cmass
conda activate cmass

exp_index=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass

nbody=mtnglike
sim=fastpm
infer=default

halo=False
galaxy=True
ngc=False
sgc=False
mtng=False

extras="nbody.zf=0.500015"
device=cpu

postfix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index"
postfix="$postfix infer.halo=$halo infer.galaxy=$galaxy"
postfix="$postfix infer.ngc_lightcone=$ngc infer.sgc_lightcone=$sgc infer.mtng_lightcone=$mtng"
postfix="$postfix infer.device=$device $extras"

echo "Running inference with $postfix"
# python -m cmass.infer.preprocess $postfix
python -m cmass.infer.train $postfix
