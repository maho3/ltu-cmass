#!/bin/bash
#SBATCH --job-name=inference  # Job name
#SBATCH --array=0-15  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=32            # Number of tasks
#SBATCH --gpus-per-node=1     # Number of GPUs
#SBATCH --time=12:00:00         # Time limit
#SBATCH --partition=gpu # Partition name
#SBATCH --account=phy240043-gpu   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=1
export TQDM_DISABLE=0

module restore cmass
conda activate cmassrun

exp_index=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run

nbody=quijotelike
sim=fastpm
halo=True
galaxy=True
ngc=False
sgc=False
extras="" # "nbody.zf=0.500015"
device=cuda

postfix="nbody=$nbody sim=$sim infer.exp_index=$exp_index"
postfix="$postfix infer.halo=$halo infer.galaxy=$galaxy"
postfix="$postfix infer.ngc_lightcone=$ngc infer.sgc_lightcone=$sgc"
postfix="$postfix infer.device=$device $extras"

echo "Running inference with $postfix"
python -m cmass.infer.train $postfix
