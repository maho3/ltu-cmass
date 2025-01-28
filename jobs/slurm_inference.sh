#!/bin/bash
#SBATCH --job-name=sgcinference  # Job name
#SBATCH --array=0-23  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=5            # Number of tasks
#SBATCH --gres=gpu:1            # Number of GPUs
#SBATCH --time=12:00:00         # Time limit
#SBATCH --partition=GPU-shared # Partition name
#SBATCH --account=phy240015p   # Account name
#SBATCH --output=/ocean/projects/phy240015p/mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/ocean/projects/phy240015p/mho1/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=null
export TQDM_DISABLE=0

module restore cmass
conda activate cmass

exp_index=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /jet/home/mho1/git/ltu-cmass

nbody=abacuslike
sim=fastpm
infer=default

halo=False
galaxy=False
ngc=False
sgc=True
mtng=False

extras="nbody.zf=0.500015"
device=cuda

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index"
suffix="$suffix infer.halo=$halo infer.galaxy=$galaxy"
suffix="$suffix infer.ngc_lightcone=$ngc infer.sgc_lightcone=$sgc infer.mtng_lightcone=$mtng"
suffix="$suffix infer.device=$device $extras"

echo "Running inference with $suffix"
# python -m cmass.infer.preprocess $suffix
python -m cmass.infer.train $suffix
