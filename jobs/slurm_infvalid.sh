#!/bin/bash
#SBATCH --job-name=validate  # Job name
#SBATCH --array=0-3  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16            # Number of tasks
#SBATCH --time=4:00:00         # Time limit
#SBATCH --partition=shared  # Partition name
#SBATCH --account=phy240043  # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=0

module restore cmass
conda activate cmassrun

exp_index=$SLURM_ARRAY_TASK_ID
net_index=0

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run

nbody=abacuslike
sim=fastpm_recnoise
infer=lightcone  # simple  #

halo=False
galaxy=False
ngc=False
sgc=False
mtng=False
simbig=True

# nbody=quijote
# sim=nbody
# infer=simple  # simple  #

# halo=False
# galaxy=True
# ngc=False
# sgc=False
# mtng=False
# simbig=False

extras="" # nbody.zf=0.500015" # "nbody.zf=0.5" # 
device=cpu

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.halo=$halo infer.galaxy=$galaxy"
suffix="$suffix infer.ngc_lightcone=$ngc infer.sgc_lightcone=$sgc infer.mtng_lightcone=$mtng infer.simbig_lightcone=$simbig"
suffix="$suffix infer.device=$device $extras"
# suffix="$suffix infer.val_frac=0 infer.test_frac=1"
suffix="$suffix infer.include_noise=False"

echo "Running inference pipeline with $suffix"

python -m cmass.infer.validate $suffix
