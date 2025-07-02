#!/bin/bash
#SBATCH --job-name=training  # Job name
#SBATCH --array=0-7  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=8            # Number of tasks
#SBATCH --time=4:00:00         # Time limit
#SBATCH --partition=shared  # Partition name
#SBATCH --account=phy240043  # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=0

module restore cmass
conda activate cmassrun

# exp_index=0
net_index=$SLURM_ARRAY_TASK_ID

sleep $net_index  # to stagger the start of each job

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run

# nbody=quijotelike
# sim=fastpm_recnoise_ngp
# infer=lightcone  # simple

# halo=False
# galaxy=False
# ngc=False
# sgc=False
# mtng=False
# simbig=True

nbody=abacuslike
sim=fastpm_recnoise_ngp
infer=lightcone  # simple

halo=False
galaxy=False
ngc=False
sgc=True
mtng=False
simbig=False

extras="" # "nbody.zf=0.5" # 
device="cpu"

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.halo=$halo infer.galaxy=$galaxy"
suffix="$suffix infer.ngc_lightcone=$ngc infer.sgc_lightcone=$sgc infer.mtng_lightcone=$mtng infer.simbig_lightcone=$simbig"
suffix="$suffix infer.device=$device $extras"
# suffix="$suffix infer.val_frac=0 infer.test_frac=1"
# suffix="$suffix infer.prior=uniform infer.include_noise=True"

echo "Running inference pipeline with $suffix"

# python -m cmass.infer.train $suffix net=nsfonly
python -m cmass.infer.optuna $suffix # net=nsfonly
