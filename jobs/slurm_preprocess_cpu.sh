#!/bin/bash
#SBATCH --job-name=preprocess  # Job name
#SBATCH --array=0-39  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=4            # Number of tasks
#SBATCH --time=0:10:00         # Time limit
#SBATCH --partition=shared  # Partition name
#SBATCH --account=phy240043  # Account name
#SBATCH --output=/anvil/scratch/x-dbartlett/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-dbartlett/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=3
if [[ -z "$PS1" ]]; then
    export TQDM_DISABLE=0
fi

module restore cmass_env
conda activate cmass

# exp_index=null
exp_index=$SLURM_ARRAY_TASK_ID
new_index=0

# Command to run for each lhid
cd /home/x-dbartlett/ltu-cmass

nbody=mtnglike
sim=fastpm
infer=default

halo=False
galaxy=False
ngc=True
sgc=False
mtng=False

extras="nbody.zf=0.500015"
device=cpu

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.halo=$halo infer.galaxy=$galaxy"
suffix="$suffix infer.ngc_lightcone=$ngc infer.sgc_lightcone=$sgc infer.mtng_lightcone=$mtng"
suffix="$suffix infer.device=$device $extras"

echo "Running inference with $suffix"
# python -m cmass.infer.preprocess $suffix
python -m cmass.infer.train $suffix net=tuning
