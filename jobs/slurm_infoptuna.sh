#!/bin/bash
#SBATCH --job-name=training  # Job name
#SBATCH --array=0-3  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=8            # Number of tasks
#SBATCH --time=12:00:00         # Time limit
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

nbody=quijotelike
sim=fastpm_4k_nmhf
infer=simple  # simple  # lightcone
tracer=galaxy
extras="nbody.zf=0.5 infer.embedding_net=mhf" # 
device="cpu"

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.tracer=$tracer"
suffix="$suffix infer.device=$device $extras"
suffix="$suffix infer.include_noise=True infer.include_hod=False"

echo "Running inference pipeline with $suffix"

python -m cmass.infer.optuna $suffix
# python -m cmass.infer.train $suffix infer.retrain=True
# python -m cmass.infer.retrain_optuna $suffix
