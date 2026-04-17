#!/bin/bash
#SBATCH --job-name=training  # Job name
#SBATCH --array=0-7  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=4            # Number of tasks
#SBATCH --time=24:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=0

source ~/.bashrc
conda activate cmass

# exp_index=0
net_index=$SLURM_ARRAY_TASK_ID

sleep $net_index  # to stagger the start of each job

# Command to run for each lhid
cd /u/maho3/git/ltu-cmass

nbody=quijotelike
sim=fastpm_4k_niall2
infer=simple  # simple  # lightcone

tracer=galaxy
extras="nbody.zf=0.5 infer.embedding_net=fun net=niall2" # 
device="cpu"

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"

# extras="$extras infer.verbose=True"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.tracer=$tracer"
suffix="$suffix infer.device=$device $extras"
suffix="$suffix infer.include_noise=True infer.include_hod=False"
# suffix="$suffix infer.subselect_cosmo=[0,4]"
# suffix="$suffix infer.loglinear_start_idx=30"

echo "Running inference pipeline with $suffix"

python -m cmass.infer.optuna $suffix
# python -m cmass.infer.train $suffix infer.retrain=True
# python -m cmass.infer.retrain_optuna $suffix
