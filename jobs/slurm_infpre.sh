#!/bin/bash
#SBATCH --job-name=preprocess  # Job name
# # SBATCH --array=0-199  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=4            # Number of tasks
#SBATCH --time=4:00:00         # Time limit
#SBATCH --partition=shared  # Partition name
#SBATCH --account=phy240043  # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

SLURM_ARRAY_TASK_ID=0
# export TQDM_DISABLE=0

module restore cmass
conda activate cmass

exp_index=null
net_index=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /jet/home/mho1/git/ltu-cmass

nbody=abacuslike
sim=fastpm_recnoise
infer=simple

tracer=galaxy

extras="nbody.zf=0.5" # "nbody.zf=0.500015" # "nbody.zf=0.5" # hydra/job_logging=disabled" #
device="cpu"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.tracer=$tracer"
suffix="$suffix infer.device=$device $extras"
# suffix="$suffix infer.val_frac=0 infer.test_frac=1"
suffix="$suffix infer.include_noise=False infer.include_hod=False"

echo "Running inference pipeline with $suffix"
python -m cmass.infer.preprocess $suffix
# python -m cmass.infer.train $suffix net=tuning
# python -m cmass.infer.validate $suffix
