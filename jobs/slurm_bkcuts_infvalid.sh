#!/bin/bash
#SBATCH --job-name=bkcuts_val  # Job name
#SBATCH --array=0-4  # one per experiment in infer=bkcuts
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16            # Number of tasks
#SBATCH --time=6:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# Self-consistent validation of the infer=bkcuts models (2026-11 PPC round).
# Copy of jobs/slurm_infvalid.sh pointed at the reparam abacuslike suite.

source ~/.bashrc
conda activate cmass

exp_index=$SLURM_ARRAY_TASK_ID
net_index=null

sleep $exp_index

cd /u/maho3/git/ltu-cmass

nbody=abacuslike
sim=fastpm_charm7_cosmoHOD_reparam
infer=bkcuts
tracer=galaxy
extras="nbody.zf=0.5 infer.embedding_net=fun net=niall2 infer.reparam_degeneracy=True"
device="cpu"

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.tracer=$tracer"
suffix="$suffix infer.device=$device $extras"
suffix="$suffix infer.include_noise=True infer.include_hod=True"

echo "Running inference pipeline with $suffix"

python -m cmass.infer.validate $suffix
