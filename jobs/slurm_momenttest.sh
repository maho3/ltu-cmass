#!/bin/bash
#SBATCH --job-name=momenttest  # Job name
#SBATCH --array=0-15  # Array range: one task per experiment (summary combo)
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=2            # Number of tasks
#SBATCH --time=3:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=0

source ~/.bashrc
conda activate cmass

exp_index=$SLURM_ARRAY_TASK_ID

sleep $exp_index  # to stagger the start of each job

cd /u/maho3/git/ltu-cmass

nbody=abacuslike
sim=fastpm_charm6_momenthod
infer=moment_test

tracer=${tracer:-galaxy}
extras="net=moment_test infer.verbose=True"
device="cpu"

# export TQDM_DISABLE=0
# extras="$extras hydra/job_logging=disabled"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=0"
suffix="$suffix infer.tracer=$tracer"
suffix="$suffix infer.device=$device $extras"
suffix="$suffix infer.include_noise=True infer.include_hod=True"

echo "Running moment-network training test with $suffix"

python -m cmass.infer.train $suffix
