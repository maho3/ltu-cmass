#!/bin/bash
#SBATCH --job-name=resim  # Job name
#SBATCH --array=0-15  # Array range, one task per experiment index
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=4            # Number of tasks
#SBATCH --time=2:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=0

# module restore cmass
source ~/.bashrc
conda activate cmass

exp_index=$SLURM_ARRAY_TASK_ID

sleep $exp_index  # to stagger the start of each job

cd /u/maho3/git/ltu-cmass


nbody=abacuslike
sim=fastpm_charm6_comphod
infer=simple  # simple  # lightcone
tracer=galaxy
extras="nbody.zf=0.5 infer.embedding_net=fun" #  infer.verbose=True"
device="cpu"

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index"
suffix="$suffix infer.tracer=$tracer"
suffix="$suffix infer.device=$device $extras"
suffix="$suffix infer.include_noise=True infer.include_hod=False"
suffix="$suffix infer.resim.n_resim=100 infer.resim.n_post=10000"
# suffix="$suffix infer.resim.pool=[train,val,test]"

echo "Running posterior resimulation with $suffix"

python -m cmass.infer.resim $suffix
