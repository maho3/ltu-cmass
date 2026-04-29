#!/bin/bash
#SBATCH --job-name=preprocess  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=4            # Number of tasks
#SBATCH --time=4:00:00         # Time limit
#SBATCH --partition=shared  # Partition name
#SBATCH --account=phy240043  # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

SLURM_ARRAY_TASK_ID=0
# export TQDM_DISABLE=0

# module restore cmass
source ~/.bashrc
conda activate cmass

exp_index=null
net_index=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /u/maho3/git/ltu-cmass

nbody=quijote
sim=nbody_hodz_gridnoise
infer=simple  # simple  # lightcone

tracer=galaxy

extras="nbody.zf=0.5 infer.Nmax=4000 infer.test_noised_summs=False" #
# extras="$extras infer.pca_features=16" 
device="cpu"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.tracer=$tracer"
suffix="$suffix infer.device=$device $extras"
suffix="$suffix infer.val_frac=0 infer.test_frac=1"
suffix="$suffix infer.include_noise=True infer.include_hod=False"

echo "Running inference pipeline with $suffix"
python -m cmass.infer.preprocess $suffix
