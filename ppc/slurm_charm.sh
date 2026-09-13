#!/bin/bash
#SBATCH --job-name=ppc_charm    # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=8              # Number of tasks
#SBATCH --time=8:00:00          # Time limit
#SBATCH --partition=ghx4        # Partition name
#SBATCH --gpus-per-node=1       # Number of GPUs per node
#SBATCH --account=bdne-dtai-gh  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

# Stage B of the PPC campaign: CHARM over all 20 posterior draws.
# Thin wrapper -- ppc/run_charm.sh holds the pinned checkpoint (charm7,
# charm_joint_v19.pth) and hydra overrides. Environment setup copied from
# jobs/slurm_charm.sh, which run_charm.sh does not do itself.
# The loop is serial and resume safe: draws with halos.h5 are skipped.

source ~/.bashrc
module load cray-hdf5/1.14.3.7
conda activate cmass

export TQDM_DISABLE=0

cd /u/maho3/git/ltu-cmass
bash ppc/run_charm.sh 0 19
