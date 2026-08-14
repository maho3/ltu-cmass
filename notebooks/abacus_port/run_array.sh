#!/bin/bash
#SBATCH --job-name=portabacus
#SBATCH --array=0-118%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-mho1/jobout/portabacus_%A_%a.out
#SBATCH --error=/anvil/scratch/x-mho1/jobout/portabacus_%A_%a.out

# Port AbacusSummit base halos -> ltu-cmass (NFW-fit c200c), one sim per array task.
#   sbatch run_array.sh                 # port all lhids 0..118 (8 at a time)
#   sbatch --array=6 run_array.sh       # one sim (validation)
# %8 caps concurrency: each task uses ~90 GB scratch temp, deleted when it finishes.
# Resumable: tasks skip any lhid whose halos.h5 already exists (add --overwrite to force).

source /apps/anvil/external/apps/conda/2024.09/etc/profile.d/conda.sh
conda activate /anvil/scratch/x-mho1/envs/abacusport
cd /home/x-mho1/git/ltu-cmass/notebooks

echo "task ${SLURM_ARRAY_TASK_ID} start $(date)"
python port_abacus_one.py ${SLURM_ARRAY_TASK_ID}
echo "task ${SLURM_ARRAY_TASK_ID} end   $(date)"
