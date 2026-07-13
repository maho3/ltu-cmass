#!/bin/bash
#SBATCH --job-name=pk_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A.out
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A.out

source ~/.bashrc
conda activate cmass
cd /u/maho3/git/ltu-cmass

export OMP_NUM_THREADS=16

# 10 lhids drawn with rng seed 0 from dirs present on disk
for lhid in 33 81 150 350 538 614 1019 1269 1627 1694; do
    PYTHONPATH=. python power_tests/compare_backends.py --lhid $lhid --threads 16
done
