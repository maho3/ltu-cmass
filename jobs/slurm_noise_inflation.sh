#!/bin/bash
#SBATCH --job-name=noiseinfl
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out

source ~/.bashrc
conda activate cmass
cd /u/maho3/git/ltu-cmass

python scripts/noise_inflation.py
