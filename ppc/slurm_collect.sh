#!/bin/bash
#SBATCH --job-name=ppc_collect  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16             # Number of tasks
#SBATCH --mem=64G               # Amount of memory
#SBATCH --time=2:00:00          # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A.out   # Error file

# Step 4 of the PPC campaign: collect x_ppc/theta_ppc and make the plots.
#
# The bands stage is cheap, but the theta-space plots (corner + logprob) load
# the 8-net ensemble and draw n_post samples from it, which blew a 40-minute
# wall on the login node. Hence a job.

source ~/.bashrc
conda activate cmass

cd /u/maho3/git/ltu-cmass

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export TQDM_DISABLE=0

PYTHONPATH=. python -u ppc/collect.py --n_post 5000 --device cpu
echo "ppc_collect status=$?"
