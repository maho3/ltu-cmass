#!/bin/bash
#SBATCH --job-name=copy_fastpm_z05   # Job name
#SBATCH --array=0-9                  # 10 shards
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # I/O-bound, single core per shard
#SBATCH --mem=1900M                  # Amount of memory
#SBATCH --time=01:00:00              # Time limit
#SBATCH --partition=RM-shared        # Partition name
#SBATCH --account=phy240015p         # Account name
#SBATCH --output=/ocean/projects/phy240015p/mho1/jobout/%x_%A_%a.out
#SBATCH --error=/ocean/projects/phy240015p/mho1/jobout/%x_%A_%a.out

source ~/.bashrc
conda activate cmass

cd /jet/home/mho1/git/ltu-cmass/jobs

LHID_LIST=/jet/home/mho1/git/ltu-cmass/jobs/lhid_list_z05.txt
LOGDIR=/jet/home/mho1/git/ltu-cmass/jobs/copy_z05_logs
mkdir -p "$LOGDIR"

python copy_fastpm_z05_subset.py \
    --lhid-list "$LHID_LIST" \
    --shard-idx "$SLURM_ARRAY_TASK_ID" \
    --num-shards 10 \
    --log "$LOGDIR/copy_log_${SLURM_ARRAY_TASK_ID}.tsv"
