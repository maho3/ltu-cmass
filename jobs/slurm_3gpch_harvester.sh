#!/bin/bash
#SBATCH --job-name=3gpch_harvest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A.out
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A.out

# Harvester: converts snapshots left by 3gpch_prod producers into nbody.h5,
# directly on shared scratch (no node-local copying; one serial converter
# per job). Scale by running N sharded instances, e.g. for 10 harvesters:
#   for k in $(seq 0 9); do
#       sbatch --export=ALL,SHARD_K=$k,SHARD_N=10 jobs/slurm_3gpch_harvester.sh
#   done
# Fully idempotent: safe to resubmit anytime to sweep up leftovers.

SHARD_K=${SHARD_K:-0}
SHARD_N=${SHARD_N:-1}
LHID_MIN=${LHID_MIN:-3000}
LHID_MAX=${LHID_MAX:-3999}

source ~/.bashrc
conda activate cmass
cd /u/maho3/git/ltu-cmass

scratchbase=/work/hdd/bdne/maho3/cmass_scratch/harvest
outbase=/work/hdd/bdne/maho3/cmass-ili/mtnglike/fastpm/L3000-N384

# NOTE: never delete scratchbase here — unconverted snapshots must survive
# for a rerun; the harvester deletes per-lhid data itself after conversion.

if [ "$SHARD_K" -eq 0 ]; then
    (while true; do
        echo "SCRATCH_USAGE $(date +%H:%M:%S) $(du -sBM $scratchbase 2>/dev/null | cut -f1)"
        sleep 60
    done) &
    monitor_pid=$!
fi

PYTHONPATH=. python scripts/harvest_fastpm.py \
    --scratchbase "$scratchbase" \
    --outbase "$outbase" \
    --lhids $LHID_MIN $LHID_MAX \
    --shard $SHARD_K $SHARD_N \
    --idle-timeout-min 3000

[ -n "$monitor_pid" ] && kill $monitor_pid 2>/dev/null
echo "shard $SHARD_K done"
