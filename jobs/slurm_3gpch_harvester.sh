#!/bin/bash
#SBATCH --job-name=3gpch_harvest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=36:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A.out
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A.out

# Harvester: converts snapshots left by 3gpch_prod producer jobs into
# nbody.h5 (see scripts/harvest_fastpm.py). Scale horizontally by running
# N sharded instances, e.g. for 4 harvesters:
#   for k in 0 1 2 3; do
#       sbatch --export=ALL,SHARD_K=$k,SHARD_N=4 \
#           jobs/slurm_3gpch_harvester.sh
#   done
# Defaults to a single unsharded harvester (SHARD_N=1).
# Fully idempotent: safe to resubmit anytime to sweep up leftovers.

SHARD_K=${SHARD_K:-0}
SHARD_N=${SHARD_N:-1}
LHID_MIN=${LHID_MIN:-3000}
LHID_MAX=${LHID_MAX:-3999}

source ~/.bashrc
conda activate cmass
cd /u/maho3/git/ltu-cmass

scratchbase=/work/hdd/bdne/maho3/cmass_scratch/harvest
localbase=/tmp/maho3/harvest_${SLURM_JOB_ID}
outbase=/work/hdd/bdne/maho3/cmass-ili/mtnglike/fastpm/L3000-N384

mkdir -p "$localbase"
trap 'rm -rf "$localbase"' EXIT
# NOTE: never trap-delete scratchbase — unconverted snapshots must survive
# for a rerun; the harvester deletes per-lhid data itself after conversion.

(while true; do
    echo "SCRATCH_USAGE $(date +%H:%M:%S) $(du -sBM $scratchbase 2>/dev/null | cut -f1) LOCAL $(du -sBM $localbase 2>/dev/null | cut -f1)"
    sleep 60
done) &
monitor_pid=$!

PYTHONPATH=. python scripts/harvest_fastpm.py \
    --scratchbase "$scratchbase" \
    --localbase "$localbase" \
    --outbase "$outbase" \
    --lhids $LHID_MIN $LHID_MAX \
    --shard $SHARD_K $SHARD_N \
    --nmove 6 --nconvert 5 --local-cap-gb 500 \
    --idle-timeout-min 120

kill $monitor_pid 2>/dev/null
echo "Peak scratch (all shards combined):"
grep -h SCRATCH_USAGE /work/hdd/bdne/maho3/jobout/3gpch_harvest_${SLURM_JOB_ID}.out | awk '{print $3}' | sort -n | tail -1
