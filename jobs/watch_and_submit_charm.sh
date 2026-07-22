#!/bin/bash
# ---------------------------------------------------------------------------
# watch_and_submit.sh  --  DeltaAI login node orchestrator
#
# Polls $OUTDIR for staged nbody.h5 files, submits one SLURM job per lhid
# that is ready (nbody.h5 present, halos.h5 absent, not already queued).
# Caps concurrent jobs at MAX_JOBS to avoid flooding the queue.
#
# Usage:
#   ./watch_and_submit.sh              # loop forever (default interval 300s)
#   ./watch_and_submit.sh --interval 600
#   ./watch_and_submit.sh --dry-run
#
# Run in a tmux session on a DeltaAI login node.
# ---------------------------------------------------------------------------

set -euo pipefail

# === CONFIG =================================================================
OUTDIR="/work/hdd/bdne/maho3/cmass-ili/abacuslike/fastpm/L2000-N256"
SCRIPT_DIR="/u/maho3/git/ltu-cmass/jobs"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_charm_single.sh"
LHID_MIN=0
LHID_MAX=3999
MAX_JOBS=200        # max concurrent charm jobs in queue (running+pending)
INTERVAL=300        # polling interval in seconds
DRY_RUN=0
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --interval)   INTERVAL="$2"; shift 2 ;;
        --interval=*) INTERVAL="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %T')] $*"; }

count_queued() {
    # Count running+pending charm jobs belonging to this user
    squeue --me --name=charm --noheader 2>/dev/null | wc -l
}

run_once() {
    local queued
    queued=$(count_queued)
    log "Queued/running charm jobs: $queued / $MAX_JOBS"

    local submitted=0
    local skipped_done=0
    local skipped_nostage=0

    for lhid in $(seq "$LHID_MIN" "$LHID_MAX"); do
        local lhid_dir="$OUTDIR/$lhid"
        local nbody_file="$lhid_dir/nbody.h5"
        local halos_file="$lhid_dir/halos.h5"

        # Already done
        if [ -f "$halos_file" ]; then
            ((skipped_done++)) || true
            continue
        fi

        # Input not staged yet
        if [ ! -f "$nbody_file" ]; then
            ((skipped_nostage++)) || true
            continue
        fi

        # Check cap before each submission
        queued=$(count_queued)
        if [ "$queued" -ge "$MAX_JOBS" ]; then
            log "Job cap ($MAX_JOBS) reached. Will retry next cycle."
            break
        fi

        if [ "$DRY_RUN" -eq 1 ]; then
            log "[dry-run] would submit lhid=$lhid"
        else
            sbatch --export=lhid="$lhid" "$SLURM_SCRIPT" > /dev/null
            log "Submitted lhid=$lhid"
        fi
        ((submitted++)) || true
    done

    log "Cycle done: submitted=$submitted done=$skipped_done waiting_for_stage=$skipped_nostage"
}

log "Starting watch_and_submit (interval=${INTERVAL}s, max_jobs=$MAX_JOBS, dry_run=$DRY_RUN)"
while true; do
    run_once
    log "Sleeping ${INTERVAL}s ..."
    sleep "$INTERVAL"
done
