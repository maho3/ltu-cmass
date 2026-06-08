#!/bin/bash
# ---------------------------------------------------------------------------
# Drain finished FastPM field maps off NCSA Delta into PSC Bridges2 via Globus,
# then delete the transferred nbody.h5 on Delta to free quota.
#
# Run this MANUALLY on PSC Bridges2 (a login node, or anywhere with globus-cli
# + outbound HTTPS to api.globus.org). Delta needs NO CLI -- it is referenced
# only by its Globus collection UUID. Run `globus login` once beforehand.
#
# SAFETY: the Delta source dir for each lhid holds ONLY nbody.h5 (+ config.yaml)
# -- the 76GB of particle snapshots live on node-local /tmp and never appear
# here. We delete ONLY nbody.h5 (keeping config.yaml as a resume tombstone so
# slurm_fastpm_2000.sh won't recompute the lhid), and only after the transfer
# task reports SUCCEEDED (checksum-verified).
#
# Usage:
#   ./offload_globus_bridges2.sh              # run once
#   ./offload_globus_bridges2.sh --dry-run    # preview, transfer/delete nothing
#   ./offload_globus_bridges2.sh --loop       # repeat every 3600s until Ctrl-C
#   ./offload_globus_bridges2.sh --loop 1800  # repeat every 1800s until Ctrl-C
#
# Scheduling: use --loop in a tmux/screen session, OR add a cron entry, e.g.
#   */30 * * * * /path/to/offload_globus_bridges2.sh >> ~/offload.log 2>&1
# ---------------------------------------------------------------------------

set -euo pipefail

# === EDIT THESE =============================================================
DELTA_EP="7e936164-de58-4e3d-85da-21aa23c07169"        # `globus endpoint search "NCSA Delta"`
BRIDGES2_EP="d9e522d3-c51e-4037-b375-55ffd155c715"  # `globus endpoint search "Bridges2"`
DELTA_PATH="/work/hdd/bdne/maho3/cmass-ili/abacuslike/fastpm/L2000-N256"
BRIDGES2_PATH="/ocean/projects/phy240015p/mho1/cmass-ili/abacuslike/fastpm/L2000-N256"  # TODO
# ============================================================================

DRY_RUN=0
LOOP_INTERVAL=0   # 0 = run once

usage() { sed -n '2,30p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --loop)
            if [[ "${2:-}" =~ ^[0-9]+$ ]]; then LOOP_INTERVAL="$2"; shift 2
            else LOOP_INTERVAL=3600; shift; fi ;;
        --loop=*) LOOP_INTERVAL="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

module load globus 2>/dev/null || true   # PSC may provide globus-cli via a module

run_offload() {
    echo "[$(date +%T)] Listing completed sims on Delta ($DELTA_PATH) ..."
    # lhid dirs whose nbody.h5 exists == completed and safe to move/delete.
    local LHIDS
    mapfile -t LHIDS < <(
        globus ls --recursive --recursive-depth-limit 1 "$DELTA_EP:$DELTA_PATH/" 2>/dev/null \
            | grep -E '/nbody\.h5$' \
            | sed -E 's#/nbody\.h5$##'
    )

    if [ "${#LHIDS[@]}" -eq 0 ]; then
        echo "No completed nbody.h5 found. Nothing to do."
        return 0
    fi
    echo "Found ${#LHIDS[@]} completed sims to offload."

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] would transfer (checksum sync) $DELTA_PATH -> $BRIDGES2_PATH"
        globus transfer --dry-run --recursive --sync-level checksum --preserve-mtime \
            "$DELTA_EP:$DELTA_PATH" "$BRIDGES2_EP:$BRIDGES2_PATH" || true
        echo "[dry-run] would then delete (keeping config.yaml tombstones):"
        printf '  %s/nbody.h5\n' "${LHIDS[@]}"
        return 0
    fi

    # --- 1. Transfer (checksum sync is idempotent + resumable) --------------
    local TASK_ID STATUS
    TASK_ID=$(
        globus transfer --recursive --sync-level checksum --preserve-mtime \
            --label "delta->bridges2 cmass $(date +%F)" \
            --format unix --jmespath 'task_id' \
            "$DELTA_EP:$DELTA_PATH" "$BRIDGES2_EP:$BRIDGES2_PATH"
    )
    echo "Submitted Globus transfer task: $TASK_ID"

    globus task wait "$TASK_ID" --polling-interval 30
    STATUS=$(globus task show "$TASK_ID" --format unix --jmespath 'status')
    echo "Transfer task status: $STATUS"

    if [ "$STATUS" != "SUCCEEDED" ]; then
        echo "Transfer did NOT succeed -- skipping all deletions for safety." >&2
        return 1
    fi

    # --- 2. Delete only nbody.h5 on Delta (keep config.yaml tombstone) ------
    local DEL_BATCH d
    DEL_BATCH=$(mktemp)
    for d in "${LHIDS[@]}"; do
        echo "$d/nbody.h5" >> "$DEL_BATCH"   # e.g. "2000/nbody.h5" -> $DELTA_PATH/2000/nbody.h5
    done

    echo "Deleting nbody.h5 for ${#LHIDS[@]} sims on Delta (keeping config.yaml tombstones) ..."
    globus delete --ignore-missing --batch "$DEL_BATCH" "$DELTA_EP:$DELTA_PATH"
    rm -f "$DEL_BATCH"

    echo "[$(date +%T)] Done. Offloaded and freed ${#LHIDS[@]} field maps from Delta."
}

if [ "$LOOP_INTERVAL" -gt 0 ]; then
    echo "Looping offload every ${LOOP_INTERVAL}s (Ctrl-C to stop)."
    while true; do
        run_offload || echo "offload iteration failed; will retry next cycle."
        echo "Sleeping ${LOOP_INTERVAL}s ..."
        sleep "$LOOP_INTERVAL"
    done
else
    run_offload
fi
