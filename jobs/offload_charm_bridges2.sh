#!/bin/bash
# ---------------------------------------------------------------------------
# offload_charm_bridges2.sh  --  Bridges2 Globus orchestrator
#
# Tracks state in a local manifest to avoid slow globus ls calls.
# States: staged | done
#   staged = nbody.h5 + config.yaml pushed to Delta, halos.h5 not yet seen
#   done   = halos.h5 pulled back to Bridges2, full $lhid/ dir deleted on Delta
#
# Sentinel convention (no tombstones):
#   $lhid/ dir present on Delta  <->  lhid is staged/in-progress
#   $lhid/ dir absent on Delta   <->  not yet staged OR already cleaned up
#
# Resync (--resync or first run):
#   staged = dirs present on Delta (flat ls, fast)
#   done   = lhids with halos.h5 on Bridges2 (flat ls + one level, fast)
#   not yet staged = absent from both
#
# Usage:
#   ./offload_charm_bridges2.sh              # loop forever (default 600s)
#   ./offload_charm_bridges2.sh --interval 1800
#   ./offload_charm_bridges2.sh --dry-run
#   ./offload_charm_bridges2.sh --resync     # rebuild manifest from globus ls
# ---------------------------------------------------------------------------

set -euo pipefail

# === CONFIG =================================================================
DELTA_EP="7e936164-de58-4e3d-85da-21aa23c07169"
BRIDGES2_EP="d9e522d3-c51e-4037-b375-55ffd155c715"

DELTA_BASE="/work/hdd/bdne/maho3/cmass-ili/abacuslike/fastpm/L2000-N256"
BRIDGES2_BASE="/ocean/projects/phy240015p/mho1/cmass-ili/abacuslike/fastpm/L2000-N256"

LHID_MIN=0
LHID_MAX=3999
STAGE_TARGET=300
INTERVAL=600
MANIFEST="./charm_offload_manifest.txt"  # lhid<TAB>state
DRY_RUN=0
RESYNC=0
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1; shift ;;
        --resync)     RESYNC=1; shift ;;
        --interval)   INTERVAL="$2"; shift 2 ;;
        --interval=*) INTERVAL="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

module load globus 2>/dev/null || true

log() { echo "[$(date '+%Y-%m-%d %T')] $*"; }

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

manifest_get() {
    local lhid="$1"
    grep -E "^${lhid}	" "$MANIFEST" 2>/dev/null | cut -f2 || true
}

manifest_set() {
    local lhid="$1" state="$2"
    if grep -qE "^${lhid}	" "$MANIFEST" 2>/dev/null; then
        sed -i "s/^${lhid}\t.*/${lhid}\t${state}/" "$MANIFEST"
    else
        echo -e "${lhid}\t${state}" >> "$MANIFEST"
    fi
}

manifest_count() {
    grep -E "	$1$" "$MANIFEST" 2>/dev/null | wc -l | tr -d '[:space:]'
}

manifest_lhids_with_state() {
    grep -E "	$1$" "$MANIFEST" 2>/dev/null | cut -f1 || true
}

# ---------------------------------------------------------------------------
# Resync manifest from Globus (flat ls only -- fast)
# ---------------------------------------------------------------------------

resync_manifest() {
    log "Resyncing manifest from Globus (flat ls on both endpoints) ..."

    # Dirs present on Delta = staged (lhid/ present means nbody.h5 is there)
    local delta_dirs
    mapfile -t delta_dirs < <(
        globus ls "$DELTA_EP:$DELTA_BASE/" 2>/dev/null \
            | grep -E "^[0-9]+/$" \
            | sed 's#/##' \
            | sort -n
    )

    # lhids with halos.h5 on Bridges2 = done
    local bridges2_done
    mapfile -t bridges2_done < <(
        globus ls --recursive --recursive-depth-limit 1 \
            "$BRIDGES2_EP:$BRIDGES2_BASE/" 2>/dev/null \
            | grep -E "^[0-9]+/halos\.h5$" \
            | sed -E 's#/halos\.h5##' \
            | sort -n
    )

    declare -A delta_set bridges2_set
    for l in "${delta_dirs[@]}";    do delta_set[$l]=1;   done
    for l in "${bridges2_done[@]}"; do bridges2_set[$l]=1; done

    # Consistency check: lhid on Delta but also done on Bridges2
    # means a previous cleanup was interrupted -- mark for re-cleanup
    local inconsistent=0
    for l in "${delta_dirs[@]}"; do
        if [ "${bridges2_set[$l]+_}" ]; then
            log "WARNING: lhid=$l has dir on Delta but halos.h5 already on Bridges2 -- will re-clean Delta."
            ((inconsistent++)) || true
        fi
    done
    [ "$inconsistent" -gt 0 ] && log "Found $inconsistent inconsistent lhids; they will be treated as staged (halos.h5 already on Bridges2 means next cycle pulls nothing new but cleanup will fire)."

    > "$MANIFEST"
    for lhid in $(seq "$LHID_MIN" "$LHID_MAX"); do
        if [ "${bridges2_set[$lhid]+_}" ]; then
            echo -e "${lhid}\tdone" >> "$MANIFEST"
        elif [ "${delta_set[$lhid]+_}" ]; then
            echo -e "${lhid}\tstaged" >> "$MANIFEST"
        fi
    done

    log "Resync complete. staged=$(manifest_count staged) done=$(manifest_count done)"
}

# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

run_once() {
    local n_staged n_done slots
    n_staged=$(manifest_count staged)
    n_done=$(manifest_count done)
    slots=$(( STAGE_TARGET - n_staged ))
    log "staged=$n_staged done=$n_done | staging slots available=$slots"

    # -------------------------------------------------------------------------
    # 1. Check staged lhids for completed halos.h5 on Delta
    #    Targeted per-lhid ls -- only touches dirs we know exist
    # -------------------------------------------------------------------------
    local staged_lhids
    mapfile -t staged_lhids < <(manifest_lhids_with_state staged | sort -n)

    if [ "${#staged_lhids[@]}" -gt 0 ]; then
        log "Checking ${#staged_lhids[@]} staged lhids for completed halos.h5 ..."

        local done_lhids=()
        for lhid in "${staged_lhids[@]}"; do
            if globus ls "$DELTA_EP:$DELTA_BASE/$lhid/" 2>/dev/null \
                    | grep -q "^halos\.h5$"; then
                done_lhids+=("$lhid")
            fi
        done
        log "Newly completed: ${#done_lhids[@]}"

        if [ "${#done_lhids[@]}" -gt 0 ]; then
            if [ "$DRY_RUN" -eq 1 ]; then
                log "[dry-run] would pull halos.h5 and delete dirs for: ${done_lhids[*]}"
            else
                # Pull halos.h5 to Bridges2
                local TRANSFER_BATCH TASK_ID STATUS
                TRANSFER_BATCH=$(mktemp)
                for lhid in "${done_lhids[@]}"; do
                    echo "$lhid/halos.h5 $lhid/halos.h5"
                done > "$TRANSFER_BATCH"

                TASK_ID=$(
                    globus transfer \
                        --sync-level checksum --preserve-mtime \
                        --label "charm halos->bridges2 $(date +%F)" \
                        --format unix --jmespath 'task_id' \
                        --batch "$TRANSFER_BATCH" \
                        "$DELTA_EP:$DELTA_BASE" "$BRIDGES2_EP:$BRIDGES2_BASE"
                )
                rm -f "$TRANSFER_BATCH"
                log "Pull task submitted: $TASK_ID"

                globus task wait "$TASK_ID" --polling-interval 30
                STATUS=$(globus task show "$TASK_ID" --format unix --jmespath 'status')
                log "Pull status: $STATUS"

                if [ "$STATUS" != "SUCCEEDED" ]; then
                    log "ERROR: pull transfer failed -- skipping cleanup."
                    return 1
                fi

                # Delete entire $lhid/ directory on Delta
                local DEL_BATCH
                DEL_BATCH=$(mktemp)
                for lhid in "${done_lhids[@]}"; do
                    echo "$lhid/"
                done > "$DEL_BATCH"

                globus delete --ignore-missing --recursive \
                    --batch "$DEL_BATCH" "$DELTA_EP:$DELTA_BASE"
                rm -f "$DEL_BATCH"
                log "Deleted ${#done_lhids[@]} lhid dirs from Delta."

                for lhid in "${done_lhids[@]}"; do
                    manifest_set "$lhid" done
                done
            fi

            n_staged=$(manifest_count staged)
            slots=$(( STAGE_TARGET - n_staged ))
        fi
    fi

    # -------------------------------------------------------------------------
    # 2. Stage new nbody.h5 + config.yaml up to STAGE_TARGET
    # -------------------------------------------------------------------------
    if [ "$slots" -le 0 ]; then
        log "Stage target met. Nothing to push."
        return 0
    fi

    local to_stage=()
    for lhid in $(seq "$LHID_MIN" "$LHID_MAX"); do
        local state
        state=$(manifest_get "$lhid")
        [ -n "$state" ] && continue
        to_stage+=("$lhid")
        [ "${#to_stage[@]}" -ge "$slots" ] && break
    done

    if [ "${#to_stage[@]}" -eq 0 ]; then
        log "No new lhids to stage. All submitted or complete."
        return 0
    fi

    log "Staging ${#to_stage[@]} lhids (nbody.h5 + config.yaml) to Delta ..."

    if [ "$DRY_RUN" -eq 1 ]; then
        log "[dry-run] would stage lhids: ${to_stage[*]}"
        return 0
    fi

    local STAGE_BATCH TASK_ID STATUS
    STAGE_BATCH=$(mktemp)
    for lhid in "${to_stage[@]}"; do
        echo "$lhid/nbody.h5 $lhid/nbody.h5"
        echo "$lhid/config.yaml $lhid/config.yaml"
    done > "$STAGE_BATCH"

    TASK_ID=$(
        globus transfer \
            --sync-level checksum --preserve-mtime \
            --label "charm nbody->delta $(date +%F)" \
            --format unix --jmespath 'task_id' \
            --batch "$STAGE_BATCH" \
            "$BRIDGES2_EP:$BRIDGES2_BASE" "$DELTA_EP:$DELTA_BASE"
    )
    rm -f "$STAGE_BATCH"
    log "Staging task submitted: $TASK_ID"

    globus task wait "$TASK_ID" --polling-interval 30
    STATUS=$(globus task show "$TASK_ID" --format unix --jmespath 'status')
    log "Staging status: $STATUS"

    if [ "$STATUS" != "SUCCEEDED" ]; then
        log "ERROR: staging transfer failed."
        return 1
    fi

    for lhid in "${to_stage[@]}"; do
        manifest_set "$lhid" staged
    done

    log "Staged ${#to_stage[@]} lhids. Cycle complete."
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if [ ! -f "$MANIFEST" ] || [ "$RESYNC" -eq 1 ]; then
    resync_manifest
fi

log "Starting offload_charm_bridges2 (interval=${INTERVAL}s, stage_target=$STAGE_TARGET, dry_run=$DRY_RUN)"
log "Manifest: $MANIFEST"

while true; do
    run_once || log "Cycle failed; will retry next interval."
    log "Sleeping ${INTERVAL}s ..."
    sleep "$INTERVAL"
done
