#!/bin/bash
# Campaign dashboard for the FastPM producer/harvester pipeline.
# Usage: scripts/harvest_status.sh [LHID_MIN] [LHID_MAX]

LHID_MIN=${1:-3000}
LHID_MAX=${2:-3999}
outbase=/work/hdd/bdne/maho3/cmass-ili/mtnglike/fastpm/L3000-N384
scratchbase=/work/hdd/bdne/maho3/cmass_scratch/harvest/mtnglike/fastpm/L3000-N384

done_n=0; missing=()
for l in $(seq $LHID_MIN $LHID_MAX); do
    if [ -f "$outbase/$l/nbody.h5" ]; then
        done_n=$((done_n+1))
    else
        missing+=($l)
    fi
done
total=$((LHID_MAX - LHID_MIN + 1))
echo "harvested: $done_n / $total"

echo "jobs:"
squeue -u $USER -h -o "  %i %j %T %M" 2>/dev/null | grep -E "3gpch" || echo "  (none)"

echo "scratch:"
du -sh ${scratchbase%/mtnglike*} 2>/dev/null || echo "  (empty)"
quota 2>/dev/null | grep "work/hdd"

# lhids with snapshots in scratch but no producer sentinel = in flight or dead
inflight=(); failed=()
for d in "$scratchbase"/*/; do
    [ -d "$d" ] || continue
    l=$(basename "$d")
    if [ -f "$d/.fastpm_failed" ]; then failed+=($l)
    elif [ ! -f "$d/.fastpm_done" ]; then inflight+=($l)
    fi
done
[ ${#inflight[@]} -gt 0 ] && echo "producing/unfinished in scratch: ${inflight[@]}"
[ ${#failed[@]} -gt 0 ] && echo "FAILED producers (rerun these): ${failed[@]}"

if [ ${#missing[@]} -gt 0 ] && [ ${#missing[@]} -le 30 ]; then
    echo "missing lhids: ${missing[@]}"
elif [ ${#missing[@]} -gt 30 ]; then
    echo "missing lhids: ${missing[@]:0:15} ... (${#missing[@]} total)"
fi
