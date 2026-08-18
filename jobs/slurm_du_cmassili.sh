#!/bin/bash
#SBATCH --job-name=du_cmassili  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=64      # Parallel du workers
#SBATCH --time=24:00:00         # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

ROOT=/work/hdd/bdne/maho3/cmass-ili
OUT=$ROOT/accounting.txt
ERR=$ROOT/accounting.err
NPROC=64

# Work units: every depth-2 entry under $ROOT, i.e. $ROOT/*/*
find "$ROOT" -mindepth 2 -maxdepth 2 -print0 2>/dev/null > /tmp/du_cmassili_worklist
echo "work units: $(tr -cd '\0' < /tmp/du_cmassili_worklist | wc -c)"

# Stream results as they complete, so a timeout still leaves partial data
: > "$OUT"
xargs -0 -n1 -P "$NPROC" du -sh -- < /tmp/du_cmassili_worklist \
    2> "$ERR" >> "$OUT"

# Sort largest-first in place
sort -hr -o "$OUT" "$OUT"

echo
echo "sizes:            $OUT ($(wc -l < "$OUT") entries)"
echo "unreadable paths: $ERR ($(wc -l < "$ERR") lines)"
echo
head -30 "$OUT"
