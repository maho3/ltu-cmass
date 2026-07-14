#!/bin/bash
#SBATCH --job-name=du_bdne      # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=64      # Parallel du workers
#SBATCH --time=12:00:00         # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

ROOT=/work/hdd/bdne
OUT=/work/hdd/bdne/maho3/du_bdne
NPROC=64

mkdir -p "$OUT"

# Work units: every depth-1 entry under each user dir (files and subdirs alike)
find "$ROOT" -mindepth 2 -maxdepth 2 -print0 2>/dev/null > "$OUT/worklist"
echo "work units: $(tr -cd '\0' < "$OUT/worklist" | wc -c)"

xargs -0 -n1 -P "$NPROC" du -s --block-size=1 -- < "$OUT/worklist" \
    2> "$OUT/errors.log" > "$OUT/raw.tsv"

# Roll up by user (2nd path component)
awk -F'\t' -v root="$ROOT" '
{
  path = $2
  sub("^" root "/", "", path)
  split(path, p, "/")
  bytes[p[1]] += $1
  n[p[1]]++
}
END { for (u in bytes) printf "%s\t%d\t%d\n", u, bytes[u], n[u] }
' "$OUT/raw.tsv" | sort -k2 -nr > "$OUT/by_user.tsv"

printf "\n%-24s %10s %8s\n" USER TiB ENTRIES
awk -F'\t' '{printf "%-24s %10.3f %8d\n", $1, $2/1024^4, $3; t+=$2}
            END {printf "%-24s %10.3f\n", "TOTAL", t/1024^4}' "$OUT/by_user.tsv"

echo
echo "per-entry sizes:  $OUT/raw.tsv"
echo "unreadable paths: $OUT/errors.log ($(wc -l < "$OUT/errors.log") lines)"
