#!/bin/bash
#SBATCH --job-name=du_cmassili   # Job name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=64       # Parallel du workers
#SBATCH --time=06:00:00          # Time limit
#SBATCH --partition=cpu          # Partition name
#SBATCH --account=bdne-delta-cpu # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

ROOT=/work/hdd/bdne/maho3/cmass-ili
OUT=/work/hdd/bdne/maho3/du_bdne/cmassili
NPROC=64

mkdir -p "$OUT"

# Work units: every <suite>/<sim> dir (depth 2 under cmass-ili), so big
# suites like quijote/quijotelike get split across many workers instead of
# being one giant single-threaded du call.
find "$ROOT" -mindepth 2 -maxdepth 2 -print0 2>/dev/null > "$OUT/worklist"
echo "work units: $(tr -cd '\0' < "$OUT/worklist" | wc -c)"

xargs -0 -n1 -P "$NPROC" du -s --block-size=1 -- < "$OUT/worklist" \
    2> "$OUT/errors.log" > "$OUT/raw.tsv"

# Roll up by suite/sim (top 2 path components) and by suite alone
awk -F'\t' -v root="$ROOT" '
{
  path = $2
  sub("^" root "/", "", path)
  split(path, p, "/")
  suitesim = p[1] "/" p[2]
  bytes_ss[suitesim] += $1
  bytes_s[p[1]] += $1
}
END {
  for (k in bytes_ss) printf "%s\t%d\n", k, bytes_ss[k] > "'"$OUT"'/by_suitesim.tsv"
  for (k in bytes_s) printf "%s\t%d\n", k, bytes_s[k] > "'"$OUT"'/by_suite.tsv"
}
' "$OUT/raw.tsv"
sort -k2 -nr -o "$OUT/by_suitesim.tsv" "$OUT/by_suitesim.tsv"
sort -k2 -nr -o "$OUT/by_suite.tsv" "$OUT/by_suite.tsv"

printf "\n%-24s %10s\n" SUITE TiB
awk -F'\t' '{printf "%-24s %10.3f\n", $1, $2/1024^4}' "$OUT/by_suite.tsv"

printf "\n%-40s %10s\n" SUITE/SIM TiB
awk -F'\t' '{printf "%-40s %10.3f\n", $1, $2/1024^4}' "$OUT/by_suitesim.tsv"

echo
echo "per-entry sizes:  $OUT/raw.tsv"
echo "unreadable paths: $OUT/errors.log ($(wc -l < "$OUT/errors.log") lines)"
