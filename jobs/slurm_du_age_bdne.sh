#!/bin/bash
#SBATCH --job-name=du_age_bdne  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=64      # Parallel find workers
#SBATCH --time=12:00:00         # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

# Per-user storage broken down by last-access age, binned like the admin report.
# Size = allocated blocks (du-equivalent). stat via find does not update atime.

ROOT=/work/hdd/bdne
OUT=/work/hdd/bdne/maho3/du_bdne
NPROC=64
export ROOT NOW=$(date +%s)

mkdir -p "$OUT"

find "$ROOT" -mindepth 2 -maxdepth 2 -print0 2>/dev/null > "$OUT/age_worklist"
echo "work units: $(tr -cd '\0' < "$OUT/age_worklist" | wc -c)"

# Emit: user \t bin \t bytes \t nfiles  (aggregated per work unit)
scan() {
  local user=${1#"$ROOT"/}; user=${user%%/*}
  find "$1" -type f -printf '%A@ %b\n' | awk -v now="$NOW" -v u="$user" '
  {
    d = (now - $1) / 86400
    b = d < 1 ? 0 : d < 7 ? 1 : d < 30 ? 2 : d < 91 ? 3 : d < 182 ? 4 : d < 365 ? 5 : 6
    s[b] += $2 * 512; n[b]++
  }
  END { for (b in s) printf "%s\t%d\t%.0f\t%d\n", u, b, s[b], n[b] }'
}
export -f scan

xargs -0 -n1 -P "$NPROC" bash -c 'scan "$1"' _ < "$OUT/age_worklist" \
    2> "$OUT/age_errors.log" > "$OUT/age_raw.tsv"

# Roll up: user \t bin \t bytes \t nfiles
awk -F'\t' '{k=$1 "\t" $2; s[k]+=$3; n[k]+=$4}
            END {for (k in s) printf "%s\t%.0f\t%d\n", k, s[k], n[k]}' \
    "$OUT/age_raw.tsv" | sort -k1,1 -k2,2n > "$OUT/age_by_user.tsv"

# Print one admin-style table for user $1 ("ALL" = whole project)
table() {
  awk -F'\t' -v u="$1" '
  function h(x) { return x >= 1024^4 ? sprintf("%.1fTi", x/1024^4) : sprintf("%.0fGi", x/1024^3) }
  BEGIN { split("0 - 1 day access|1 day - 1 week access|1 week - 1 month access|1 month - 3 month access|3 month - 6 month access|6 month - 1 year access|1 year+ access", L, "|") }
  u == "ALL" || $1 == u { s[$2] += $3; n[$2] += $4 }
  END { for (b = 0; b < 7; b++) printf "  %-26s %9d Files; %s\n", L[b+1] ":", n[b], h(s[b]) }' \
    "$OUT/age_by_user.tsv"
}

{
  echo "=== PROJECT TOTAL ==="; table ALL
  printf "\n%-24s %10s %10s\n" USER TiB ">6mo_TiB"
  awk -F'\t' '{t[$1]+=$3; if ($2>=5) o[$1]+=$3}
              END {for (u in t) printf "%-24s %10.3f %10.3f\n", u, t[u]/1024^4, o[u]/1024^4}' \
      "$OUT/age_by_user.tsv" | sort -k2 -nr | tee "$OUT/age_users.txt"
  for u in $(awk '{print $1}' "$OUT/age_users.txt"); do
    echo; echo "=== $u ==="; table "$u"
  done
} > "$OUT/age_report.txt"
rm -f "$OUT/age_users.txt"

cat "$OUT/age_report.txt"
echo
echo "rolled-up table:  $OUT/age_by_user.tsv  (user, bin 0-6, bytes, nfiles)"
echo "unreadable paths: $OUT/age_errors.log ($(wc -l < "$OUT/age_errors.log") lines)"
