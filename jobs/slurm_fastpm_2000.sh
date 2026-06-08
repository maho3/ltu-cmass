#!/bin/bash
#SBATCH --job-name=abacus2000   # Job name
#SBATCH --array=2376-3999%50    # array index == lhid; %25 caps concurrent tasks
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=128            # Number of tasks
#SBATCH --mem=240G              # Amount of memory
#SBATCH --time=3:00:00          # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# Run FastPM for lhid 2000-3999, one simulation per array task.
#
# Storage strategy: the ~76GB of transient FastPM particle snapshots are staged
# on node-local /tmp (740GB SSD, per-node, auto-purged at job end) via
# meta.scratchdir, so they never touch the quota'd /work/hdd. Only the final
# ~2.7GB nbody.h5 (+ config.yaml) is written to /work/hdd.

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

module load cray-mpich/8.1.32 gsl
export LD_LIBRARY_PATH=/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/gsl-2.8-zty4u3k/lib:$LD_LIBRARY_PATH

source ~/.bashrc
conda activate cmass

lhid=$SLURM_ARRAY_TASK_ID

cd /u/maho3/git/ltu-cmass

nbody=abacuslike
sim=fastpm
multisnapshot=True

# Node-local scratch base for transient particle snapshots (per job+node).
# get_source_path appends suite/sim/L-N/lhid underneath this.
scratchbase=/tmp/$USER/cmass_${SLURM_ARRAY_JOB_ID}
mkdir -p "$scratchbase"

extras="nbody.matchIC=0 meta.scratchdir=$scratchbase"
L=2000
N=256

outdir=/work/hdd/bdne/maho3/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"
echo "scratchbase=$scratchbase"

postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid multisnapshot=$multisnapshot $extras"

# Skip if this lhid is already done (resume-safe). We key on config.yaml, which
# is written last (so a crashed mid-run leaves an empty dir and re-runs), and
# which the Globus offload KEEPS as a tombstone after deleting nbody.h5 (so an
# offloaded sim is still skipped). nbody.h5 is an extra fallback.
cfgfile=$outdir/$lhid/config.yaml
h5file=$outdir/$lhid/nbody.h5
if [ -f "$cfgfile" ] || [ -f "$h5file" ]; then
    echo "$outdir/$lhid already done/offloaded. Skipping."
else
    # # --- Quota guard: refuse to start if the /work/hdd allocation is near its
    # # quota, so the final ~2.7GB nbody.h5 write can't fail mid-job. (The ~76GB of
    # # intermediates go to /tmp, not /work/hdd.) We gate on the SOFT quota (where
    # # Lustre starts the grace clock). On trip, exit non-zero so the task shows as
    # # not-done; offload to free space, then resubmit. The project id comes from
    # # `lfs project -d /work/hdd/bdne/$USER`.
    # HDD_PROJID=18904
    # RESERVE_GB=200
    # read -r used_kb soft_kb hard_kb < <(
    #     lfs quota -q -p "$HDD_PROJID" /work/hdd 2>/dev/null \
    #         | awk '/work\/hdd/ {gsub(/\*/,""); print $2, $3, $4; exit}'
    # )
    # ceil_kb=${soft_kb:-0}; [ "${ceil_kb}" -gt 0 ] 2>/dev/null || ceil_kb=${hard_kb:-0}
    # if [ -n "${used_kb:-}" ] && [ "${ceil_kb:-0}" -gt 0 ]; then
    #     free_gb=$(( (ceil_kb - used_kb) / 1048576 ))
    #     echo "Quota /work/hdd: used=$((used_kb/1048576))GB ceiling=$((ceil_kb/1048576))GB free=${free_gb}GB (reserve ${RESERVE_GB}GB)"
    #     if [ "$free_gb" -lt "$RESERVE_GB" ]; then
    #         echo "QUOTA GUARD: only ${free_gb}GB below quota (< ${RESERVE_GB}GB). Refusing lhid $lhid; offload + resubmit."
    #         rm -rf "$scratchbase"
    #         exit 1
    #     fi
    # else
    #     echo "WARNING: could not read /work/hdd quota (projid $HDD_PROJID); proceeding without guard." >&2
    # fi

    echo "$outdir/$lhid not done. Running FastPM."
    python -m cmass.nbody.fastpm $postfix
fi

# Safety net: clear this job's node-local scratch (also auto-purged at job end).
rm -rf "$scratchbase"
