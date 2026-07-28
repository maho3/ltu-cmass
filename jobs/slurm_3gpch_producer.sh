#!/bin/bash
#SBATCH --job-name=3gpch_prod
#SBATCH --array=3000-3014%4
#SBATCH --nodes=3
#SBATCH --ntasks=384
#SBATCH --mem=240G
#SBATCH --time=1:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out

# FastPM producer: runs the simulation only (no postprocess), leaves
# snapshots in the shared harvest scratch for the harvester job.

module load cray-mpich/8.1.32 gsl
export LD_LIBRARY_PATH=/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/gsl-2.8-zty4u3k/lib:$LD_LIBRARY_PATH

source ~/.bashrc
conda activate cmass
cd /u/maho3/git/ltu-cmass

lhid=$SLURM_ARRAY_TASK_ID

nbody=mtnglike
sim=fastpm
L=3000
N=384

scratchbase=/work/hdd/bdne/maho3/cmass_scratch/harvest
outdir=/work/hdd/bdne/maho3/cmass-ili/$nbody/$sim/L$L-N$N
workdir=$scratchbase/$nbody/$sim/L$L-N$N/$lhid

if [ -f "$outdir/$lhid/nbody.h5" ]; then
    echo "$outdir/$lhid/nbody.h5 exists. Skipping."
    exit 0
fi

# Quota guard: a producer writes ~322GB of raw snapshots. If the shared
# /work/hdd/bdne allocation is already near full, starting would risk the
# hard limit (21.48T) and deadlock the harvesters (which can't free space
# without first writing a temp file). Refuse to start; the lhid simply stays
# missing and is picked up by the next producer resubmission. Threshold is
# below the soft limit (19.53T) so that up to %throttle producers can each
# add their footprint and still stay under the hard limit.
QUOTA_MAX_T=19.0
usage=$(quota 2>/dev/null | awk -F'|' '/work\/hdd\/bdne/ {gsub(/ /,"",$3); print $3}')
usage=${usage%\*}
if [ "${usage: -1}" = "T" ] && \
   awk "BEGIN{exit !(${usage%?} >= $QUOTA_MAX_T)}"; then
    echo "QUOTA GUARD: /work/hdd/bdne at ${usage} >= ${QUOTA_MAX_T}T; " \
         "refusing to produce lhid=$lhid (will rerun once harvest frees space)"
    exit 1
fi

extras="nbody.matchIC=0 meta.scratchdir=$scratchbase"
extras="$extras nbody.postprocess=False nbody.harvest=True"

echo "Producing lhid=$lhid (snapshots -> $workdir)"
if python -m cmass.nbody.fastpm nbody=$nbody sim=$sim nbody.lhid=$lhid \
        multisnapshot=True $extras; then
    touch "$workdir/.fastpm_done"
    echo "lhid=$lhid produced OK"
else
    touch "$workdir/.fastpm_failed"
    echo "lhid=$lhid FAILED"
    exit 1
fi
