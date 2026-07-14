#!/bin/bash
#SBATCH --job-name=3gpch # Job name
#SBATCH --nodes=3               # Number of nodes
#SBATCH --ntasks=384            # Number of tasks
#SBATCH --mem=240G              # Amount of memory
#SBATCH --time=2:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task


SLURM_ARRAY_TASK_ID=3002
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
# baseoffset=2000

module load cray-mpich/8.1.32 gsl
export LD_LIBRARY_PATH=/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/gsl-2.8-zty4u3k/lib:$LD_LIBRARY_PATH

source ~/.bashrc
conda activate cmass

lhid=$((SLURM_ARRAY_TASK_ID + baseoffset))

# Command to run for each lhid
cd /u/maho3/git/ltu-cmass

nbody=mtnglike
sim=fastpm
multisnapshot=True
L=3000
N=384

# Shared scratch for transient FastPM snapshots. Must be on a shared
# filesystem (/tmp is node-local and breaks multi-node runs: ranks on other
# nodes cannot read the ICs or write snapshot shards). Cleaned up on exit.
scratchbase=/work/hdd/bdne/maho3/cmass_scratch/${SLURM_JOB_ID}
# Node-local disk (driver node only) for streaming postprocess
localbase=/tmp/$USER/cmass_${SLURM_JOB_ID}
mkdir -p "$scratchbase" "$localbase"
trap 'rm -rf "$scratchbase" "$localbase"' EXIT

# Sample scratch + local usage every 30s to measure peak transient storage
(while true; do
    echo "SCRATCH_USAGE $(date +%H:%M:%S) $(du -sBM $scratchbase 2>/dev/null | cut -f1) LOCAL $(du -sBM $localbase 2>/dev/null | cut -f1)"
    sleep 30
done) &
monitor_pid=$!

extras="nbody.matchIC=0 meta.scratchdir=$scratchbase meta.localdir=$localbase nbody.stream_postprocess=True" #  meta.cosmofile=./params/abacus_cosmologies.txt"

outdir=/work/hdd/bdne/maho3/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"
echo "scratchbase=$scratchbase"

# export TQDM_DISABLE=0
# extras="$extras hydra/job_logging=disabled"


for offset in 0; do # $(seq 0 20 200); do
    loff=$((lhid + offset))
    
    postfix="nbody=$nbody sim=$sim nbody.lhid=$loff multisnapshot=$multisnapshot $extras"

    cfgfile=$outdir/$loff/config.yaml
    h5file=$outdir/$loff/nbody.h5
    if [ -f "$cfgfile" ] || [ -f "$h5file" ]; then
        echo "$outdir/$loff already done/offloaded. Skipping."
    else
        echo "$outdir/$loff not done. Running FastPM."
        python -m cmass.nbody.fastpm $postfix
        # python -m cmass.nbody.postprocess_fastpm $postfix
    fi
done

kill $monitor_pid 2>/dev/null
echo "Peak scratch usage:"
grep -h SCRATCH_USAGE /work/hdd/bdne/maho3/jobout/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_*.out 2>/dev/null | sort -t' ' -k3 -n | tail -1

# Safety net (trap also fires on failure/timeout)
rm -rf "$scratchbase"
