#!/bin/bash
#SBATCH --job-name=ppc_nbody    # Job name
#SBATCH --array=0-99%50         # One task per posterior draw (array idx == draw id == lhid)
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=128            # Number of tasks
#SBATCH --mem=240G              # Amount of memory
#SBATCH --time=3:00:00          # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# Stage A of the PPC campaign: FastPM only.
#   Stage B  = CHARM (cmass.bias.rho_to_halo), run by hand on a GPU machine.
#   Stage C  = ppc/slurm_hod.sh (apply_hod + diagnostics).
#
# The forward chain replicates jobs/slurm_abacuslike_bias.sh, which produced the
# training suite: nbody=abacuslike, bias=zheng_composite, multisnapshot=False,
# nbody.zf=0.500015 (a=0.666660, the analysis snapshot).
#
# Cosmology per draw comes from params/ppc_<tag>_cosmo.txt, written by
# ppc/draw.py, indexed by nbody.lhid. matchIC=0 so each draw gets its
# own IC phase (gen_white_noise is seeded by lhid).
#
# Storage strategy copied from jobs/slurm_fastpm_2000.sh: the transient FastPM
# particle snapshots are staged on node-local /tmp via meta.scratchdir so they
# never touch the quota'd /work/hdd (which is at 17.6T of a 19.5T soft quota).

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

module load cray-mpich/8.1.32 gsl
export LD_LIBRARY_PATH=/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/gsl-2.8-zty4u3k/lib:$LD_LIBRARY_PATH

source ~/.bashrc
conda activate cmass

lhid=$SLURM_ARRAY_TASK_ID

cd /u/maho3/git/ltu-cmass

tag=obs01880
ppcdir=ppc/abacuslike_fastpm_charm6_comphod/zPk0+zPk2+zPk4_kmin-0.0_kmax-0.4/$tag

nbody=abacuslike
sim=fastpm
L=2000
N=256
multisnapshot=False

# Node-local scratch base for transient particle snapshots (per job+node).
# get_source_path appends suite/sim/L-N/lhid underneath this.
scratchbase=/tmp/$USER/cmass_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$scratchbase"

extras="bias=zheng_composite nbody.zf=0.500015 nbody.matchIC=0"
extras="$extras meta.cosmofile=./params/ppc_${tag}_cosmo.txt"
extras="$extras meta.scratchdir=$scratchbase"
extras="$extras hydra/job_logging=disabled"

export TQDM_DISABLE=0

outdir=/work/hdd/bdne/maho3/cmass-ili/$ppcdir/$sim/L$L-N$N
echo "outdir=$outdir"
echo "scratchbase=$scratchbase"

postfix="nbody=$nbody sim=$sim nbody.suite='$ppcdir' nbody.lhid=$lhid"
postfix="$postfix multisnapshot=$multisnapshot $extras"

# Resume guard, same convention as slurm_fastpm_2000.sh: config.yaml is written
# last, so a crashed mid-run leaves an incomplete dir and re-runs.
cfgfile=$outdir/$lhid/config.yaml
h5file=$outdir/$lhid/nbody.h5
if [ -f "$cfgfile" ] || [ -f "$h5file" ]; then
    echo "$outdir/$lhid already done. Skipping."
else
    echo "$outdir/$lhid not done. Running FastPM for draw $lhid."
    start=$(date +%s)
    python -m cmass.nbody.fastpm $postfix
    status=$?
    echo "draw=$lhid stage=nbody status=$status wall_s=$(( $(date +%s) - start ))"
fi

# Safety net: clear this job's node-local scratch (also auto-purged at job end).
rm -rf "$scratchbase"
