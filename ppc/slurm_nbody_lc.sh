#!/bin/bash
#SBATCH --job-name=ppclc_nbody  # Job name
#SBATCH --array=0-99%25         # One task per posterior draw (array idx == draw id == lhid)
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=128            # Number of tasks
#SBATCH --mem=240G              # Amount of memory
#SBATCH --time=8:00:00          # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# Stage A of a LIGHTCONE PPC campaign: FastPM, multi-snapshot.
#   Stage B = ppc/run_charm_lc.sh (CHARM, by hand on a GPU machine)
#   Stage C = ppc/slurm_lightcone.sh (hodlightcone + diagnostics)
#
# Replicates jobs/slurm_fastpm_3gpch.sh on the rundelta branch, which produced
# the mtnglike training suite: nbody=mtnglike, multisnapshot=True, L3000-N384.
#
# multisnapshot=True is REQUIRED here and is the main difference from the
# snapshot campaign: cmass.survey.hodlightcone stitches the lightcone out of
# cfg.nbody.asave, so a single snapshot gives it nothing to interpolate over.
# It costs proportionally more wall time and scratch than slurm_nbody.sh.
#
# Cosmology per draw comes from params/ppc_<tag>_cosmo.txt, written by
# ppc/draw.py, indexed by nbody.lhid. matchIC=0 so each draw gets its own IC
# phase (gen_white_noise is seeded by lhid).

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

module load cray-mpich/8.1.32 gsl
export LD_LIBRARY_PATH=/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/gsl-2.8-zty4u3k/lib:$LD_LIBRARY_PATH

source ~/.bashrc
conda activate cmass

lhid=$SLURM_ARRAY_TASK_ID

cd /u/maho3/git/ltu-cmass

tag=obs00000
ppcdir=ppc/mtnglike_fastpm_charm7/testing/Pk0+Pk2+Pk4+Bk0_kmin-0.0_kmax-Bk=0.2__Pk=0.2/$tag

nbody=mtnglike
sim=fastpm
L=3000
N=384
multisnapshot=True

# Transient FastPM particle snapshots. Multi-snapshot keeps every asave entry
# alive until postprocessing, so this needs more room than the snapshot chain.
scratchbase=/work/hdd/bdne/maho3/cmass_scratch/ppc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
localbase=/tmp/$USER/cmass_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$scratchbase" "$localbase"
trap 'rm -rf "$scratchbase" "$localbase"' EXIT

extras="nbody.matchIC=0 nbody.stream_postprocess=True"
extras="$extras meta.cosmofile=./params/ppc_${tag}_cosmo.txt"
extras="$extras meta.scratchdir=$scratchbase meta.localdir=$localbase"
extras="$extras hydra/job_logging=disabled"

export TQDM_DISABLE=0

outdir=/work/hdd/bdne/maho3/cmass-ili/$ppcdir/$sim/L$L-N$N
echo "outdir=$outdir"
echo "scratchbase=$scratchbase"

postfix="nbody=$nbody sim=$sim nbody.suite='$ppcdir' nbody.lhid=$lhid"
postfix="$postfix multisnapshot=$multisnapshot $extras"

# Resume guard: config.yaml is written last, so a crashed mid-run leaves an
# incomplete dir and re-runs.
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

rm -rf "$scratchbase" "$localbase"
