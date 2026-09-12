#!/bin/bash
#SBATCH --job-name=ppc_hod      # Job name
#SBATCH --array=0-99%50         # One task per posterior draw (array idx == draw id == lhid)
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16             # Number of tasks
#SBATCH --mem=64G               # Amount of memory
#SBATCH --time=2:00:00          # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# Stage C of the PPC campaign: apply_hod + diagnostics.
# Run after stage A (ppc/slurm_nbody.sh) and stage B (ppc/run_charm.sh).
#
# Mirrors jobs/slurm_abacuslike_bias.sh, which produced the training summaries:
#   bias=zheng_composite, noise_uniform=False, multisnapshot=False,
#   nbody.zf=0.500015, diag.from_scratch=True, rm_galaxies=True.
#
# Two deliberate differences from that job, both required by the PPC:
#   * HOD parameters and noise are INJECTED per draw from overrides/<id>.txt
#     rather than sampled. bias.hod.seed=1 still resolves to lhid*1e4+1 inside
#     parse_hod (so galaxy-placement RNG varies per draw, as in training), but
#     the theta dict is applied afterwards and overwrites all 10 sampled
#     params. Verified: resolved theta == the drawn theta, exactly.
#   * noise=fixed instead of noise=reciprocal, so noise_radial/transverse take
#     the drawn values instead of being redrawn from the prior.
#
# diag.summaries is pinned to [Pk,Bk] because HEAD's diag/default.yaml now also
# lists nz, which the training runs did not have.

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

source ~/.bashrc
conda activate cmass

lhid=$SLURM_ARRAY_TASK_ID

cd /u/maho3/git/ltu-cmass

tag=obs01880
ppcdir=ppc/abacuslike_fastpm_charm6_comphod/zPk0+zPk2+zPk4_kmin-0.0_kmax-0.4/$tag
ppcbase=/work/hdd/bdne/maho3/cmass-ili/$ppcdir

nbody=abacuslike
sim=fastpm
L=2000
N=256
multisnapshot=False
rm_galaxies=True

outdir=$ppcbase/$sim/L$L-N$N
ovrfile=$ppcbase/overrides/$lhid.txt

if [ ! -f "$outdir/$lhid/halos.h5" ]; then
    echo "draw=$lhid: no halos.h5 (CHARM not run yet). Skipping."
    exit 0
fi
if [ ! -f "$ovrfile" ]; then
    echo "draw=$lhid: no override file at $ovrfile. Aborting."
    exit 1
fi

# bias.hod.theta={...} and the two noise values. No spaces inside the braces,
# so this splices in as three words.
draw_extras=$(cat "$ovrfile")
echo "draw=$lhid overrides: $draw_extras"

postfix="nbody=$nbody sim=$sim nbody.suite='$ppcdir' nbody.lhid=$lhid"
postfix="$postfix multisnapshot=$multisnapshot nbody.zf=0.500015 nbody.matchIC=0"
postfix="$postfix bias=zheng_composite noise=fixed"
postfix="$postfix bias.hod.seed=1 bias.hod.from_samples=False"
postfix="$postfix bias.hod.noise_uniform=False"
postfix="$postfix meta.cosmofile=./params/ppc_${tag}_cosmo.txt"
postfix="$postfix $draw_extras hydra/job_logging=disabled"

export TQDM_DISABLE=0

diag_file=$outdir/$lhid/diag/galaxies/hod00001.h5
if [ -f "$diag_file" ]; then
    echo "draw=$lhid: $diag_file exists. Skipping."
else
    start=$(date +%s)
    python -m cmass.bias.apply_hod $postfix
    st_hod=$?
    python -m cmass.diagnostics.summ $postfix \
        diag.galaxy=True diag.from_scratch=True 'diag.summaries=[Pk,Bk]'
    st_summ=$?
    echo "draw=$lhid stage=hod status=$st_hod summ_status=$st_summ wall_s=$(( $(date +%s) - start ))"
fi

# Trash collection. Only removes galaxies/, which this job created; nbody.h5
# and halos.h5 are left alone.
if [ "$rm_galaxies" = "True" ]; then
    echo "Removing galaxies/ for draw=$lhid"
    rm -rf "$outdir/$lhid/galaxies"
fi
