#!/bin/bash
#SBATCH --job-name=ppc_lightcone  # Job name
#SBATCH --array=0-99%50         # One task per posterior draw (array idx == draw id == lhid)
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=32             # Number of tasks
#SBATCH --mem=64G               # Amount of memory
#SBATCH --time=4:00:00          # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# Stage C of a LIGHTCONE PPC campaign: hodlightcone + diagnostics.
#   Stage A = ppc/slurm_nbody_lc.sh, Stage B = ppc/run_charm_lc.sh,
#   Stage D = ppc/slurm_collect.sh (with --ppc_dir).
#
# Mirrors the mtng_lightcone block of jobs/slurm_mtng_bias.sh on the rundelta
# branch, which produced the training summaries.
#
# NOTE cmass.survey.hodlightcone REPLACES cmass.bias.apply_hod -- it reads
# halos.h5 and applies the HOD while stitching the lightcone, so there is no
# separate galaxies/ stage here. It still goes through parse_hod/parse_noise,
# so the per-draw overrides written by ppc/draw.py inject exactly as they do in
# the snapshot campaign.
#
# Deviations from the training run, matching ppc/slurm_hod.sh:
#   * HOD parameters and noise are INJECTED per draw from overrides/<id>.txt
#     rather than sampled. bias.hod.seed=1 still resolves to lhid*1e4+1 inside
#     parse_hod, so galaxy-placement RNG varies per draw as in training.
#   * noise=fixed instead of noise=reciprocal, so noise_radial/transverse take
#     the drawn values rather than being sampled.
#   * diag.summaries pinned to match the training run rather than inheriting
#     today's diag/default.yaml.
#
# To run a different geometry, change `cap` -- the diag flag, survey.geometry
# and bias.hod.custom_prior all follow it. The experiment's tracer must match
# (<cap>_lightcone).

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

source ~/.bashrc
conda activate cmass

module load gsl
export CPATH=$GSL_ROOT_DIR/include:$CPATH
export LIBRARY_PATH=$GSL_ROOT_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/u/maho3/anaconda3/envs/cmass/lib:$GSL_ROOT_DIR/lib:$LD_LIBRARY_PATH

lhid=$SLURM_ARRAY_TASK_ID

cd /u/maho3/git/ltu-cmass

cap=mtng          # mtng | ngc | sgc | simbig
tag=obs00000
ppcdir=ppc/mtnglike_fastpm_charm7/testing/Pk0+Pk2+Pk4+Bk0_kmin-0.0_kmax-Bk=0.2__Pk=0.2/$tag
ppcbase=/work/hdd/bdne/maho3/cmass-ili/$ppcdir

nbody=mtnglike
sim=fastpm
L=3000
N=384
multisnapshot=True

hod_seed=1
aug_seed=1        # ppc/draw.py records hod00001_aug00001.h5 in the manifest

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
postfix="$postfix multisnapshot=$multisnapshot nbody.matchIC=0"
postfix="$postfix bias=zhenginterp_biased noise=fixed"
postfix="$postfix bias.hod.seed=$hod_seed bias.hod.from_samples=False"
postfix="$postfix bias.hod.noise_uniform=False"
postfix="$postfix survey.geometry=$cap survey.aug_seed=$aug_seed"
postfix="$postfix bias.hod.custom_prior=$cap"
postfix="$postfix diag.$cap=True diag.high_res=True"
postfix="$postfix meta.cosmofile=./params/ppc_${tag}_cosmo.txt"
postfix="$postfix $draw_extras hydra/job_logging=disabled"

export TQDM_DISABLE=0

printf -v hod_str "%05d" $hod_seed
printf -v aug_str "%05d" $aug_seed
cat_file=$outdir/$lhid/${cap}_lightcone/hod${hod_str}_aug${aug_str}.h5
diag_file=$outdir/$lhid/diag/${cap}_lightcone/hod${hod_str}_aug${aug_str}.h5

if [ -f "$diag_file" ]; then
    echo "draw=$lhid already has $diag_file. Skipping."
else
    start=$(date +%s)
    if [ -f "$cat_file" ]; then
        echo "draw=$lhid: lightcone catalog exists, only summarizing."
        st_lc=0
    else
        python -m cmass.survey.hodlightcone $postfix
        st_lc=$?
    fi
    python -m cmass.diagnostics.summ $postfix \
        diag.from_scratch=True 'diag.summaries=[nz,Pk,Bk]'
    st_summ=$?
    echo "draw=$lhid stage=lightcone status=$st_lc summ_status=$st_summ wall_s=$(( $(date +%s) - start ))"
fi
