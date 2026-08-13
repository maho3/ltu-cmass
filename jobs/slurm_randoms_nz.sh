#!/bin/bash
#SBATCH --job-name=randoms_nz   # Job name
#SBATCH --array=0-2             # 0=ngc, 1=sgc, 2=mtng
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=32             # Number of tasks
#SBATCH --mem=180GB             # ngc draws 1.6e8 uniform points x 3 snapshots
#SBATCH --time=06:00:00         # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out

# Regenerate the shared survey randoms so they trace 10x the observed CMASS
# n(z) instead of being uniform in comoving volume.
#
# The previous randoms were built with survey.fix_nz=False, which feeds a
# uniform random cube through the lightcone: they carried only the angular
# mask and the z-range cut, so their dN/dz tracked dV/dz to ~3% rather than
# the observed n(z). fix_nz=True downsamples to the target n(z) in
# cmass/lightcone/nz_DR12v5_CMASS_*.dat, and survey.nz_factor scales that
# target so we keep the observed *shape* at 10x the density.
#
# survey.randoms_nbar must exceed the peak target density in every z-bin or
# the downsampling cannot saturate. Required minima measured against the old
# randoms: ngc 1.45x, sgc 1.38x, mtng 2.70x the old 3e-3, so we use 6e-3 for
# ngc/sgc and 1.2e-2 for mtng (doubled to 2.4e-2 below, since mtng's target is
# pre-inflated for the post-finalize octant cut).
#
# survey.randoms=True now also sets skip_fibcoll in the C++ lightcone, so the
# randoms skip the (expensive) fiber-collision pass and the pre-inflated
# downsampling that compensates for it. Mocks are unaffected.
#
# NOTE: this OVERWRITES the randoms in place. Every existing lightcone diag
# summary was computed against the old randoms and becomes inconsistent.

source ~/.bashrc
conda activate cmass

module load gsl
export CPATH=$GSL_ROOT_DIR/include:${CPATH:-}
export LIBRARY_PATH=$GSL_ROOT_DIR/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/u/maho3/anaconda3/envs/cmass/lib:$GSL_ROOT_DIR/lib:${LD_LIBRARY_PATH:-}

# Fail the array task if the run fails, rather than falling through to the
# trailing echo and reporting COMPLETED. Set after sourcing bashrc/module,
# which are not -e clean.
set -eo pipefail

cd /u/maho3/git/ltu-cmass

# export TQDM_DISABLE=0

case $SLURM_ARRAY_TASK_ID in
    0)  # ngc -> quijote3gpch/randoms/L3000-N384/0/ngc_lightcone
        cap=ngc
        nbody=quijote3gpch
        # multisnapshot is pointless for randoms: stitch_lightcone draws an
        # independent uniform cube per snapshot with zero velocities, so the
        # snapshots are statistically identical and there is nothing to
        # interpolate. This halves the draw, 2 cubes -> 1 (15.6 -> 7.8 GB).
        #
        # nbody.zf=0.5 is required, not cosmetic: multisnapshot=False sets
        # asave=[1/(1+zf)], and quijote3gpch has zf=0.3, which the
        # zmin<z<zmax snapshot filter then discards -- leaving zero
        # snapshots and an empty lightcone. abacus (sgc/mtng) already has
        # zf=0.5, which is why they run single-snapshot as-is.
        multisnapshot=False
        nbar=6.0e-3
        nz_factor=10
        extras="nbody.zf=0.5"
        ;;
    1)  # sgc -> abacus/randoms/L2000-N256/0/sgc_lightcone
        cap=sgc
        nbody=abacus
        multisnapshot=False
        nbar=6.0e-3
        nz_factor=10
        extras="meta.cosmofile=./params/abacus_cosmologies.txt"
        ;;
    2)  # mtng -> abacus/randoms/L2000-N256/0/mtng_lightcone
        # mtng's angular cut (ra,dec in [0,90)) is applied in
        # hodlightcone.main AFTER lightcone.finalize(), i.e. after the C++
        # downsamples to the target, so ~30% of the downsampled sample is
        # discarded and a flat factor lands short. These per-bin factors
        # pre-inflate the target by 1/survival; regenerate them with
        #     python scripts/mtng_nz_survival.py --achieved 10 --want 10
        # Pre-cut target is 11.3M (vs 7.9M post-cut), so nbar is raised 2x.
        cap=mtng
        nbody=abacus
        multisnapshot=False
        nbar=2.4e-2
        nz_factor="[15.604,15.133,14.747,14.433,14.171,13.938,13.734,13.594,13.424,13.291]"
        extras="meta.cosmofile=./params/abacus_cosmologies.txt"
        ;;
    *)
        echo "Unknown array index $SLURM_ARRAY_TASK_ID"; exit 1
        ;;
esac

echo "Generating $cap randoms: nbody=$nbody nbar=$nbar factor=$nz_factor"

# sim=randoms and lhid=0 put the output exactly on top of the existing
# randoms, which is what cmass.diagnostics.pypower.get_randoms_file reads.
python -m cmass.survey.hodlightcone \
    nbody=$nbody sim=randoms nbody.lhid=0 \
    multisnapshot=$multisnapshot \
    survey.geometry=$cap \
    survey.randoms=True \
    survey.fix_nz=True \
    "survey.nz_factor=$nz_factor" \
    survey.randoms_nbar=$nbar \
    survey.aug_seed=0 \
    bias.hod.seed=0 \
    bias.hod.noise_uniform=False \
    $extras \
    hydra/job_logging=disabled

echo "Done $cap"
