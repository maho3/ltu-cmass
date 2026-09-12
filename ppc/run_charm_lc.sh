#!/bin/bash
# Stage B of a LIGHTCONE PPC campaign: CHARM (cmass.bias.rho_to_halo).
#
# NOT a SLURM script -- run this by hand on a GPU machine, after
# ppc/slurm_nbody_lc.sh has produced nbody.h5 for every draw.
#
#   usage:  bash ppc/run_charm_lc.sh [first_draw] [last_draw]
#   e.g.    bash ppc/run_charm_lc.sh 0 99
#
# Differs from ppc/run_charm.sh in two ways:
#   * multisnapshot=True. rho_to_halo loops cfg.nbody.asave, so this populates
#     halos.h5 for every snapshot the lightcone will later stitch.
#   * NO charm_ckpt override. jobs/slurm_charm.sh (rundelta), which built the
#     mtnglike suite, passes none either, so HEAD's default is the matching
#     model. If HEAD's default CHARM ever moves off the one that built your
#     training suite, pin it here the way run_charm.sh does.
#
# rho_to_halo reads nbody.h5 and writes halos.h5 into the SAME directory, so
# nothing needs symlinking.

set -u

first=${1:-0}
last=${2:-99}

WDIR=/work/hdd/bdne/maho3/cmass-ili

cd /u/maho3/git/ltu-cmass

tag=obs00000
ppcdir=ppc/mtnglike_fastpm_charm7/testing/Pk0+Pk2+Pk4+Bk0_kmin-0.0_kmax-Bk=0.2__Pk=0.2/$tag
nbody=mtnglike
sim=fastpm
L=3000
N=384
multisnapshot=True

if [ ! -f "./params/ppc_${tag}_cosmo.txt" ]; then
    echo "MISSING: ./params/ppc_${tag}_cosmo.txt"; exit 1
fi

outdir=$WDIR/$ppcdir/$sim/L$L-N$N
echo "outdir=$outdir"

extras="nbody.matchIC=0 meta.cosmofile=./params/ppc_${tag}_cosmo.txt"
extras="$extras hydra/job_logging=disabled"

for lhid in $(seq "$first" "$last"); do
    nbody_file=$outdir/$lhid/nbody.h5
    halo_file=$outdir/$lhid/halos.h5

    if [ ! -f "$nbody_file" ]; then
        echo "draw=$lhid: no $nbody_file yet. Skipping."
        continue
    fi
    if [ -f "$halo_file" ]; then
        echo "draw=$lhid: $halo_file exists. Skipping."
        continue
    fi

    echo "draw=$lhid: running CHARM."
    start=$(date +%s)
    python -m cmass.bias.rho_to_halo \
        nbody=$nbody sim=$sim \
        nbody.suite="'$ppcdir'" nbody.lhid=$lhid \
        multisnapshot=$multisnapshot \
        $extras
    status=$?
    echo "draw=$lhid stage=charm status=$status wall_s=$(( $(date +%s) - start ))"
done
