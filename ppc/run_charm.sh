#!/bin/bash
# Stage B of the PPC campaign: CHARM (cmass.bias.rho_to_halo).
#
# NOT a SLURM script -- run this by hand on a GPU machine, after
# ppc/slurm_nbody.sh has produced nbody.h5 for every draw.
#
#   usage:  bash ppc/run_charm.sh [first_draw] [last_draw]
#   e.g.    bash ppc/run_charm.sh 0 19
#
# CHARM VERSION: the training suite (abacuslike/fastpm_charm7_cosmoHOD) was
# built with charm7 = charm_joint_v19.pth, which is also HEAD's default in
# cmass/bias/rho_to_halo.py. CHARM_CKPT pins it anyway, so the campaign keeps
# testing the forward model the posterior was trained on if that default moves.
#
# rho_to_halo reads nbody.h5 and writes halos.h5 into the SAME directory, so
# nothing needs symlinking.

set -u

first=${1:-0}
last=${2:-19}

# --- adjust these three for the machine you are running on -------------------
WDIR=/work/hdd/bdne/maho3/cmass-ili
CHARM_CKPT=$WDIR/scratch/charm_joint_v19.pth
CHARM_YAML=/u/maho3/git/CHARM/run_configs/TRAIN_CHARM_JOINT_v2vel_finetune2.yaml
# -----------------------------------------------------------------------------

cd /u/maho3/git/ltu-cmass

tag=obs00038
ppcdir=ppc/abacuslike_fastpm_charm7_cosmoHOD/zPk0+zPk2+zPk4_kmin-0.0_kmax-0.4/testing/abacus_nbody_comp_gridnoise/$tag
L=2000
N=256

for f in "$CHARM_CKPT" "$CHARM_YAML" "./params/ppc_${tag}_cosmo.txt"; do
    if [ ! -f "$f" ]; then echo "MISSING: $f"; exit 1; fi
done

outdir=$WDIR/$ppcdir/fastpm/L$L-N$N
echo "outdir=$outdir"
echo "ckpt=$CHARM_CKPT"

for lhid in $(seq "$first" "$last"); do
    if [ ! -f "$outdir/$lhid/nbody.h5" ]; then
        echo "draw=$lhid: no nbody.h5 yet, skipping"
        continue
    fi
    if [ -f "$outdir/$lhid/halos.h5" ]; then
        echo "draw=$lhid: halos.h5 exists, skipping"
        continue
    fi

    start=$(date +%s)
    python -m cmass.bias.rho_to_halo \
        nbody=abacuslike sim=fastpm bias=zheng_composite \
        nbody.suite="'$ppcdir'" nbody.lhid=$lhid \
        multisnapshot=False nbody.zf=0.500015 nbody.matchIC=0 \
        meta.wdir=$WDIR \
        meta.cosmofile=./params/ppc_${tag}_cosmo.txt \
        +bias.halo.charm_ckpt=$CHARM_CKPT \
        +bias.halo.charm_yaml=$CHARM_YAML \
        hydra/job_logging=disabled
    echo "draw=$lhid stage=charm status=$? wall_s=$(( $(date +%s) - start ))"
done
