#!/bin/bash
set -e

module restore cmass
conda activate cmass


# Command to run for each lhid
cd /jet/home/mho1/git/ltu-cmass

# Nhod=5

echo "~~~ RUNNING NGC DIAGNOSTICS ~~~"

nbody=cmass_ngc
sim=cmass_ngc
noise_uniform_invoxel=False  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=fixed
diag_from_scratch=True
extras="noise.params.radial=4 noise.params.transverse=4"

L=3000
N=384
lhid=0
hod_seed=0
aug_seed=0

postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
postfix="$postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed"
postfix="$postfix diag.from_scratch=$diag_from_scratch"
postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
postfix="$postfix noise=$noise"
postfix="$postfix $extras"

python -m cmass.diagnostics.summ diag.ngc=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 



echo "~~~ RUNNING SGC DIAGNOSTICS ~~~"

nbody=cmass_sgc
sim=cmass_sgc
noise_uniform_invoxel=False  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=fixed
diag_from_scratch=True
extras="noise.params.radial=4 noise.params.transverse=4"

L=2000
N=256
lhid=0
hod_seed=0
aug_seed=0

postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
postfix="$postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed"
postfix="$postfix diag.from_scratch=$diag_from_scratch"
postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
postfix="$postfix noise=$noise"
postfix="$postfix $extras"

python -m cmass.diagnostics.summ diag.sgc=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
