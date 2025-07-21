#!/bin/bash
set -e

module restore cmass
conda activate cmass


# Command to run for each lhid
cd /jet/home/mho1/git/ltu-cmass

# Nhod=5

echo "~~~ GENERATE SIMBIG RANDOMS ~~~"

nbody=quijote
sim=randoms
noise_uniform_invoxel=False  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=fixed
diag_from_scratch=True
extras="survey.randoms=True"

L=1000
N=128
lhid=0
hod_seed=0
aug_seed=0

postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
postfix="$postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed"
postfix="$postfix diag.from_scratch=$diag_from_scratch"
postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
postfix="$postfix noise=$noise"
postfix="$postfix $extras"

python -m cmass.survey.hodlightcone survey.geometry=simbig $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed




# echo "~~~ GENERATE NGC RANDOMS ~~~"

# nbody=quijote3gpch
# sim=randoms
# noise_uniform_invoxel=False  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
# noise=fixed
# diag_from_scratch=True
# extras="survey.randoms=True noise.params.radial=4 noise.params.transverse=4"

# L=3000
# N=384
# lhid=2000
# hod_seed=0
# aug_seed=0

# postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
# postfix="$postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed"
# postfix="$postfix diag.from_scratch=$diag_from_scratch"
# postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
# postfix="$postfix noise=$noise"
# postfix="$postfix $extras"

# python -m cmass.survey.hodlightcone survey.geometry=ngc $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
