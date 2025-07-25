#!/bin/bash
set -e

module restore cmass
conda activate cmassrun


# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run

# make the lightcone script
cd cmass/lightcone
make
cd ../..


echo "~~~ GENERATE SIMBIG RANDOMS ~~~"

nbody=quijote
sim=randoms
noise_uniform_invoxel=False  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=fixed
diag_from_scratch=True
extras="survey.randoms=True"

L=1000
N=128
hod_seed=0
aug_seed=0

for lhid in {0..8}; do
    postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
    postfix="$postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed"
    postfix="$postfix diag.from_scratch=$diag_from_scratch"
    postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
    postfix="$postfix noise=$noise"
    postfix="$postfix $extras"

    python -m cmass.survey.hodlightcone survey.geometry=simbig $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
done



echo "~~~ GENERATE SGC/MTNG RANDOMS ~~~"

nbody=abacus
sim=randoms
noise_uniform_invoxel=False  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=fixed
diag_from_scratch=True
extras="survey.randoms=True"

L=2000
N=256
hod_seed=0
aug_seed=0

for lhid in {0..8}; do
    postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
    postfix="$postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed"
    postfix="$postfix diag.from_scratch=$diag_from_scratch"
    postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
    postfix="$postfix noise=$noise"
    postfix="$postfix $extras"

    python -m cmass.survey.hodlightcone survey.geometry=simbig $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
    python -m cmass.survey.hodlightcone survey.geometry=sgc $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
    python -m cmass.survey.hodlightcone survey.geometry=mtng $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
done


echo "~~~ GENERATE NGC RANDOMS ~~~"

nbody=quijote3gpch
sim=randoms
noise_uniform_invoxel=False  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=fixed
diag_from_scratch=True
extras="survey.randoms=True"

L=3000
N=384
hod_seed=0
aug_seed=0

for lhid in {0..8}; do
    postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
    postfix="$postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed"
    postfix="$postfix diag.from_scratch=$diag_from_scratch"
    postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
    postfix="$postfix noise=$noise"
    postfix="$postfix $extras"

    python -m cmass.survey.hodlightcone survey.geometry=ngc $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
done
