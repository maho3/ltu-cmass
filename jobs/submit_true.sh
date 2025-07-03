#!/bin/bash
set -e
cd /jet/home/mho1/git/ltu-cmass


seriesname="NO"

# --- Fixed, global variables ---
Nhod=1
Naug=1
multisnapshot=False
diag_from_scratch=True
rm_galaxies=True
noise=fixed
TQDM_DISABLE=0

# Define a base set of extras common to all jobs
common_extras="diag.use_ngp=True bias=zhenginterp_biased diag.focus_z=0.5 hydra/job_logging=disabled"


# # --- Quijote ---
# nbody=quijote
# sim=nbody_nonoise_ngp
# L=1000
# N=128
# noise_uniform_invoxel=False
# job_extras="$common_extras" # This job has no special extras

# sbatch --job-name="${seriesname}_quibias" \
#        --export=Nhod=$Nhod,Naug=$Naug,multisnapshot=$multisnapshot,diag_from_scratch=$diag_from_scratch,rm_galaxies=$rm_galaxies,noise="$noise",extras="$job_extras",TQDM_DISABLE=$TQDM_DISABLE,nbody=$nbody,sim="$sim",L=$L,N=$N,noise_uniform_invoxel=$noise_uniform_invoxel \
#        ./jobs/slurm_quijotelike_bias.sh


# --- Abacus ---
nbody=abacus
sim=custom_nonoise_ngp
L=2000
N=256
noise_uniform_invoxel=False
job_extras="$common_extras nbody.zf=0.5 meta.cosmofile=./params/abacus_custom_cosmologies.txt"

sbatch --job-name="${seriesname}_ababias" \
       --export=Nhod=$Nhod,Naug=$Naug,multisnapshot=$multisnapshot,diag_from_scratch=$diag_from_scratch,rm_galaxies=$rm_galaxies,noise="$noise",extras="$job_extras",TQDM_DISABLE=$TQDM_DISABLE,nbody=$nbody,sim="$sim",L=$L,N=$N,noise_uniform_invoxel=$noise_uniform_invoxel \
       ./jobs/slurm_abacus_bias.sh


# # --- Quijote 3 Gpc/h ---
# nbody=quijote3gpch
# sim=nbody_nonoise
# L=3000
# N=384
# noise_uniform_invoxel=False
# job_extras="$common_extras nbody.zf=0.500015"

# sbatch --job-name="${seriesname}_mtngbias" \
#        --export=Nhod=$Nhod,Naug=$Naug,multisnapshot=$multisnapshot,diag_from_scratch=$diag_from_scratch,rm_galaxies=$rm_galaxies,noise="$noise",extras="$job_extras",TQDM_DISABLE=$TQDM_DISABLE,nbody=$nbody,sim="$sim",L=$L,N=$N,noise_uniform_invoxel=$noise_uniform_invoxel \
#        ./jobs/slurm_mtng_bias.sh
