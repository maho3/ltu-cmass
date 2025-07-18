#!/bin/bash
set -e
cd /jet/home/mho1/git/ltu-cmass


seriesname="REC"

# --- Fixed, global variables ---
Nhod=20
multisnapshot=False
diag_from_scratch=True
rm_galaxies=True
noise=reciprocal
TQDM_DISABLE=0

# Define a base set of extras common to all jobs
common_extras="bias=zhenginterp_biased diag.focus_z=0.5 hydra/job_logging=disabled"


# --- Quijotelike ---
nbody=quijotelike
sim=fastpm_recnoise
L=1000
N=128
noise_uniform_invoxel=True
job_extras="$common_extras" # This job has no special extras

sbatch --job-name="${seriesname}_quibias" \
       --export=Nhod=$Nhod,multisnapshot=$multisnapshot,diag_from_scratch=$diag_from_scratch,rm_galaxies=$rm_galaxies,noise="$noise",extras="$job_extras",TQDM_DISABLE=$TQDM_DISABLE,nbody=$nbody,sim="$sim",L=$L,N=$N,noise_uniform_invoxel=$noise_uniform_invoxel \
       ./jobs/slurm_quijotelike_bias.sh


# --- Abacuslike ---
nbody=abacuslike
sim=fastpm_recnoise
L=2000
N=256
noise_uniform_invoxel=True
job_extras="$common_extras nbody.zf=0.500015" # Add specific extra for this job

sbatch --job-name="${seriesname}_ababias" \
       --export=Nhod=$Nhod,multisnapshot=$multisnapshot,diag_from_scratch=$diag_from_scratch,rm_galaxies=$rm_galaxies,noise="$noise",extras="$job_extras",TQDM_DISABLE=$TQDM_DISABLE,nbody=$nbody,sim="$sim",L=$L,N=$N,noise_uniform_invoxel=$noise_uniform_invoxel \
       ./jobs/slurm_abacus_bias.sh


# --- MTNGlike ---
nbody=mtnglike
sim=fastpm_recnoise
L=3000
N=384
noise_uniform_invoxel=True
job_extras="$common_extras nbody.zf=0.500015" # Add specific extra for this job

sbatch --job-name="${seriesname}_mtngbias" \
       --export=Nhod=$Nhod,multisnapshot=$multisnapshot,diag_from_scratch=$diag_from_scratch,rm_galaxies=$rm_galaxies,noise="$noise",extras="$job_extras",TQDM_DISABLE=$TQDM_DISABLE,nbody=$nbody,sim="$sim",L=$L,N=$N,noise_uniform_invoxel=$noise_uniform_invoxel \
       ./jobs/slurm_mtng_bias.sh
