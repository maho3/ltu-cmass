#!/bin/bash
#SBATCH --job-name=quijotefastpm_bias   # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=4            # Number of tasks
#SBATCH --time=04:00:00         # Time limit
#SBATCH --partition=shared      # Partition name
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task


set -e

module restore cmass
conda activate cmassrun
lhid=$SLURM_ARRAY_TASK_ID


# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run

nbody=quijotelike
sim=fastpm_test
noise_uniform_invoxel=True  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=reciprocal

multisnapshot=False
diag_from_scratch=True
rm_galaxies=True
extras="bias=zheng_biased" # meta.cosmofile=./params/big_sobol_params.txt" # "nbody.zf=0.500015"
L=1000
N=128

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"

outdir=/anvil/scratch/x-mho1/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"

# which simulation to run
lhid=0
hod_seed=1
aug_seed=$hod_seed

# setup command postfix
postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
postfix="$postfix multisnapshot=$multisnapshot diag.from_scratch=$diag_from_scratch"
postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
postfix="$postfix noise=$noise"
postfix="$postfix $extras"

# ~~ N-body ~~~
python -m cmass.nbody.fastpm $postfix

# ~~ Halos ~~
python -m cmass.bias.rho_to_halo $postfix
python -m cmass.diagnostics.summ $postfix diag.halo=True

# ~~ Galaxies ~~
python -m cmass.bias.apply_hod $postfix bias.hod.seed=$hod_seed
python -m cmass.diagnostics.summ $postfix diag.galaxy=True bias.hod.seed=$hod_seed

# ~~ SimBig Lightcone ~~
python -m cmass.survey.hodlightcone survey.geometry=simbig $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
python -m cmass.diagnostics.summ diag.simbig=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix
