#!/bin/bash
#SBATCH --job-name=mtnglike_bias   # Job name
#SBATCH --array=0-199         # Job array range for lhid
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=8            # Number of tasks
#SBATCH --time=12:00:00         # Time limit
#SBATCH --partition=shared      # Partition name
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=2

module restore cmass
conda activate cmassrun
lhid=$SLURM_ARRAY_TASK_ID


# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run

Nhod=10

nbody=mtnglike
sim=fastpm_recnoise_rot
noise_uniform_invoxel=True  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=reciprocal

multisnapshot=True
diag_from_scratch=False
rm_galaxies=True
extras="diag.high_res=false bias=zhenginterp_biased" # meta.cosmofile=./params/big_sobol_params.txt" # "nbody.zf=0.500015"
L=3000
N=384

outdir=/anvil/scratch/x-mho1/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"


for offset in $(seq 0 200 2999); do
    lhid=$(($SLURM_ARRAY_TASK_ID+offset))

    postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
    postfix="$postfix multisnapshot=$multisnapshot diag.from_scratch=$diag_from_scratch"
    postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
    postfix="$postfix noise=$noise"
    postfix="$postfix $extras"


    for hod_seed in $(seq 1 $(($Nhod))); do
        printf -v hod_str "%05d" $hod_seed

        # set aug_seed the same as hod_seed for simplicity
        aug_seed=$hod_seed
        printf -v aug_str "%05d" $aug_seed

        # simbig_lightcone
        python -m cmass.survey.hodlightcone survey.geometry=simbig $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
        python -m cmass.diagnostics.summ diag.simbig=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix

        # sgc_lightcone
        python -m cmass.survey.hodlightcone survey.geometry=sgc $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
        python -m cmass.diagnostics.summ diag.sgc=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 

        # # mtng_lightcone
        python -m cmass.survey.hodlightcone survey.geometry=mtng $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
        python -m cmass.diagnostics.summ diag.mtng=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix
    done

    # Trash collection
    if [ "$rm_galaxies" = "True" ]; then
        echo "Removing galaxy and lightcone directories for lhid=$lhid"
        rm -rf "$outdir/$lhid/galaxies"
        rm -rf "$outdir/$lhid/simbig_lightcone"
        rm -rf "$outdir/$lhid/sgc_lightcone"
        rm -rf "$outdir/$lhid/mtng_lightcone"
    fi
done
