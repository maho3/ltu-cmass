#!/bin/bash
#SBATCH --job-name=abacuslike_bias   # Job name
#SBATCH --array=0-99         # Job array range for lhid
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=5            # Number of tasks
#SBATCH --gpus=v100-32:1,h100:1  # Request either one V100-32 or one H100 GPU
#SBATCH --time=24:00:00         # Time limit
#SBATCH --partition=GPU-shared      # Partition name
#SBATCH --account=phy240015p   # Account name
#SBATCH --output=/ocean/projects/phy240015p/mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/ocean/projects/phy240015p/mho1/jobout/%x_%A_%a.out   # Error file for each array task

# set -e

# SLURM_ARRAY_TASK_ID=663

module restore cmass
conda activate cmass
lhid=$SLURM_ARRAY_TASK_ID


# Command to run for each lhid
cd /jet/home/mho1/git/ltu-cmass

# Nhod=5

# nbody=abacuslike
# sim=fastpm_recnoise
# noise_uniform_invoxel=True  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
# noise=reciprocal

# multisnapshot=False
# diag_from_scratch=True
# rm_galaxies=True
# extras="bias=zhenginterp_biased diag.focus_z=0.5 nbody.zf=0.500015" #  meta.cosmofile=./params/abacus_custom_cosmologies.txt" #  bias=zheng_biased 
# L=2000
# N=256

# export TQDM_DISABLE=0
# extras="$extras hydra/job_logging=disabled"

outdir=/ocean/projects/phy240015p/mho1/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"


for offset in $(seq 0 100 1999); do
    lhid=$(($SLURM_ARRAY_TASK_ID+offset))

    postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
    postfix="$postfix multisnapshot=$multisnapshot diag.from_scratch=$diag_from_scratch"
    postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
    postfix="$postfix noise=$noise"
    postfix="$postfix $extras"
    
    for hod_seed in $(seq 1 $(($Nhod))); do
        printf -v hod_str "%05d" $hod_seed

        # # galaxies
        # diag_file=$outdir/$lhid/diag/galaxies/hod$hod_str.h5
        # if [ -f "$diag_file" ]; then
        #     echo "Diag file $diag_file exists."
        # else
        #     echo "Diag file $diag_file does not exist."
        #     python -m cmass.bias.apply_hod $postfix bias.hod.seed=$hod_seed
        #     python -m cmass.diagnostics.summ $postfix diag.galaxy=True bias.hod.seed=$hod_seed
        # fi

        # set aug_seed the same as hod_seed for simplicity
        aug_seed=$hod_seed
        printf -v aug_str "%05d" $aug_seed

        # simbig_lightcone
        diag_file=$outdir/$lhid/diag/simbig_lightcone/hod${hod_str}_aug${aug_str}.h5
        if [ -f "$diag_file" ]; then
            echo "Diag file $diag_file exists."
        else
            echo "Diag file $diag_file does not exist."
            python -m cmass.survey.hodlightcone survey.geometry=simbig $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
            python -m cmass.diagnostics.summ diag.simbig=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
        fi

        # sgc_lightcone
        diag_file=$outdir/$lhid/diag/sgc_lightcone/hod${hod_str}_aug${aug_str}.h5
        if [ -f "$diag_file" ]; then
            echo "Diag file $diag_file exists."
        else
            echo "Diag file $diag_file does not exist."
            python -m cmass.survey.hodlightcone survey.geometry=sgc $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
            python -m cmass.diagnostics.summ diag.sgc=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
        fi

        # mtng_lightcone
        diag_file=$outdir/$lhid/diag/mtng_lightcone/hod${hod_str}_aug${aug_str}.h5
        if [ -f "$diag_file" ]; then
            echo "Diag file $diag_file exists."
        else
            echo "Diag file $diag_file does not exist."
            python -m cmass.survey.hodlightcone survey.geometry=mtng $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
            python -m cmass.diagnostics.summ diag.mtng=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
        fi
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
