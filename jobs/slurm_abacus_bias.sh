#!/bin/bash
#SBATCH --job-name=abacuslike_bias   # Job name
#SBATCH --array=0-118         # Job array range for lhid
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=5            # Number of tasks
#SBATCH --gpus=v100-32:1     # Number of GPUs
#SBATCH --time=48:00:00         # Time limit
#SBATCH --partition=GPU-shared      # Partition name
#SBATCH --account=phy240015p   # Account name
#SBATCH --output=/ocean/projects/phy240015p/mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/ocean/projects/phy240015p/mho1/jobout/%x_%A_%a.out   # Error file for each array task

SLURM_ARRAY_TASK_ID=13

module restore cmass
conda activate cmass
lhid=$SLURM_ARRAY_TASK_ID


# Command to run for each lhid
cd /jet/home/mho1/git/ltu-cmass

Nhod=1
Naug=1

nbody=abacuslike
sim=fastpm
multisnapshot=True
diag_from_scratch=True
rm_galaxies=False
extras="bias=zheng_biased diag.high_res=True diag.focus_z=0.5" #  meta.cosmofile=./params/abacus_custom_cosmologies.txt" #  bias=zheng_biased 
L=2000
N=256

# export TQDM_DISABLE=0
# extras="$extras hydra/job_logging=disabled"

outdir=/ocean/projects/phy240015p/mho1/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"


for offset in 0; do # $(seq 0 100 2999); do
    lhid=$(($SLURM_ARRAY_TASK_ID+offset))

    postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid multisnapshot=$multisnapshot diag.from_scratch=$diag_from_scratch $extras"

    # density
    # python -m cmass.diagnostics.summ diag.density=True $postfix 

    # halos
    file=$outdir/$lhid/halos.h5
    if [ -f $file ]; then
        echo "File $file exists."
    else
        echo "File $file does not exist."
        # python -m cmass.bias.rho_to_halo $postfix
    fi
    # python -m cmass.diagnostics.summ diag.halo=True $postfix 

    # galaxies
    for i in $(seq 0 $(($Nhod-1))); do
        hod_seed=$((lhid*10+i+1))
        printf -v hod_str "%05d" $hod_seed
        file=$outdir/$lhid/galaxies/hod$hod_str.h5
        if [ -f $file ]; then
            echo "File $file exists."
        else
            echo "File $file does not exist."
            python -m cmass.bias.apply_hod $postfix bias.hod.seed=$hod_seed
        fi
        python -m cmass.diagnostics.summ $postfix diag.galaxy=True bias.hod.seed=$hod_seed

        # # ngc_lightcone
        # for aug_seed in $(seq 0 $(($Naug-1))); do
        #     printf -v aug_str "%05d" $aug_seed
        #     # lightcone
        #     file=$outdir/$lhid/ngc_lightcone/hod${hod_str}_aug${aug_str}.h5
        #     if [ -f $file ]; then
        #         echo "File $file exists."
        #     else
        #         echo "File $file does not exist."
        #         python -m cmass.survey.lightcone $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
        #     fi
        #     python -m cmass.diagnostics.summ diag.ngc=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
        # done

        # # mtng_lightcone
        # for aug_seed in $(seq 0 $(($Naug-1))); do
        #     printf -v aug_str "%05d" $aug_seed
        #     # lightcone
        #     file=$outdir/$lhid/mtng_lightcone/hod${hod_str}_aug${aug_str}.h5
        #     if [ -f $file ]; then
        #         echo "File $file exists."
        #     else
        #         echo "File $file does not exist."
        #         python -m cmass.survey.mtng_selection $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
        #     fi
        #     python -m cmass.diagnostics.summ diag.mtng=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
        # done

        # Trash collection
        if [ $rm_galaxies = True ]; then
            # galaxies
            echo "Removing galaxies for lhid=$lhid hod_seed=$hod_seed"
            rm $outdir/$lhid/galaxies/hod$hod_str.h5

            # ngc_lightcone
            echo "Removing lightcone for lhid=$lhid hod_seed=$hod_seed"
            rm $outdir/$lhid/ngc_lightcone/hod${hod_str}_aug*.h5

            # lightcone
            echo "Removing lightcone for lhid=$lhid hod_seed=$hod_seed"
            rm $outdir/$lhid/mtng_lightcone/hod${hod_str}_aug*.h5
        fi
    done
done