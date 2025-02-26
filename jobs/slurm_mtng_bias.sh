#!/bin/bash
#SBATCH --job-name=mtnglike_bias0000   # Job name
# #SBATCH --array=0-999         # Job array range for lhid
#SBATCH --array=16,166,400,819,821,933
# #SBATCH --array=201,674,819,831,876
# #SBATCH --array=150,963
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=128            # Number of tasks
#SBATCH --time=03:00:00         # Time limit
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

Nhod=5
Naug=1

nbody=mtnglike
sim=fastpm
multisnapshot=True
diag_from_scratch=False
rm_galaxies=True
extras="nbody.zf=0.500015 hydra/job_logging=disabled"
L=3000
N=384

outdir=/anvil/scratch/x-mho1/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"


for offset in 0; do
    lhid=$(($SLURM_ARRAY_TASK_ID+offset))

    postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid diag.from_scratch=$diag_from_scratch $extras"

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
        echo "hod_str=$hod_str"

        file1=$outdir/$lhid/ngc_lightcone/hod${hod_str}_aug00000.h5
        file2=$outdir/$lhid/mtng_lightcone/hod${hod_str}_aug00000.h5
        if [ -f $file1 ] && [ -f $file2 ]; then
            echo "File mtng and ngc exist."
            continue
        else
            echo "File mtng or ngc does not exist."
        fi

        # galaxies
        file=$outdir/$lhid/galaxies/hod$hod_str.h5
        if [ -f $file ]; then
            echo "File $file exists."
        else
            echo "File $file does not exist."
            python -m cmass.bias.apply_hod $postfix bias.hod.seed=$hod_seed multisnapshot=$multisnapshot 
        fi
        # python -m cmass.diagnostics.summ $postfix diag.galaxy=True bias.hod.seed=$hod_seed

        # ngc_lightcone
        for aug_seed in $(seq 0 $(($Naug-1))); do
            printf -v aug_str "%05d" $aug_seed
            # lightcone
            file=$outdir/$lhid/ngc_lightcone/hod${hod_str}_aug${aug_str}.h5
            if [ -f $file ]; then
                echo "File $file exists."
            else
                echo "File $file does not exist."
                python -m cmass.survey.lightcone $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed multisnapshot=$multisnapshot 
            fi
            python -m cmass.diagnostics.summ diag.ngc=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
        done

        # mtng_lightcone
        for aug_seed in $(seq 0 $(($Naug-1))); do
            printf -v aug_str "%05d" $aug_seed
            # lightcone
            file=$outdir/$lhid/mtng_lightcone/hod${hod_str}_aug${aug_str}.h5
            if [ -f $file ]; then
                echo "File $file exists."
            else
                echo "File $file does not exist."
                python -m cmass.survey.mtng_selection $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed multisnapshot=False 
            fi
            python -m cmass.diagnostics.summ diag.mtng=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
        done

        # Trash collection
        if [ $rm_galaxies = True ]; then
            # galaxies
            echo "Removing galaxies for lhid=$lhid hod_seed=$hod_seed"
            rm $outdir/$lhid/galaxies/hod$hod_str.h5

            # # ngc_lightcone
            # echo "Removing lightcone for lhid=$lhid hod_seed=$hod_seed"
            # rm $outdir/$lhid/ngc_lightcone/hod${hod_str}_aug*.h5

            # # mtng_lightcone
            # echo "Removing lightcone for lhid=$lhid hod_seed=$hod_seed"
            # rm $outdir/$lhid/mtng_lightcone/hod${hod_str}_aug*.h5
        fi
    done
done