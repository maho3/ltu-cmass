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

# SLURM_ARRAY_TASK_ID=663

module restore cmass
conda activate cmassrun
lhid=$SLURM_ARRAY_TASK_ID


# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run

Nhod=1

nbody=mtnglike
sim=fastpm_unconstrained
noise_uniform_invoxel=True  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=reciprocal
use_custom_prior=False

multisnapshot=True
diag_from_scratch=False
rm_galaxies=True
extras="diag=nzonly diag.high_res=false bias=zhenginterp_biased" # meta.cosmofile=./params/big_sobol_params.txt" # "nbody.zf=0.500015"
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

        # Set custom_prior argument if flag is True
        if [ "$use_custom_prior" = "True" ]; then
            simbig_prior="bias.hod.custom_prior=simbig"
            sgc_prior="bias.hod.custom_prior=sgc"
            mtng_prior="bias.hod.custom_prior=mtng"
            ngc_prior="bias.hod.custom_prior=ngc"
        else
            simbig_prior=""
            sgc_prior=""
            mtng_prior=""
            ngc_prior=""
        fi

        # simbig_lightcone
        file=$outdir/$lhid/diag/simbig_lightcone/hod${hod_str}_aug${aug_str}.h5
        if ! h5ls "$file" | grep -q '^Pk[[:space:]]'; then
            echo "Running $file"
            scriptargs="diag.simbig=True survey.geometry=simbig $simbig_prior bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix"
            datafile=$outdir/$lhid/simbig_lightcone/hod${hod_str}_aug${aug_str}.h5
            if [ ! -f "$datafile" ]; then
                python -m cmass.survey.hodlightcone $scriptargs
            fi
            python -m cmass.diagnostics.summ $scriptargs
        else
            echo "skipping $file"
        fi

        # sgc_lightcone
        file=$outdir/$lhid/diag/sgc_lightcone/hod${hod_str}_aug${aug_str}.h5
        if ! h5ls "$file" | grep -q '^Pk[[:space:]]'; then
            echo "Running $file"
            scriptargs="diag.sgc=True survey.geometry=sgc $sgc_prior bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix"
            datafile=$outdir/$lhid/sgc_lightcone/hod${hod_str}_aug${aug_str}.h5
            if [ ! -f "$datafile" ]; then
                python -m cmass.survey.hodlightcone $scriptargs
            fi
            python -m cmass.diagnostics.summ $scriptargs
        else
            echo "skipping $file"
        fi

        # mtng_lightcone
        file=$outdir/$lhid/diag/mtng_lightcone/hod${hod_str}_aug${aug_str}.h5
        if ! h5ls "$file" | grep -q '^Pk[[:space:]]'; then
            echo "Running $file"
            scriptargs="diag.mtng=True survey.geometry=mtng $mtng_prior bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix"
            datafile=$outdir/$lhid/mtng_lightcone/hod${hod_str}_aug${aug_str}.h5
            if [ ! -f "$datafile" ]; then
                python -m cmass.survey.hodlightcone $scriptargs
            fi
            python -m cmass.diagnostics.summ $scriptargs
        else
            echo "skipping $file"
        fi

        # ngc_lightcone
        file=$outdir/$lhid/diag/ngc_lightcone/hod${hod_str}_aug${aug_str}.h5
        if ! h5ls "$file" | grep -q '^Pk[[:space:]]'; then
            echo "Running $file"
            scriptargs="diag.ngc=True survey.geometry=ngc $ngc_prior bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix"
            datafile=$outdir/$lhid/ngc_lightcone/hod${hod_str}_aug${aug_str}.h5
            if [ ! -f "$datafile" ]; then
                python -m cmass.survey.hodlightcone $scriptargs
            fi
            python -m cmass.diagnostics.summ $scriptargs
        else
            echo "skipping $file"
        fi
    done

    if [ "$rm_galaxies" = "True" ]; then
        echo "Removing galaxy and lightcone directories for lhid=$lhid"
        rm -rf "$outdir/$lhid/galaxies"
        rm -rf "$outdir/$lhid/simbig_lightcone"
        rm -rf "$outdir/$lhid/sgc_lightcone"
        rm -rf "$outdir/$lhid/mtng_lightcone"
        rm -rf "$outdir/$lhid/ngc_lightcone"
    fi
done
