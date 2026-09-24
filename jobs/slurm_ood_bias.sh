#!/bin/bash
#SBATCH --job-name=ood_bias   # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=32            # Number of tasks
#SBATCH --mem=64GB            # Memory per node
#SBATCH --time=08:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

# OOD test summaries for mtnglike/fastpm_charm7 models. Submit per target:
#   sbatch --export=ALL,target=quijote jobs/slurm_ood_bias.sh
#       quijote3gpch/nbody/2000 -> galaxy box summaries, HOD seeds as in slurm_mtng_bias.sh,
#                                  noise grid rows applied at the diag stage
#   sbatch --export=ALL,target=mtng jobs/slurm_ood_bias.sh
#       mtng/nbody/0            -> real MTNG lightcone (hod00000), noised per grid row
# Overrides: hod_seeds="1 2 3 4 5", noise_rows="8 16 24" (or "all")
# Run scripts/diagnose_ood_testdata.py first.

source ~/.bashrc
conda activate cmass

module load gsl
export CPATH=$GSL_ROOT_DIR/include:$CPATH
export LIBRARY_PATH=$GSL_ROOT_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/u/maho3/anaconda3/envs/cmass/lib:$GSL_ROOT_DIR/lib:$LD_LIBRARY_PATH

cd /u/maho3/git/ltu-cmass

target=${target:-quijote}
hod_seeds=${hod_seeds:-"1 2 3 4 5"}
noise_rows=${noise_rows:-"8 16 24"}  # noisegrid diagonal: sigma_r = sigma_t = 0.75, 1.50, 2.26 Mpc/h

wdir=/work/hdd/bdne/maho3/cmass-ili
noise_file=$wdir/noise_priors/noisegrid.csv
if [ "$noise_rows" = "all" ]; then
    noise_rows=$(seq 0 $(($(wc -l < $noise_file) - 1)))
fi

diag_from_scratch=True
rm_galaxies=True
L=3000
N=384

export TQDM_DISABLE=0
extras="hydra/job_logging=disabled diag.high_res=True diag.noise_file=$noise_file"

if [ "$target" = "quijote" ]; then
    nbody=quijote3gpch
    sim=nbody
    lhid=2000
    outdir=$wdir/$nbody/$sim/L$L-N$N

    for hod_seed in $hod_seeds; do
        printf -v hod_str "%05d" $hod_seed

        postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid nbody.zf=0.5 multisnapshot=False"
        postfix="$postfix meta.cosmofile=./params/latin_hypercube_params.txt"
        postfix="$postfix bias=zheng_composite bias.hod.noise_uniform=False bias.hod.seed=$hod_seed"
        postfix="$postfix noise=reciprocal diag.from_scratch=$diag_from_scratch $extras"

        galfile=$outdir/$lhid/galaxies/hod$hod_str.h5
        if [ ! -f "$galfile" ]; then
            python -m cmass.bias.apply_hod $postfix
        fi

        for noise_seed in $noise_rows; do
            printf -v noise_str "%06d" $noise_seed
            diag_file=$outdir/$lhid/diag/galaxies/hod${hod_str}_noise${noise_str}.h5
            if h5ls "$diag_file" 2>/dev/null | grep -q '^0.666667'; then
                echo "skipping $diag_file"
                continue
            fi
            python -m cmass.diagnostics.summ $postfix diag.galaxy=True diag.noise_seed=$noise_seed
        done

        if [ "$rm_galaxies" = "True" ]; then
            rm -f "$galfile"
        fi
    done
elif [ "$target" = "mtng" ]; then
    nbody=mtng
    sim=nbody
    lhid=0
    src=$wdir/$nbody/$sim/L$L-N$N/$lhid

    # stale 2025 summary (old binning, no noise attrs) would be picked up by preprocess
    stale=$src/diag/mtng_lightcone/hod00000_aug00000.h5
    if [ -f "$stale" ]; then
        mkdir -p $src/diag_stale/mtng_lightcone
        mv -n "$stale" $src/diag_stale/mtng_lightcone/
    fi

    python scripts/noise_mtng_lightcone.py --wdir $wdir --lhid $lhid \
        --noise-file $noise_file --rows "$(echo $noise_rows | tr ' ' ',')" --write-bias-cfg

    postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid nbody.zf=0.5 multisnapshot=False"
    postfix="$postfix meta.cosmofile=./params/mtng_cosmologies.txt"
    postfix="$postfix bias=zhenginterp_biased bias.hod.custom_prior=mtng bias.hod.seed=0"
    postfix="$postfix bias.hod.noise_uniform=False noise=reciprocal"
    postfix="$postfix survey.geometry=mtng diag.mtng=True diag.from_scratch=$diag_from_scratch $extras"

    for noise_seed in $noise_rows; do
        aug_seed=$(($noise_seed+1))
        printf -v aug_str "%05d" $aug_seed
        printf -v noise_str "%06d" $noise_seed
        file=$src/diag/mtng_lightcone/hod00000_aug${aug_str}_noise${noise_str}.h5
        keys=$(h5ls "$file" 2>/dev/null)
        if grep -q '^Bk[[:space:]]' <<< "$keys" && grep -q '^Pk[[:space:]]' <<< "$keys"; then
            echo "skipping $file"
            continue
        fi
        python -m cmass.diagnostics.summ $postfix survey.aug_seed=$aug_seed diag.noise_seed=$noise_seed
    done

    if [ "$rm_galaxies" = "True" ]; then
        for noise_seed in $noise_rows; do
            printf -v aug_str "%05d" $(($noise_seed+1))
            rm -f $src/mtng_lightcone/hod00000_aug${aug_str}.h5  # never aug00000
        done
    fi
else
    echo "Unknown target=$target"; exit 1
fi
