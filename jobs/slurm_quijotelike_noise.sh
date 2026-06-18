#!/bin/bash
#SBATCH --job-name=quijote_noise   # Job name
# # SBATCH --array=0-99         # Job array range for lhid
#SBATCH --array=1000,1001,1006,101,1010,1017,103,1055,1057,1061,1080,1103,1109,1137,1139,1141,1150,1158,1162,1168,1169,1170,1174,1175,1196,1197,1202,1203,1205,121,1217,1226,1236,1248,1260,1266,1269,1271,1275,1277,1284,1285,1310,1342,1345,136,1362,1378,1394,140,1409,1426,1427,1433,1441,1448,1450,1451,1464,1479,1489,1493,15,1503,1504,151,1519,1521,1539,1557,1573,1582,1587,1592,1599,1610,1617,162,1622,1624,1644,1651,1665,1686,1695,1701,174,1762,1786,179,1797,180,1807,1811,1813,182,1824,1828,1831,184,1847,1848,1850,1855,1861,1868,1885,1903,1917,1935,1944,196,1965,1968,1969,197,1976,1986,1996,1999,202,221,232,245,250,26,266,28,291,298,31,318,327,329,337,340,349,351,354,358,37,373,376,382,388,39,414,416,43,430,443,444,456,459,472,483,484,489,514,545,549,550,551,570,58,587,596,612,619,634,641,645,653,671,69,708,733,747,750,76,778,781,789,807,81,812,833,838,841,844,847,851,860,862,872,873,886,899,914,922,923,929,943,956,973,978,982,995
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=8            # Number of tasks
#SBATCH --time=04:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task


# set -e

# SLURM_ARRAY_TASK_ID=1000

source ~/.bashrc
conda activate cmass
lhid=$SLURM_ARRAY_TASK_ID


# Command to run for each lhid
cd /u/maho3/git/ltu-cmass

Nhod=1
Nnoise=49

nbody=quijote
sim=nbody_hodz_gridnoise_rebin
noise_uniform_invoxel=False  # whether to uniformly distribute galaxies in each voxel (for CHARM only)
noise=reciprocal

multisnapshot=False
diag_from_scratch=True
rm_galaxies=True
extras="bias=zhenginterp_biased bias.hod.custom_prior=ngc nbody.zf=0.5" # meta.cosmofile=./params/abacus_custom_cosmologies.txt bias.hod.use_conc=False" # "noise.params.radial=0 noise.params.transverse=0
L=1000
N=128

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"

outdir=/work/hdd/bdne/maho3/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"


for offset in 0; do # $(seq 0 100 120); do
    lhid=$(($SLURM_ARRAY_TASK_ID+offset))

    postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid"
    postfix="$postfix multisnapshot=$multisnapshot diag.from_scratch=$diag_from_scratch"
    postfix="$postfix bias.hod.noise_uniform=$noise_uniform_invoxel"
    postfix="$postfix noise=$noise"
    postfix="$postfix $extras"

    if [ ! -d "$outdir/$lhid" ]; then
        echo "Directory $outdir/$lhid does not exist. Skipping lhid=$lhid"
        continue
    fi

    # # halos
    # diag_file=$outdir/$lhid/diag/halos.h5
    # if [ -f "$diag_file" ]; then
    #     echo "Diag file $diag_file exists."
    # else
    #     echo "Diag file $diag_file does not exist."
    #     python -m cmass.diagnostics.summ $postfix diag.halo=True
    # fi

    hod_seed=1
    python -m cmass.bias.apply_hod $postfix bias.hod.seed=$hod_seed

    for noise_seed in $(seq 0 $(($Nnoise))); do
        printf -v hod_str "%05d" $hod_seed
        printf -v noise_str "%05d" $noise_seed

        # # halos
        # python -m cmass.diagnostics.summ $postfix diag.halo=True diag.noise_seed=$noise_seed

        # galaxies
        python -m cmass.diagnostics.summ $postfix diag.galaxy=True bias.hod.seed=$hod_seed diag.noise_seed=$noise_seed

        # # set aug_seed the same as hod_seed for simplicity
        # aug_seed=$hod_seed
        # printf -v aug_str "%05d" $aug_seed

        # # simbig_lightcone
        # diag_file=$outdir/$lhid/diag/ngc_lightcone/hod${hod_str}_aug${aug_str}.h5
        # if [ -f "$diag_file" ]; then
        #     echo "Diag file $diag_file exists."
        # else
        #     echo "Diag file $diag_file does not exist."
        #     python -m cmass.survey.hodlightcone survey.geometry=ngc $postfix bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed
        #     python -m cmass.diagnostics.summ diag.ngc=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
        # fi
    done

    # Trash collection
    if [ "$rm_galaxies" = "True" ]; then
        echo "Removing galaxy and lightcone directories for lhid=$lhid"
        rm -rf "$outdir/$lhid/galaxies"
        rm -rf "$outdir/$lhid/simbig_lightcone"
    fi
done
