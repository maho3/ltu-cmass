#!/bin/bash
#SBATCH --job-name=charm
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --account=bdne-dtai-gh
#SBATCH --output=/work/hdd/bdne/maho3/jobout/charm_%j.out
#SBATCH --error=/work/hdd/bdne/maho3/jobout/charm_%j.out

# Called by watch_and_submit.sh with --export=lhid=<N>
# e.g. sbatch --export=lhid=42 slurm_charm_single.sh
lhid=2000
echo "lhid=$lhid"

source ~/.bashrc
conda activate cmass

cd /u/maho3/git/ltu-cmass

nbody=quijote3gpch
sim=fastpm_charm6
multisnapshot=True
extras="nbody.matchIC=0" #  hydra/job_logging=disabled"
L=3000
N=384

outdir=/work/hdd/bdne/maho3/cmass-ili/$nbody/$sim/L$L-N$N
file=$outdir/$lhid/halos.h5

if [ -f "$file" ]; then
    echo "halos.h5 already exists for lhid=$lhid. Skipping."
    exit 0
fi

if [ ! -d "$outdir/$lhid" ]; then
    echo "Directory $outdir/$lhid does not exist (nbody.h5 not staged yet?). Exiting."
    exit 1
fi

echo "Running rho_to_halo for lhid=$lhid ..."
python -m cmass.bias.rho_to_halo \
    nbody=$nbody sim=$sim nbody.lhid=$lhid \
    multisnapshot=$multisnapshot $extras
