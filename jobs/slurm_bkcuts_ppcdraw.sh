#!/bin/bash
#SBATCH --job-name=bkcuts_draw  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16             # Number of tasks
#SBATCH --mem=64G               # Amount of memory
#SBATCH --time=2:00:00          # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

# Stage 0 of the 2026-11 mixed-cut PPC round: 50 draws per campaign at Abacus
# lhid 38, noise row 0 (sigma_rad = sigma_tran = 0), against the reparam Abacus
# test preprocessing. Light enough for the login node:
#   bash jobs/slurm_bkcuts_ppcdraw.sh
# Set cases="..." to run a subset (space-separated tags).

source ~/.bashrc
conda activate cmass
cd /u/maho3/git/ltu-cmass
export OMP_NUM_THREADS=${SLURM_NTASKS:-8}

M=/work/hdd/bdne/maho3/cmass-ili/abacuslike/fastpm_charm7_cosmoHOD_reparam/models/galaxy
all=("zPk0+zPk2+zPk4+zBk0/kmin-0.0_kmax-zBk=0.08__zPk=0.2 zPk024k02zBk0k008"
     "zPk0+zPk2+zPk4+zBk0/kmin-0.0_kmax-zBk=0.2__zPk=0.25 zPk024k025zBk0k02"
     "zPk0+zPk2+zPk4+zBk0/kmin-0.0_kmax-zBk=0.1__zPk=0.3 zPk024k03zBk0k01"
     "zPk0+zPk2+zPk4+zBk0/kmin-0.0_kmax-zBk=0.12__zPk=0.3 zPk024k03zBk0k012"
     "zPk0+zPk2+zPk4/kmin-0.0_kmax-0.25 zPk024k025")
for c in "${all[@]}"; do
    read -r exp tag <<< "$c"
    if [ -n "${cases:-}" ] && [[ " $cases " != *" $tag "* ]]; then continue; fi
    echo "=== $exp ($tag)"
    PYTHONPATH=. python -u ppc/draw.py --exp_path "$M/$exp" --ndraw 50 \
        --obs_lhid 38 --obs_noiseid 0 \
        --testing_suite abacus --testing_sim nbody_comp_gridnoise_reparam \
        --tag obs00038n000_$tag
    echo "draw status=$?"
done
