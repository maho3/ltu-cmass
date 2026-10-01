#!/bin/bash
#SBATCH --job-name=ppc_draw     # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16             # Number of tasks
#SBATCH --mem=64G               # Amount of memory
#SBATCH --time=2:00:00          # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

# Stage 0 for the 2026-09-30 OOD PPCs: 50 draws per campaign at Abacus lhid 38,
# noise (0, 0), against the reparam Abacus test preprocessing.

source ~/.bashrc
conda activate cmass
cd /u/maho3/git/ltu-cmass
export OMP_NUM_THREADS=16

M=/work/hdd/bdne/maho3/cmass-ili/abacuslike/fastpm_charm7_cosmoHOD_reparam/models/galaxy
for c in "zPk0+zPk2+zPk4+zBk0/kmin-0.0_kmax-zBk=0.2__zPk=0.2 zPk024zBk0k02" \
         "zPk0/kmin-0.0_kmax-0.2 zPk0k02" \
         "zPk0+zPk2+zPk4/kmin-0.0_kmax-0.2 zPk024k02" \
         "zPk0+zPk2+zPk4/kmin-0.0_kmax-0.4 zPk024k04"; do
    read -r exp tag <<< "$c"
    echo "=== $exp ($tag)"
    PYTHONPATH=. python -u ppc/draw.py --exp_path "$M/$exp" --ndraw 50 \
        --obs_lhid 38 --obs_noiseid 0 \
        --testing_suite abacus --testing_sim nbody_comp_gridnoise_reparam \
        --tag obs00038n000_$tag
    echo "draw status=$?"
done
