#!/bin/bash
#SBATCH --job-name=bkcuts_pre  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16            # Number of tasks
#SBATCH --mem=128G              # Amount of memory
#SBATCH --time=2:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

# Preprocessing for the 2026-11 mixed P(k)/B(k) k-cut PPC round (infer=bkcuts).
#   target=train    on the training suite abacuslike/fastpm_charm7_cosmoHOD_reparam
#   target=abacus   on the Abacus OOD test suite, all -> test split
#   sbatch --export=ALL,target=train jobs/slurm_bkcuts_infpre.sh
# Also fine interactively (srun -p cpu-interactive -n 16 --mem=64G) when the queue is busy.
#
# No Optuna for these cuts. After the train preprocess, copy the study db from
# the nearest existing cut (exp_index -> source):
#   0 zPk024 k0.25            <- zPk0+zPk2+zPk4/kmin-0.0_kmax-0.2
#   1 zPk024+zBk0 {0.2,0.08}  <- zPk0+zPk2+zPk4+zBk0/kmin-0.0_kmax-zBk=0.2__zPk=0.2
#   2 zPk024+zBk0 {0.25,0.2}  <- same {0.2,0.2}
#   3 zPk024+zBk0 {0.3,0.1}   <- zPk0+zPk2+zPk4+zBk0/kmin-0.0_kmax-zBk=0.2__zPk=0.4
#   4 zPk024+zBk0 {0.3,0.12}  <- same {0.4,0.2}

source ~/.bashrc
conda activate cmass

cd /u/maho3/git/ltu-cmass

export TQDM_DISABLE=0

target=${target:?set target=train|abacus}

common="infer=bkcuts infer.tracer=galaxy infer.device=cpu nbody.zf=0.5 infer.Nmax=4000"
common="$common infer.include_noise=True infer.include_hod=True"
common="$common infer.reparam_degeneracy=True hydra/job_logging=disabled"
train="nbody=abacuslike sim=fastpm_charm7_cosmoHOD_reparam"
abacus="nbody=abacus sim=nbody_comp_gridnoise_reparam infer.val_frac=0 infer.test_frac=1"

run() {
    echo "Running preprocessing with $*"
    python -m cmass.infer.preprocess "$@"
    echo "preprocess status=$?"
}

case $target in
    train)  run $train $common ;;
    abacus) run $abacus $common ;;
    *) echo "Unknown target=$target"; exit 1 ;;
esac
