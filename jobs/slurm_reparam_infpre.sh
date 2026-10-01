#!/bin/bash
#SBATCH --job-name=reparam_pre  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16             # Number of tasks
#SBATCH --mem=128G              # Amount of memory
#SBATCH --time=6:00:00          # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

# Preprocessing for the fastpm_charm7_cosmoHOD_reparam validation / PPC round.
#   target=train_check  mixk on the training suite into $WDIR/scratch/reparam_repre,
#                       to diff against the trained arrays before installing
#                       the new hodprior.csv / reparam_bounds.yaml (no retrain)
#   target=train_k03    k03 (kmax=0.3) on the training suite, into the live tree
#   target=abacus       mixk + k03 on the Abacus test suite, all -> test split,
#                       into abacus/nbody_comp_gridnoise_reparam
#   sbatch --export=ALL,target=abacus jobs/slurm_reparam_infpre.sh

source ~/.bashrc
conda activate cmass

cd /u/maho3/git/ltu-cmass

export TQDM_DISABLE=0

WDIR=/work/hdd/bdne/maho3/cmass-ili
target=${target:?set target=train_check|train_k03|abacus}

common="infer.tracer=galaxy infer.device=cpu nbody.zf=0.5 infer.Nmax=4000"
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
    train_check)
        run $train infer=mixk infer.save_dir=$WDIR/scratch/reparam_repre $common ;;
    train_k03)
        run $train infer=k03 $common ;;
    abacus)
        run $abacus infer=mixk $common
        run $abacus infer=k03 $common ;;
    *)
        echo "Unknown target=$target"; exit 1 ;;
esac
