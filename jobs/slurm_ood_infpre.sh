#!/bin/bash
#SBATCH --job-name=ood_preprocess  # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16            # Number of tasks
#SBATCH --time=2:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%j.out  # Output file
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%j.out   # Error file

# Preprocess OOD test summaries (all -> test split) into
#   $wdir/{quijote3gpch,mtng}/nbody/models/<tracer>/...
# for use as infer.testing.{suite,sim} in slurm_ood_infvalid.sh.
# include_hod/include_noise must match the trained mtnglike/fastpm_charm7 theta dims.

source ~/.bashrc
conda activate cmass

cd /u/maho3/git/ltu-cmass

export TQDM_DISABLE=0

# target=quijote | mtng | all
target=${target:-all}

# nbody tracer infer cosmofile
cases=()
if [ "$target" = "quijote" ] || [ "$target" = "all" ]; then
    cases+=("quijote3gpch galaxy mixk ./params/latin_hypercube_params.txt")
fi
if [ "$target" = "mtng" ] || [ "$target" = "all" ]; then
    cases+=("mtng mtng_lightcone mixk_survey ./params/mtng_cosmologies.txt")
fi

for c in "${cases[@]}"; do
    read -r nbody tracer infer cosmofile <<< "$c"

    suffix="nbody=$nbody sim=nbody infer=$infer infer.exp_index=null"
    suffix="$suffix infer.tracer=$tracer infer.device=cpu"
    suffix="$suffix nbody.zf=0.5 infer.Nmax=4000 meta.cosmofile=$cosmofile"
    suffix="$suffix infer.val_frac=0 infer.test_frac=1"
    suffix="$suffix infer.include_noise=True infer.include_hod=True"
    suffix="$suffix hydra/job_logging=disabled"

    echo "Running preprocessing with $suffix"
    python -m cmass.infer.preprocess $suffix
done
