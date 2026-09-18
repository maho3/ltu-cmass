#!/bin/bash
#SBATCH --job-name=ood_validate  # Job name
#SBATCH --array=0-20  # one per split experiment in mixk / mixk_survey
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16            # Number of tasks
#SBATCH --time=12:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# Test mtnglike/fastpm_charm7 models on OOD points. Submit once per tracer:
#   sbatch --export=ALL,tracer=galaxy jobs/slurm_ood_infvalid.sh          # on quijote3gpch/nbody
#   sbatch --export=ALL,tracer=mtng_lightcone jobs/slurm_ood_infvalid.sh  # on mtng/nbody
# Outputs: <model exp_path>/testing/<suite>_<sim>/

# SLURM_ARRAY_TASK_ID=0

source ~/.bashrc
conda activate cmass

exp_index=$SLURM_ARRAY_TASK_ID
net_index=null

sleep $exp_index  # to stagger the start of each job

cd /u/maho3/git/ltu-cmass

nbody=mtnglike
sim=fastpm_charm7
tracer=${tracer:-galaxy}

if [ "$tracer" = "galaxy" ]; then
    infer=mixk
    test_suite=quijote3gpch
elif [ "$tracer" = "mtng_lightcone" ]; then
    infer=mixk_survey
    test_suite=mtng
else
    echo "Unknown tracer=$tracer"; exit 1
fi
test_sim=nbody

# clean_models=False: don't prune the trained ensemble from an OOD run
extras="nbody.zf=0.5 infer.embedding_net=fun net=niall2 infer.clean_models=False"
device="cpu"

export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.tracer=$tracer"
suffix="$suffix infer.device=$device $extras"
suffix="$suffix infer.include_noise=True infer.include_hod=True"
suffix="$suffix infer.testing.suite=$test_suite infer.testing.sim=$test_sim"

echo "Running inference pipeline with $suffix"

python -m cmass.infer.validate $suffix
