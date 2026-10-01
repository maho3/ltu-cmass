#!/bin/bash
#SBATCH --job-name=reparam_val  # Job name
#SBATCH --array=0-20            # one per split experiment: mixk 0-20, k03 0-2
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=16             # Number of tasks
#SBATCH --time=12:00:00         # Time limit
#SBATCH --partition=cpu         # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

# Validate abacuslike/fastpm_charm7_cosmoHOD_reparam, self-consistent or on the
# Abacus N-body test suite (preprocessed by jobs/slurm_reparam_infpre.sh).
#   sbatch --export=ALL,mode=self,infer=mixk jobs/slurm_reparam_infvalid.sh
#   sbatch --export=ALL,mode=ood,infer=mixk jobs/slurm_reparam_infvalid.sh
#   sbatch --array=0-2 --export=ALL,mode=ood,infer=k03 jobs/slurm_reparam_infvalid.sh
# Outputs: <exp_path>/posterior_samples.npy (self) or
#          <exp_path>/testing/abacus_nbody_comp_gridnoise_reparam/ (ood)
# clean_models=False keeps the full ensemble for the PPC draws.

source ~/.bashrc
conda activate cmass

exp_index=$SLURM_ARRAY_TASK_ID
sleep $exp_index  # to stagger the start of each job

cd /u/maho3/git/ltu-cmass

mode=${mode:?set mode=self|ood}
infer=${infer:-mixk}

extras="nbody.zf=0.5 infer.embedding_net=fun net=niall2 infer.clean_models=False"
extras="$extras infer.reparam_degeneracy=True hydra/job_logging=disabled"

export TQDM_DISABLE=0

suffix="nbody=abacuslike sim=fastpm_charm7_cosmoHOD_reparam infer=$infer"
suffix="$suffix infer.exp_index=$exp_index infer.net_index=null"
suffix="$suffix infer.tracer=galaxy infer.device=cpu $extras"
suffix="$suffix infer.include_noise=True infer.include_hod=True"
if [ "$mode" = "ood" ]; then
    suffix="$suffix infer.testing.suite=abacus infer.testing.sim=nbody_comp_gridnoise_reparam"
fi

echo "Running validation with $suffix"
python -m cmass.infer.validate $suffix
