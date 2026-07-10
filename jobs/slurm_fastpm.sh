#!/bin/bash
#SBATCH --job-name=abacusmore  # Job name
#SBATCH --array=0-19         # Job array range for lhid
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=128            # Number of tasks
#SBATCH --mem=240G            # Amount of memory
#SBATCH --time=3:00:00         # Time limit
#SBATCH --partition=cpu  # Partition name
#SBATCH --account=bdne-delta-cpu  # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task


# SLURM_ARRAY_TASK_ID=1
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
baseoffset=2000

module load cray-mpich/8.1.32 gsl
export LD_LIBRARY_PATH=/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/gsl-2.8-zty4u3k/lib:$LD_LIBRARY_PATH

source ~/.bashrc
conda activate cmass

lhid=$((SLURM_ARRAY_TASK_ID + baseoffset))

# Command to run for each lhid
cd /u/maho3/git/ltu-cmass

nbody=abacuslike
sim=fastpm
multisnapshot=True
extras="nbody.matchIC=0" #  meta.cosmofile=./params/abacus_cosmologies.txt"
L=2000
N=256

outdir=/work/hdd/bdne/maho3/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"

# export TQDM_DISABLE=0
# extras="$extras hydra/job_logging=disabled"


for offset in $(seq 0 20 200); do
    loff=$((lhid + offset))
    
    postfix="nbody=$nbody sim=$sim nbody.lhid=$loff multisnapshot=$multisnapshot $extras"

    file=$outdir/$loff/nbody.h5
    if [ -f $file ]; then
        echo "File $file exists."
    else
        echo "File $file does not exist."
        python -m cmass.nbody.fastpm $postfix
        # python -m cmass.nbody.postprocess_fastpm $postfix
    fi
done
