#!/bin/bash
#SBATCH --job-name=mtnglike_whole   # Job name
#SBATCH --array=109-1000%10        # Job array range for lhid
#SBATCH --time=02:00:00         # Time limit
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task
#SBATCH --nodes=7               # Number of nodes
#SBATCH --ntasks=896            # Number of tasks
#SBATCH --partition=wholenode      # Partition name
#SBATCH --nice=2              # Decrease priority

# SLURM_ARRAY_TASK_ID=102
offset=2000

module restore cmass
conda activate cmassrun
lhid=$((SLURM_ARRAY_TASK_ID + offset))

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run
outdir=/anvil/scratch/x-mho1/cmass-ili/mtnglike/fastpm/L3000-N384

# check if nbody.h5 exists
file=$outdir/$lhid/nbody.h5
if [ -f $file ]; then
    echo "File $file exists."
else
    echo "File $file does not exist."
    python -m cmass.nbody.fastpm nbody=mtnglike nbody.lhid=$lhid +nbody.postprocess=False

    sbatch jobs/slurm_mtng_postprocess.sh --array=$SLURM_ARRAY_TASK_ID
fi

