#!/bin/bash
#SBATCH --job-name=abacuslike   # Job name
#SBATCH --array=0-999         # Job array range for lhid
#SBATCH --nodes=2               # Number of nodes
#SBATCH --ntasks=256            # Number of tasks
#SBATCH --time=00:60:00         # Time limit
#SBATCH --partition=wholenode      # Partition name
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

module restore cmass
source /anvil/projects/x-phy240043/x-mho1/anaconda3/bin/activate
conda activate cmassrun
lhid=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run
outdir=/anvil/scratch/x-mho1/cmass-ili/abacuslike/fastpm/L2000-N256

# 0-1000
# check if nbody.h5 exists
file=$outdir/$lhid/nbody.h5
if [ -f $file ]; then
    echo "File $file exists."
else
    echo "File $file does not exist."
    python -m cmass.nbody.fastpm nbody=abacuslike nbody.lhid=$lhid
fi

# 1000-2000
lhid=$((lhid+1000))
file=$outdir/$lhid/nbody.h5
if [ -f $file ]; then
    echo "File $file exists."
else
    echo "File $file does not exist."
    python -m cmass.nbody.fastpm nbody=abacuslike nbody.lhid=$lhid
fi
