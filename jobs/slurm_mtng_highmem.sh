#!/bin/bash
#SBATCH --job-name=mtnglike_highm   # Job name
#SBATCH --array=1         # Job array range for lhid
#SBATCH --time=00:60:00         # Time limit
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=128            # Number of tasks
#SBATCH --partition=highmem      # Partition name

module restore cmass
module load openmpi
source /anvil/projects/x-phy240043/x-mho1/anaconda3/bin/activate
conda activate cmassrun
lhid=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run
outdir=/anvil/scratch/x-mho1/cmass-ili/mtnglike/fastpm/L3000-N384

# 0-1000
# check if nbody.h5 exists
file=$outdir/$lhid/nbody.h5
if [ -f $file ]; then
    echo "File $file exists."
else
    echo "File $file does not exist."
    python -m cmass.nbody.fastpm nbody=mtnglike nbody.lhid=$lhid
fi

# # 1000-2000
# lhid=$((lhid+1000))
# file=$outdir/$lhid/nbody.h5
# if [ -f $file ]; then
#     echo "File $file exists."
# else
#     echo "File $file does not exist."
#     python -m cmass.nbody.fastpm nbody=mtnglike nbody.lhid=$lhid
# fi
