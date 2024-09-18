#!/bin/bash
#SBATCH --job-name=mtnglike_whole   # Job name
#SBATCH --array=766-999         # Job array range for lhid
#SBATCH --time=06:00:00         # Time limit
#SBATCH --account=phy240043   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task
#SBATCH --nodes=7               # Number of nodes
#SBATCH --ntasks=896            # Number of tasks
#SBATCH --partition=wholenode      # Partition name

module restore cmass
module load openmpi
source /anvil/projects/x-phy240043/x-mho1/anaconda3/bin/activate
conda activate cmassrun
lhid=$SLURM_ARRAY_TASK_ID
# lhid=432
# lhid=766

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass-run
# outdir=/anvil/scratch/x-mho1/cmass-ili/mtnglike/fastpm/L3000-N384

# 0-1000
python -m cmass.nbody.fastpm nbody=mtnglike nbody.lhid=$lhid
# sbatch --export=lhid=$lhid jobs/slurm_mtng_postprocess.sh

# 1000-2000
lhid=$((lhid+1000))
python -m cmass.nbody.fastpm nbody=mtnglike nbody.lhid=$lhid
# sbatch --export=lhid=$lhid jobs/slurm_mtng_postprocess.sh

