#!/bin/bash
#SBATCH --job-name=quijotelike  # Job name
#SBATCH --array=0-199         # Job array range for lhid
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=32            # Number of tasks
#SBATCH --time=12:00:00         # Time limit
#SBATCH --partition=gpu        # Partition name
#SBATCH --gpus-per-node=1       # Number of GPUs per node
#SBATCH --account=phy240043-gpu   # Account name
#SBATCH --output=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/anvil/scratch/x-mho1/jobout/%x_%A_%a.out   # Error file for each array task

# SLURM_ARRAY_TASK_ID=0
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
baseoffset=0

module restore cmass
conda activate cmass
lhid=$((SLURM_ARRAY_TASK_ID + baseoffset))

# Command to run for each lhid
cd /home/x-mho1/git/ltu-cmass

nbody=quijotelike-fid
sim=fastpm
extras="" # "nbody.matchIC=0 meta.cosmofile=./params/big_sobol_params.txt"
L=1000
N=128

outdir=/anvil/scratch/x-mho1/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"

for offset in $(seq 0 200 1800); do
    loff=$((lhid + offset))
    
    postfix="nbody=$nbody sim=$sim nbody.lhid=$loff $extras"

    file=$outdir/$loff/halos.h5
    if [ -f $file ]; then
        echo "File $file exists."
    else
        echo "File $file does not exist."
        python -m cmass.bias.rho_to_halo $postfix
    fi
done
