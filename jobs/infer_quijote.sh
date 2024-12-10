#!/bin/sh -l
# FILENAME:  infer_quijote
#SBATCH -A phy240043-gpu
#SBATCH --nodes=1             # Total # of nodes
#SBATCH --ntasks-per-node=1   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gpus-per-node=1     # Number of GPUs per node
#SBATCH --time=0:05:00        # Total run time limit (hh:mm:ss)
#SBATCH -J infer_quijote          # Job name
#SBATCH --output=/anvil/scratch/x-dbartlett/cmass/quijotelike/infer_log_%j.out
#SBATCH --error=/anvil/scratch/x-dbartlett/cmass/quijotelike/infer_log_%j.err
#SBATCH -p gpu                # Queue (partition) name
#SBATCH --mail-user=deaglan.bartlett@physics.ox.ac.uk
#SBATCH --mail-type=all       # Send email to above address at begin and end of job

# Cancel job if error raised
set -e

# Print the hostname of the compute node on which this job is running.
hostname

# Change to correct directory
cd /home/x-dbartlett/ltu-cmass
pwd

# Modules
module purge
module load gcc/11.2.0
module load openmpi/4.1.6
module load gsl/2.4
module load fftw/3.3.8
module load anaconda

# Environment
conda deactivate
conda activate cmass

# Run inference
python -m cmass.infer.train nbody=quijotelike sim=fastpm infer=quijotelike

conda deactivate
exit 0


