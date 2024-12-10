#!/bin/sh

# Cancel job if error raised
set -e

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

cd ..

# Run Pinocchio
python -m cmass.infer.train nbody=quijotelike sim=fastpm infer=quijotelike

conda deactivate
exit 0

