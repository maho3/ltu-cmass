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
python -m cmass.nbody.pinocchio nbody=pin_1gpch_z0.5_id3_N512

conda deactivate
exit 0

