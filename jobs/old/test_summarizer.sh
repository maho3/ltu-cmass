#!/bin/sh -l
# FILENAME: test_summarizer

#SBATCH -A phy240043
#SBATCH -p shared # the default queue is "shared" queue
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0:20:00
#SBATCH --job-name test_summarizer
#SBATCH --output=/anvil/scratch/x-dbartlett/cmass/test_summarizer_log_%j.out
#SBATCH --error=/anvil/scratch/x-dbartlett/cmass/test_summarizer_log_%j.err

# Cancel job if error raised
set -e

# Modules
module purge
#module load gmp/6.2.1
#module load mpfr/4.0.2
#module load mpc/1.1.0
#module load zlib/1.2.11
#module load gcc/11.2.0
#module load xalt/2.10.45
#module load modtree/cpu
#module load gsl/2.4
#module load mvapich2/2.3.6
#module load fftw/3.3.8
#module load anaconda
module restore cmass_env

# Environment
conda deactivate
#conda activate cmass_summ
conda activate cmass

cd /home/x-dbartlett/ltu-cmass 

# Run summarizer
export LD_PRELOAD=/usr/lib64/libslurm.so
#python -m cmass.bias.apply_hod nbody=pinocchio_quijote sim=pinocchio
python -m cmass.diagnostics.summ nbody=pinocchio_quijote sim=pinocchio

conda deactivate
exit 0

