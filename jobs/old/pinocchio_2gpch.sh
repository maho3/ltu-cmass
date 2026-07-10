#!/bin/sh -l
# FILENAME:  pinocchio_2gpch

#SBATCH -A phy240043
#SBATCH -p wholenode # the default queue is "shared" queue
#SBATCH --nodes=2
#SBATCH --ntasks=256 
#SBATCH --time=0:30:00
#SBATCH --job-name pinocchio_2gpch
#SBATCH --output=/anvil/scratch/x-dbartlett/cmass/abacuslike/pinocchio/pinocchio_log_%j.out
#SBATCH --error=/anvil/scratch/x-dbartlett/cmass/abacuslike/pinocchio/pinocchio_log_%j.err

module purge 
module restore cmass_env
module list 
conda activate cmass

# Print the hostname of the compute node on which this job is running.
hostname

# Change to correct directory
cd /home/x-dbartlett/ltu-cmass
pwd

python -m cmass.nbody.pinocchio nbody=pinocchio_2gpch nbody.lhid=3 

exit 0
