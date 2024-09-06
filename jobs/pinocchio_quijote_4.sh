#!/bin/sh -l
# FILENAME:  pinocchio_quijote_4

#SBATCH -A phy240043
#SBATCH -p shared # the default queue is "shared" queue
#SBATCH --nodes=1
#SBATCH --ntasks=32 
#SBATCH --time=0:10:00
#SBATCH --job-name pinocchio_quijote_4
#SBATCH --output=/anvil/scratch/x-dbartlett/cmass/quijotelike/pinocchio/pinocchio_log_%j.out
#SBATCH --error=/anvil/scratch/x-dbartlett/cmass/quijotelike/pinocchio/pinocchio_log_%j.err

module purge 
module restore cmass_env
module list 
conda activate cmass

# Print the hostname of the compute node on which this job is running.
hostname

# Change to correct directory
cd /home/x-dbartlett/ltu-cmass
pwd

#lhid_list=(12 42 102 140 169 546 784 885 992 998 1050 1132 1218 1546 1707)
lhid_list=(140)

# Loop over lhid from start_lhid to end_lhid
for lhid in "${lhid_list[@]}"
do
  python -m cmass.nbody.pinocchio nbody=pinocchio_quijote nbody.lhid=$lhid
done

exit 0
