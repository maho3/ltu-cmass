#!/bin/sh -l
# FILENAME:  pinocchio_quijote_0

#SBATCH -A phy240043
#SBATCH -p shared # the default queue is "shared" queue
#SBATCH --nodes=1
#SBATCH --ntasks=32 
#SBATCH --time=8:00:00
#SBATCH --job-name pinocchio_quijote_1000_1249
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

# Define the range for lhid
start_lhid=1192
end_lhid=1249

# Loop over lhid from start_lhid to end_lhid
for lhid in $(seq $start_lhid $end_lhid)
do
  python -m cmass.nbody.pinocchio nbody=pinocchio_quijote nbody.lhid=$lhid nbody.matchIC=0 
done

exit 0
