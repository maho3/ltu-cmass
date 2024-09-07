#!/bin/sh -l
# FILENAME:  anvil_gen_quijote_ic

#SBATCH -A phy240043
#SBATCH -p shared # the default queue is "shared" queue
#SBATCH --nodes=1
#SBATCH --ntasks=32 
#SBATCH --time=1:00:00
#SBATCH --job-name gen_quijote_ic
#SBATCH --output=/anvil/scratch/x-dbartlett/cmass/quijote/wn/gen_ic.out
#SBATCH --error=/anvil/scratch/x-dbartlett/cmass/quijote/wn/gen_ic.out

module purge
module restore cmass_env
module list
conda activate cmass

set -e

# Print the hostname of the compute node on which this job is running.
hostname

# Change to correct directory
cd /home/x-dbartlett/ltu-cmass/
pwd

# Define the range for lhid
start_lhid=0
end_lhid=1999

# Resolution required
N=512

# Directories
datadir=/anvil/scratch/x-dbartlett/cmass/quijote/wn/N${N}
basedir=/home/x-dbartlett/ltu-cmass/quijote_wn

# Make datadir if it doesn't exist
mkdir -p $datadir

# Loop over lhid from start_lhid to end_lhid
for lhid in $(seq $start_lhid $end_lhid)
do
  echo "Running NGenicWhiteNoise for N=${N} and lhid=${lhid}"
  ${basedir}/NGenicWhiteNoise/ngenic_white_noise ${N} 512 ${lhid} ${datadir}/wn_${lhid}.dat 32
done

exit 0
