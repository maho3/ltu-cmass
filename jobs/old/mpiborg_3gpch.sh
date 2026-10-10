
#!/bin/sh
#PBS -S /bin/sh
#PBS -N cmass3gpch
#PBS -l nodes=7:ppn=16,mem=256gb
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -m a
#PBS -t 0
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log


module restore myborg
source /data80/mattho/anaconda3/bin/activate
conda activate borg310run

# This (IS ESTIMATED) to work for 3gpc/h runs at B=1, ss=3
NODES=7  # number of physical nodes (same as in nodes hereinabove)
PPN=2  # (number of MPI tasks per node)
CORES=14  # (number of cores indicated in ppn hereinabove)
# Do not touch here
THREADS=$(($CORES / $PPN))
TASKS=$(($NODES * $PPN))

# Change directory to adequate workdir
cd /home/mattho/data/cmass-ili/rundir_cmass

# Run things
lhid=$(($PBS_ARRAYID))
nbody=3gpch_nolc_B1
# lhid=1001
mpirun -genv OMP_NUM_THREADS=$THREADS -genv BORG_TBB_NUM_THREADS=$THREADS -f $PBS_NODEFILE -np $TASKS python -m cmass.nbody.borgpm nbody=$nbody nbody.lhid=$lhid
