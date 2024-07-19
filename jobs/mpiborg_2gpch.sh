
#!/bin/sh
#PBS -S /bin/sh
#PBS -N mpiborg
#PBS -l nodes=2:ppn=64,mem=128gb
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -m a
#PBS -t 1001
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log


module restore myborg
source /data80/mattho/anaconda3/bin/activate
conda activate borg310run

# This works for 2gpc/h runs at B=2, ss=3
NODES=2  # number of physical nodes (same as in nodes hereinabove)
PPN=2  # (number of MPI tasks per node)
CORES=16  # (number of cores indicated in ppn hereinabove)
# Do not touch here
THREADS=$(($CORES / $PPN))
TASKS=$(($NODES * $PPN))

# Change directory to adequate workdir
cd /home/mattho/data/cmass-ili/rundir_cmass
# lhid=$(($PBS_ARRAYID))
lhid=1001
mpirun -genv OMP_NUM_THREADS=$THREADS -genv BORG_TBB_NUM_THREADS=$THREADS -f $PBS_NODEFILE -np $TASKS python -m cmass.nbody.borgpm nbody=2gpch_0704 nbody.lhid=$lhid
