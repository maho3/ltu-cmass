
#!/bin/sh
#PBS -S /bin/sh
#PBS -N cmass3gpch
#PBS -l nodes=1:ppn=128,mem=512gb
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -m a
#PBS -t 0-99
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log

module restore myborg
source /data80/mattho/anaconda3/bin/activate
conda activate borg310run

# Change directory to adequate workdir
cd /home/mattho/data/cmass-ili/rundir_cmass

nbody=mtnglike

# FOR INDIVIDUAL RUNS
lhid=$(($PBS_ARRAYID))
python -m cmass.nbody.borgpm nbody=$nbody nbody.lhid=$lhid meta.cosmofile=./params/mtng_cosmologies.txt  # NOTE: MTNG

# # FOR LOOPING AUGMENTATION
# # PBS_ARRAYID=3
# for i in $(seq 1000 100 1900); do
#     lhid=$(($i + $PBS_ARRAYID))
#     python -m cmass.nbody.borgpm nbody=$nbody nbody.lhid=$lhid
# done
