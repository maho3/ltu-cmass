
#!/bin/sh
#PBS -S /bin/sh
#PBS -N cmass1gpch
#PBS -l nodes=1:ppn=32,mem=64gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -m a
#PBS -t 0-99
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log

module restore cmass
source /data80/mattho/anaconda3/bin/activate
conda activate cmassrun

# Change directory to adequate workdir
cd /home/mattho/data/cmass-ili/rundir_cmass

nbody=quijotelike
sim=fastpm

# # FOR INDIVIDUAL RUNS
# lhid=$(($PBS_ARRAYID))
# python -m cmass.nbody.fastpm nbody=$nbody nbody.lhid=$lhid meta.cosmofile=./params/mtng_cosmologies.txt  # NOTE: MTNG

# FOR LOOPING AUGMENTATION
for i in $(seq 0 100 1900); do
    lhid=$(($i + $PBS_ARRAYID))
    python -m cmass.nbody.fastpm nbody=$nbody nbody.lhid=$lhid
    python -m cmass.bias.rho_to_halo sim=$sim nbody=$nbody nbody.lhid=$lhid
    python -m cmass.diagnostics.summ sim=$sim nbody=$nbody nbody.lhid=$lhid
done
