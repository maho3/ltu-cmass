#PBS -N bias_survey
#PBS -q batch
#PBS -l walltime=4:00:00
#PBS -l nodes=1:ppn=32,mem=64gb
#PBS -t 0-999
#PBS -j oe
#PBS -m a
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log

echo cd-ing...

cd /home/mattho/data/cmass-ili/rundir_cmass

echo activating environment...
module restore cmass
source /data80/mattho/anaconda3/bin/activate
conda activate cmassrun

echo running script...
echo "arrayind is ${PBS_ARRAYID}"

lhid=$(($PBS_ARRAYID))
python -m cmass.bias.rho_to_halo nbody=test nbody.lhid=$lhid sim=borgpm
python -m cmass.bias.apply_hod nbody=test nbody.lhid=$lhid sim=borgpm
python -m cmass.survey.ngc_selection nbody=test nbody.lhid=$lhid sim=borgpm

echo done