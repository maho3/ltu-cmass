#PBS -N aba_bias_survey
#PBS -q batch
#PBS -l walltime=4:00:00
#PBS -l nodes=1:ppn=32,mem=64gb
#PBS -t 0-199
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

nbody=2gpch_0704
sim=borgpm

# # # FOR INDIVIDUAL RUNS
# lhid=$(($PBS_ARRAYID))
# # lhid=131
# hod_seed=0
# python -m cmass.bias.rho_to_halo nbody=$nbody nbody.lhid=$lhid sim=$sim meta.cosmofile=./params/abacus_cosmologies.txt
# python -m cmass.bias.apply_hod nbody=$nbody nbody.lhid=$lhid sim=$sim bias.hod.seed=$hod_seed meta.cosmofile=./params/abacus_cosmologies.txt
# # python -m cmass.survey.ngc_selection nbody=$nbody nbody.lhid=$lhid sim=$sim
# python -m cmass.diagnostics.summ nbody=$nbody nbody.lhid=$lhid sim=$sim bias.hod.seed=$hod_seed meta.cosmofile=./params/abacus_cosmologies.txt

# FOR LOOPING AUGMENTATION 
for i in $(seq 0 200 800); do
    lhid=$(($i + $PBS_ARRAYID))
    # python -m cmass.bias.rho_to_halo nbody=$nbody nbody.lhid=$lhid sim=$sim
    # python -m cmass.bias.apply_hod nbody=$nbody nbody.lhid=$lhid sim=$sim
    # python -m cmass.survey.ngc_selection nbody=$nbody nbody.lhid=$lhid sim=$sim
    python -m cmass.diagnostics.summ nbody=$nbody nbody.lhid=$lhid sim=$sim
done

# # FOR LOOPING AUGMENTATION WITH HOD
# # PBS_ARRAYID=3
# for i in $(seq 0 200 1800); do
#     lhid=$(($i + $PBS_ARRAYID))
#     # python -m cmass.bias.rho_to_halo nbody=$nbody nbody.lhid=$lhid sim=$sim
#     for j in {0..0}; do
#         hod_seed=$(($i/10 + $j))
#         python -m cmass.bias.apply_hod nbody=$nbody nbody.lhid=$lhid sim=$sim bias.hod.seed=$hod_seed
#         # python -m cmass.survey.ngc_selection nbody=$nbody nbody.lhid=$lhid sim=$sim bias.hod.seed=$hod_seed
#         python -m cmass.diagnostics.summ nbody=$nbody nbody.lhid=$lhid sim=$sim bias.hod.seed=$hod_seed
#     done
# done


echo done