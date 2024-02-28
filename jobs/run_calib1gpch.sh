#PBS -N calib1gpch
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=00:30:00
#PBS -t 1401-2000
# for pmwd PBS -l nodes=1:has1gpu:ppn=32,mem=128gb
# for borg PBS -l nodes=1:ppn=32,mem=64gb
#PBS -l nodes=1:ppn=8,mem=16gb

source /data80/mattho/anaconda3/bin/activate
cd /home/mattho/git/ltu-cmass


echo "~~~~~ Running lhid=${PBS_ARRAYID} ~~~~~"

## BORG nbody
# N=128
# conda activate cmass-env
# cd ./quijote_wn
# sh gen_quijote_ic.sh ${N} ${PBS_ARRAYID}
# cd ..
# python -m cmass.nbody.borglpt nbody=quijote nbody.lhid=${PBS_ARRAYID}
# rm ./data/quijote/wn/N${N}/wn_${PBS_ARRAYID}.dat

## PMWD nbody
# N=384
# conda activate cmass
# cd ./quijote_wn
# sh gen_quijote_ic.sh ${N} ${PBS_ARRAYID}
# cd ..
# python -m cmass.nbody.pmwd nbody=quijote nbody.lhid=${PBS_ARRAYID}
# rm ./data/quijote/wn/N${N}/wn_${PBS_ARRAYID}.dat

## BORG fit_halo_bias
conda activate cmass
python -m cmass.bias.fit_halo_bias sim=pmwd nbody=quijote nbody.lhid=${PBS_ARRAYID}
python -m cmass.bias.rho_to_halo sim=pmwd nbody=quijote nbody.lhid=${PBS_ARRAYID}