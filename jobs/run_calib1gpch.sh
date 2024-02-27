#PBS -N calib1gpch
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=04:00:00
#PBS -t 901-1000
#PBS -l nodes=1:ppn=32,mem=64gb
# for pmwd PBS -l nodes=1:has1gpu:ppn=32,mem=128gb

source /data80/mattho/anaconda3/bin/activate
conda activate cmass-env
cd /home/mattho/git/ltu-cmass


echo "~~~~~ Running lhid=${PBS_ARRAYID} ~~~~~"

# BORG
N=128
cd ./quijote_wn
sh gen_quijote_ic.sh ${N} ${PBS_ARRAYID}
cd ..
python -m cmass.nbody.borglpt nbody=quijote nbody.lhid=${PBS_ARRAYID}
rm ./data/quijote/wn/N${N}/wn_${PBS_ARRAYID}.dat


# # PMWD
# N=384
# cd ./quijote_wn
# sh gen_quijote_ic.sh ${N} ${PBS_ARRAYID}
# cd ..
# python -m cmass.nbody.pmwd nbody=quijote nbody.lhid=${PBS_ARRAYID}
# rm ./data/quijote/wn/N${N}/wn_${PBS_ARRAYID}.dat
