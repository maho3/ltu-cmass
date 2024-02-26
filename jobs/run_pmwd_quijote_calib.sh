#PBS -N pmwd_quijote
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=04:00:00
#PBS -l nodes=1:has1gpu:ppn=32,mem=128gb
#PBS -t 513-550

source /data80/mattho/anaconda3/bin/activate
conda activate cmass
cd /home/mattho/git/ltu-cmass

N=384

echo "~~~~~ Running lhid=${PBS_ARRAYID} ~~~~~"
cd ./quijote_wn
sh gen_quijote_ic.sh ${N} ${PBS_ARRAYID}
cd ..
python -m cmass.nbody.pmwd nbody.lhid=${PBS_ARRAYID}
rm ./data/quijote/wn/N${N}/wn_${PBS_ARRAYID}.dat
