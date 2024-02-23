#PBS -N pmwd_quijote
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/.jupyter_log/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=04:00:00
#PBS -l nodes=1:has1gpu:ppn=32,mem=128gb
#PBS -t 0-512

source /data80/mattho/anaconda3/bin/activate
conda activate cmass
cd /home/mattho/git/ltu-cmass


echo "~~~~~ Running lhid=${PBS_ARRAYID} ~~~~~"
# cd ./quijote_wn
# sh gen_quijote_ic.sh 256 ${PBS_ARRAYID}
# cd ..
python -m cmass.nbody.pmwd nbody.lhid=${PBS_ARRAYID}
# rm ./data/quijote/wn/N256/wn_${i}.dat
