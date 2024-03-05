#PBS -N calib1gpch
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=01:00:00
#PBS -t 5-499
#PBS -l nodes=1:has1gpu:ppn=32,mem=128gb
# for rho_to_halo PBS -l nodes=1:ppn=32,mem=64gb
# for borg PBS -l nodes=1:ppn=32,mem=64gb

source /data80/mattho/anaconda3/bin/activate
cd /home/mattho/git/ltu-cmass



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

# fit_halo_bias
sim=pmwd
conda activate cmass

for i in 0 500 1000 1500
do
    idx=$(($i + ${PBS_ARRAYID}))
    echo "~~~~~ Running lhid=${idx} ~~~~~"
    python -m cmass.bias.fit_halo_bias sim=${sim} nbody=quijote nbody.lhid=${idx}
    echo "~~~~~ Finished lhid=${idx} ~~~~~"
done

# conda activate cmass
# source_path=/home/mattho/git/ltu-cmass/data/calib_1gpch_z0.5/${sim}/L1000-N128/${PBS_ARRAYID}
# if [ -f ${source_path}/halo_bias.npy ]
# then 
#     echo "halo_bias.npy exists, skipping fit_halo_bias"
# else
#     python -m cmass.bias.fit_halo_bias sim=${sim} nbody=quijote nbody.lhid=${PBS_ARRAYID}
# fi

# if [ -f ${source_path}/halo_pos.npy ]
# then 
#     echo "halo_pos.npy exists, skipping rho_to_halo"
# else
#     python -m cmass.bias.rho_to_halo sim=${sim} nbody=quijote nbody.lhid=${PBS_ARRAYID}
# fi
