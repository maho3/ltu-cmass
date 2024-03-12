#PBS -N quijote
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=01:00:00
#PBS -t 0-3
#PBS -l nodes=1:ppn=16,mem=32gb
# for pmwd PBS -l nodes=1:has1gpu:ppn=32,mem=128gb
# for fit_halo_bias PBS -l nodes=1:has1gpu:ppn=32,mem=128gb
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
# rm ./data/quijote/wn/N${N}/wn_${PBS_ARRAYID}.dat

# # fit_halo_bias
sim=latin_hypercube_HR
suite=quijote

for i in 0 500 1000 1500
do
    idx=$(($i + ${PBS_ARRAYID}))
    echo "~~~~~ Running lhid=${idx} ~~~~~"

    # module load cuda
    # conda activate cmass
    # python -m cmass.nbody.pmwd nbody=2gpch nbody.lhid=${idx}

#     # source_path=/home/mattho/git/ltu-cmass/data/calib_1gpch_z0.5/${sim}/L1000-N128/${idx}
#     # if [ -f ${source_path}/halo_bias.npy ]
#     # then 
#     #     echo "halo_bias.npy exists, skipping fit_halo_bias"
#     # else
#     #     python -m cmass.bias.fit_halo_bias sim=${sim} nbody=quijote nbody.lhid=${idx}
#     # fi

#     # if [ -f ${source_path}/halo_mass.npy ]
#     # then 
#     #     echo "halo_mass.npy exists, skipping rho_to_halo"
#     # else
#     #     conda activate cmass
#     #     python -m cmass.bias.rho_to_halo sim=${sim} nbody=quijote nbody.lhid=${idx}
#     # fi
    module restore cmass
    conda activate cmass

    python -m cmass.survey.remap_cuboid sim=${sim} nbody=quijote nbody.lhid=${idx} nbody.suite=${suite}

    for j in {0..5}
    do
        python -m cmass.bias.apply_hod sim=${sim} nbody=quijote nbody.lhid=${idx} nbody.suite=${suite} bias.hod.seed=${j}
        python -m cmass.survey.ngc_selection sim=${sim} nbody=quijote nbody.lhid=${idx} nbody.suite=${suite} bias.hod.seed=${j}
    done
    
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
