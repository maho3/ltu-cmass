#PBS -N charm_z0.5
#PBS -q batch
#PBS -j oe
#PBS -o ${HOME}/data/jobout/${PBS_JOBNAME}.${PBS_JOBID}.log
#PBS -l walltime=4:00:00
#PBS -t 0-3
#PBS -l nodes=1:ppn=128,mem=512gb
# for pmwd PBS -l nodes=1:has1gpu:ppn=32,mem=128gb
# for fit_halo_bias PBS -l nodes=1:has1gpu:ppn=32,mem=128gb
# for borg PBS -l nodes=1:ppn=32,mem=64gb

source /data80/mattho/anaconda3/bin/activate
cd /home/mattho/git/ltu-cmass

sim=borgpm
# suite=charm_z0.5
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
#     #     
#     # fi

    # Generate white noise
    # N=384
    # cd ./quijote_wn
    # sh gen_quijote_ic.sh ${N} ${idx}
    # cd ..

    # Run borgpm
    module restore myborg
    conda activate borg310
    python -m cmass.nbody.borgpm nbody=test nbody.lhid=${idx}
    # rm ./data/quijote/wn/N${N}/wn_${idx}.dat


    # # Run the rest of things
    # module restore cmass
    # conda activate cmass

    # # python -m cmass.bias.rho_to_halo sim=${sim} nbody=quijote nbody.lhid=${idx}
    # # python -m cmass.survey.remap_cuboid sim=${sim} nbody=quijote nbody.lhid=${idx} nbody.suite=${suite}

    # for j in {0..1}
    # do
    #     python -m cmass.bias.apply_hod sim=${sim} nbody=quijote nbody.lhid=${idx} nbody.suite=${suite} bias.hod.seed=${j}
    #     python -m cmass.survey.ngc_selection sim=${sim} nbody=quijote nbody.lhid=${idx} nbody.suite=${suite} bias.hod.seed=${j}
    #     python -m cmass.summaries.Pk sim=${sim} nbody=quijote nbody.lhid=${idx} nbody.suite=${suite} bias.hod.seed=${j}
    # done
    
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
