
#!/bin/bash

cd /u/maho3/git/ltu-cmass

for case in galaxy ; do # simbig sgc mtng ngc
    name="abaOms8"
    # Set all variables to False, then set the current case to True
    export galaxy=False simbig=False sgc=False ngc=False mtng=False
    export $case=True
    for exp_index in 0 1 2 3; do
        echo "Submitting inference job for case=$case, exp_index=$exp_index"
        # sbatch --export=exp_index=$exp_index,galaxy=$galaxy,simbig=$simbig,sgc=$sgc,ngc=$ngc,mtng=$mtng --job-name="${name}$exp_index" jobs/slurm_infoptuna.sh
        sbatch --export=exp_index=$exp_index,galaxy=$galaxy,simbig=$simbig,sgc=$sgc,ngc=$ngc,mtng=$mtng --job-name="${name}$exp_index" jobs/slurm_infretrain.sh
        sleep 0.1
    done
done
