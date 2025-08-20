
#!/bin/bash

cd /home/x-mho1/git/ltu-cmass-run

for case in simbig sgc mtng; do # ngc
    name="${case}inf"
    # Set all variables to False, then set the current case to True
    export simbig=False sgc=False ngc=False mtng=False
    export $case=True
    for exp_index in 0 1; do
        echo "Submitting inference job for case=$case, exp_index=$exp_index"
        sbatch --export=exp_index=$exp_index,simbig=$simbig,sgc=$sgc,ngc=$ngc,mtng=$mtng --job-name="${name}$exp_index" jobs/slurm_infoptuna.sh
        sleep 0.1
    done
done
