
#!/bin/bash

name=fastnn

cd /home/x-mho1/git/ltu-cmass-run
for exp_index in {0..8}; do  # 9
    echo "Submitting inference job for exp_index=$exp_index"
    # sbatch --export=exp_index=$exp_index --job-name="inf$exp_index" jobs/slurm_inftrain.sh
    sbatch --export=exp_index=$exp_index --job-name="${name}inf$exp_index" jobs/slurm_infoptuna.sh
    sleep 0.1
done
