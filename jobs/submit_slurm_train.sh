
#!/bin/bash

cd /home/x-mho1/git/ltu-cmass-run
for exp_index in {0..4}; do
    echo "Submitting inference job for exp_index=$exp_index"
    sbatch --export=exp_index=$exp_index --job-name="inf$exp_index" jobs/slurm_inftrain.sh
    sleep 0.1
done