
#!/bin/bash

cd /home/x-mho1/git/ltu-cmass
for exp_index in {0..14}; do
    echo "Submitting inference job for exp_index=$exp_index"
    sbatch --export=exp_index=$exp_index --job-name="inf$exp_index" jobs/slurm_inference_cpu.sh
done