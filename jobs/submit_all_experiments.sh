#!/bin/bash

cd /home/x-dbartlett/ltu-cmass/jobs
for exp_index in {0..39}; do
    for Nmax in 500 1000 1500 2000 2500 3000; do
        echo "Submitting inference job for exp_index=$exp_index Nmax=$Nmax"
        sbatch --export=exp_index=$exp_index,Nmax=$Nmax --job-name="inf$exp_index" slurm_inference_cpu.sh
    done
done