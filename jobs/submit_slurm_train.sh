#!/bin/bash

cd /u/maho3/git/ltu-cmass

for case in galaxy; do  # simbig sgc mtng ngc
    name="charm6_rebin"
    # submit_script="jobs/slurm_infoptuna.sh"
    submit_script="jobs/slurm_infretrain.sh"
    prev_job_id=""
    for exp_index in {0..18}; do
        echo "Submitting inference job for case=$case, exp_index=$exp_index"
        dep_flag=""
        if [[ -n "$prev_job_id" ]]; then
            dep_flag="--dependency=after:${prev_job_id}"
        fi
        prev_job_id=$(sbatch $dep_flag \
            --export=exp_index=$exp_index,tracer=$case \
            --job-name="${name}$exp_index" \
            "$submit_script" \
            | awk '{print $NF}')
        echo "  Submitted job $prev_job_id"
        sleep 0.1
    done
done