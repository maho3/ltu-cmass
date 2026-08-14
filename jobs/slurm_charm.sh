#!/bin/bash
#SBATCH --job-name=mtngcharm  # Job name
#SBATCH --array=0-19         # Job array range for lhid
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=8            # Number of tasks
#SBATCH --time=4:00:00         # Time limit
#SBATCH --partition=ghx4        # Partition name
#SBATCH --gpus-per-node=1       # Number of GPUs per node
#SBATCH --account=bdne-dtai-gh   # Account name
#SBATCH --output=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out  # Output file for each array task
#SBATCH --error=/work/hdd/bdne/maho3/jobout/%x_%A_%a.out   # Error file for each array task

SLURM_ARRAY_TASK_ID=10
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
# baseoffset=0

# module restore cmass
source ~/.bashrc
module load cray-hdf5/1.14.3.7
conda activate cmass

lhid=$((SLURM_ARRAY_TASK_ID + baseoffset))
echo "lhid=$lhid"

# Command to run for each lhid
cd /u/maho3/git/ltu-cmass

nbody=mtnglike
sim=fastpm_charm7
multisnapshot=True
extras="nbody.matchIC=0" # "meta.cosmofile=./params/mtng_cosmologies.txt" # meta.cosmofile=./params/abacus_cosmologies.txt" # nbody.zf=0.500015"
L=3000
N=384
keys_to_check=(0.586220 0.606330 0.626440 0.646550 0.666660 0.686770 0.706880)
# keys_to_check=(0.666667)

outdir=/work/hdd/bdne/maho3/cmass-ili/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"


export TQDM_DISABLE=0
extras="$extras hydra/job_logging=disabled"


# Loop through offsets and process files
for offset in $(seq 0 20 3999); do  # 4799
    loff=$((lhid + offset))
    postfix="nbody=$nbody sim=$sim nbody.lhid=$loff multisnapshot=$multisnapshot $extras"
    file=$outdir/$loff/halos.h5
    nbody_file=$outdir/$loff/nbody.h5

    # Check if directory exists
    if [ ! -d "$outdir/$loff" ]; then
        echo "Directory $outdir/$loff does not exist. Continuing..."
        continue
    fi

    # Check if nbody.h5 (required input) exists
    if [ ! -f $nbody_file ]; then
        echo "Input $nbody_file does not exist. Skipping..."
        continue
    fi

    # Check if file exists
    if [ -f $file ]; then
        echo "File $file exists."
        all_keys_exist=true

        # Check if all keys exist in the file
        for key in $keys_to_check; do
            if ! h5ls $file | grep -q $key; then
                all_keys_exist=false
                echo "Key $key does not exist in $file."
                break
            fi
        done

        # Process file based on key existence
        if $all_keys_exist; then
            echo "All keys exist in $file. Skipping..."
        else
            echo "Not all keys exist in $file. Rerunning..."
            python -m cmass.bias.rho_to_halo $postfix
        fi
    else
        echo "File $file does not exist. Running..."
        python -m cmass.bias.rho_to_halo $postfix
    fi
done
