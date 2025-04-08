#!/bin/bash

module restore cmass_env
conda activate cmass

lhid=0

# Command to run for each lhid
cd /home/x-dbartlett/ltu-cmass

Nhod=5
Naug=1

nbody=mtng_test
sim=fastpm
rm_galaxies=True
extras="bias=zdep bias.hod.model=zheng07 bias.hod.default_params=reid2014_cmass bias.hod.assem_bias=True bias.hod.vel_assem_bias=True diag.summaries=['Pk']"
L=3000
N=384

outdir=/anvil/scratch/x-dbartlett/cmass/$nbody/$sim/L$L-N$N
echo "outdir=$outdir"

postfix="nbody=$nbody sim=$sim nbody.lhid=$lhid $extras"


# galaxies
for i in $(seq 0 $(($Nhod-1))); do
    hod_seed=$((lhid*10+i+0))
    printf -v hod_str "%05d" $hod_seed
    echo "hod_str=$hod_str"

    file1=$outdir/$lhid/ngc_lightcone/hod${hod_str}_aug00000.h5
    file2=$outdir/$lhid/mtng_lightcone/hod${hod_str}_aug00000.h5
    if [ -f $file1 ] && [ -f $file2 ]; then
        echo "File mtng and ngc exist."
        continue
    else
        echo "File mtng or ngc does not exist."
    fi

    # mtng_lightcone
    for aug_seed in $(seq 0 $(($Naug-1))); do
        printf -v aug_str "%05d" $aug_seed
        # lightcone
        file=$outdir/$lhid/mtng_lightcone/hod${hod_str}_aug${aug_str}.h5
        if [ -f $file ]; then
            echo "File $file exists."
        else
            echo "File $file does not exist."
            python -m cmass.survey.hodlightcone $postfix bias.hod.seed=$hod_seed
        fi
        python -m cmass.diagnostics.summ diag.ngc=True bias.hod.seed=$hod_seed survey.aug_seed=$aug_seed $postfix 
    done
done