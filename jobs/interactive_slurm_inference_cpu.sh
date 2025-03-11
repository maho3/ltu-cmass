#!/bin/bash

module restore cmass_env
conda activate cmass

# exp_index=null
exp_index=0
net_index=0

# Command to run for each lhid
cd /home/x-dbartlett/ltu-cmass

nbody=mtnglike
sim=fastpm
infer=default

halo=False
galaxy=False
ngc=True
sgc=False
mtng=False
Nmax=2000

extras="nbody.zf=0.500015"
device=cpu

suffix="nbody=$nbody sim=$sim infer=$infer infer.exp_index=$exp_index infer.net_index=$net_index"
suffix="$suffix infer.halo=$halo infer.galaxy=$galaxy"
suffix="$suffix infer.ngc_lightcone=$ngc infer.sgc_lightcone=$sgc infer.mtng_lightcone=$mtng infer.Nmax=$Nmax"
suffix="$suffix infer.device=$device $extras"

echo "Running inference with $suffix"
# python -m cmass.infer.preprocess $suffix
python -m cmass.infer.train $suffix net=tuning
#python -m cmass.infer.validate $suffix
