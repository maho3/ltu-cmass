
#!/bin/bash

N=$1
lhid=$2
prefix=/home/mattho/git/ltu-cmass
# prefix=/home/x-mho1/git/ltu-cmass
prefix=//anvil/scratch/x-dbartlett/cmass
datadir=$prefix/data/quijote/wn/N${N}
basedir=$prefix/quijote_wn
basedir=/home/x-dbartlett/ltu-cmass/quijote_wn

# Make datadir if it doesn't exist
mkdir -p $datadir

echo "Running NGenicWhiteNoise for N=${N} and lhid=${lhid}"
echo "Output ${datadir}/wn_${lhid}.dat"
# ${basedir}/NGenicWhiteNoise/ngenic_white_noise ${N} 512 ${lhid} ${datadir}/wn_${lhid}.dat 1
