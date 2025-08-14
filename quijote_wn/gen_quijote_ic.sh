
#!/bin/bash

N=$1
lhid=$2
# prefix=/home/mattho/git/ltu-cmass
prefix=/home/x-mho1/git/ltu-cmass
datadir=$prefix/data/quijote/wn/N${N}
basedir=$prefix/quijote_wn

# Make datadir if it doesn't exist
mkdir -p $datadir

echo "Running NGenicWhiteNoise for N=${N} and lhid=${lhid}"
${basedir}/NGenicWhiteNoise/ngenic_white_noise ${N} 512 ${lhid} ${datadir}/wn_${lhid}.dat 64
