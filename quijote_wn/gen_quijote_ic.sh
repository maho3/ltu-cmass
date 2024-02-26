
#!/bin/bash

N=$1
lhid=$2
datadir=/home/mattho/git/ltu-cmass/data/quijote/wn/N${N}
basedir=/home/mattho/git/ltu-cmass/quijote_wn

# Make datadir if it doesn't exist
mkdir -p $datadir

echo "Running NGenicWhiteNoise for N=${N} and lhid=${lhid}"
${basedir}/NGenicWhiteNoise/ngenic_white_noise ${N} ${N} ${lhid} ${datadir}/wn_${lhid}.dat 64
