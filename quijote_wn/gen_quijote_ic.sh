
#!/bin/bash

N=$1
lhid=$2
datadir=/home/mattho/git/ltu-cmass/data/quijote/wn/N${N}

# Make datadir if it doesn't exist
mkdir -p $datadir

echo "Running NGenicWhiteNoise for N=${N} and lhid=${lhid}"
./NGenicWhiteNoise/ngenic_white_noise ${N} ${N} ${lhid} ${datadir}/wn_${lhid}.dat 64
