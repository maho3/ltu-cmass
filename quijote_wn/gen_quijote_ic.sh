
#!/bin/bash

N=$1
lhid=$2
# prefix=/home/mattho/git/ltu-cmass
prefix=/u/maho3/git/ltu-cmass
datadir=$prefix/data/quijote/wn/N${N}
basedir=$prefix/quijote_wn

module load fftw/3.3.10-gcc13.3.1 gsl/2.8-gcc13.3.1
export LD_LIBRARY_PATH=$FFTW_HOME/lib:$GSL_HOME/lib:$LD_LIBRARY_PATH

# Make datadir if it doesn't exist
mkdir -p $datadir

echo "Running NGenicWhiteNoise for N=${N} and lhid=${lhid}"
${basedir}/NGenicWhiteNoise/ngenic_white_noise ${N} 512 ${lhid} ${datadir}/wn_${lhid}.dat 64
