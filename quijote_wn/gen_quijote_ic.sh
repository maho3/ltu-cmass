
N=128
datadir=/home/mattho/git/ltu-cmass/data/quijote/wn

for i in {0..2000}
do
    echo "Running ${i}"
    ./NGenicWhiteNoise/ngenic_white_noise ${N} ${N} ${i} ${datadir}/wn-N${N}/wn_${i}.dat 64
done
