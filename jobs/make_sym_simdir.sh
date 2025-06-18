#!/bin/bash

set -e

BASE_DIR="/ocean/projects/phy240015p/mho1/cmass-ili"
SUITE="quijotelike"
SIM1="fastpm"
SIM2="fastpm_varnoise"
SUBDIR="L1000-N128"
N=1999

cd "$BASE_DIR/$SUITE/$SIM2/$SUBDIR"

for i in $(seq 0 "$N"); do
  (
    mkdir -p "$i" && cd "$i"
    ln -s "../../../$SIM1/$SUBDIR/$i/config.yaml" .
    ln -s "../../../$SIM1/$SUBDIR/$i/halos.h5" .
    echo "$i"
  )
done
