#!/bin/bash

set -e

SIM1="/ocean/projects/phy240015p/mho1/cmass-ili/quijotelike/fastpm"
SIM2="/ocean/projects/phy240015p/mho1/cmass-ili/quijotelike/fastpm_expnoise"
SUBDIR="L1000-N128"
N=1999

mkdir -p "$SIM2/$SUBDIR"
cd "$SIM2/$SUBDIR"

for i in $(seq 0 "$N"); do
  (
    mkdir -p "$i" && cd "$i"
    ln -s "$SIM1/$SUBDIR/$i/config.yaml" .
    ln -s "$SIM1/$SUBDIR/$i/halos.h5" .
    echo "$i"
  )
done
