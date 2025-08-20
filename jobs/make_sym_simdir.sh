#!/bin/bash

set -e

SIM1="/anvil/scratch/x-mho1/cmass-ili/mtnglike/fastpm"
SIM2="/anvil/scratch/x-mho1/cmass-ili/mtnglike/fastpm_constrained"
SUBDIR="L3000-N384"
N=2999

mkdir -p "$SIM2/$SUBDIR"
cd "$SIM2/$SUBDIR"

echo "Source:      $SIM1/$SUBDIR"
echo "Destination: $SIM2/$SUBDIR"
read -p "Are you sure you want to create symbolic links for $((N+1)) simulations from the above source to destination? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Aborted."
  exit 1
fi

for i in $(seq 0 "$N"); do
  (
    mkdir -p "$i" && cd "$i"
    ln -s "$SIM1/$SUBDIR/$i/config.yaml" .
    ln -s "$SIM1/$SUBDIR/$i/halos.h5" .
    echo "$i"
  )
done
