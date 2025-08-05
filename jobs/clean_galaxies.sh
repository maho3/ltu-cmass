#!/bin/bash

wdir=/ocean/projects/phy240015p/mho1/cmass-ili/quijote/varnoise/L1000-N128
N=3000

# Confirm with the user
read -p "Are you sure you want to clean directories in $wdir? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Operation canceled."
    exit 0
fi

# Function to clean directories for a given index
clean_directory() {
    local i=$1
    rm -r "$wdir/$i/galaxies" 2>/dev/null
    rm -r "$wdir/$i/sgc_lightcone" 2>/dev/null
    rm -r "$wdir/$i/mtng_lightcone" 2>/dev/null
    rm -r "$wdir/$i/ngc_lightcone" 2>/dev/null
    rm -r "$wdir/$i/simbig_lightcone" 2>/dev/null
    echo "Processed $i"
}

# Run the loop in parallel
for ((i=0; i<N; i++)); do
    clean_directory "$i" &
    # Limit the number of parallel jobs to avoid overloading the system
    if (( i % 10 == 0 )); then
        wait
    fi
done

# Wait for all background jobs to finish
wait