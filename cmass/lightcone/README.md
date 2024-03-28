Code adapted from [here](https://github.com/leanderthiele/nuvoid_production).

The Makefile should hopefully work out of the box.
External dependences are GSL and pybind11 (I have copied Martin White's cuboid remapping code,
pymangle, and a minimal healpix into this repository).

Basic usage example please see `example.py`.
The heavy sections of the code are multi-threaded, control via `OMP_NUM_THREADS`.

## Constructor

**Mandatory**
* `boss_dir`: a directory containing the following files:
    * all `.ply` files for the angular mask. If `veto=False`,
      only `mask_DR12v5_CMASS_North.ply` is required.
    * a text file named `nz_DR12v5_CMASS_North_zmin%.4f_zmax%.4f.dat`,
      where the placeholders are filled by the `zmin` and `zmax` arguments.
      This file defines the desired redshift distribution.
      It is assumed that the histogramming is in uniform redshift bins
      between `zmin` and `zmax`.
      Each line of the file should contain a single integer for the number
      of galaxies in the corresponding bin.
* `Omega_m`
* `zmin`, `zmax`: the redshift boundaries
* `snap_times`: a monotonically *decreasing* list of scale factors
  (i.e., increasing redshift)

**Optional**
* `BoxSize = 3e3`: Mpc/h
* `remap_case = 0`: either 0 or 1
* `correct = True`: whether extrapolation dependent on host halo velocity is performed
* `veto = True`: whether the six veto masks are applied
* `stitch_before_RSD = True`: I wasn't able to figure out whether real or redshift space
  galaxy positions are appropriate when deciding how to stitch the snapshots.
  It probably doesn't matter much but can be played with.
* `verbose = False`
* `augment = 0`: between 0 and 47. There are 96 possible augmentations implemented,
  namely the product of 2 remaps (above argument), 8 reflections, 6 transpositions.
  Please see the function `Lightcone::remap_snapshot` for the exact definition if it
  is important.
* `seed = 137`: randomness is introduced in downsampling to n(z) and fiber collisions.
  I believe determinism will only be possible when running on a single thread.


## adding galaxies

Using the `add_snap` method. This can be called in any order of snapshots.

**Arguments**
* `snap_idx`: integer index into the `snap_times` array
* `xgal`: Nx3 numpy array, galaxy positions in Mpc/h
* `vgal`: Nx3 numpy array, physical km/s.
* `vhlo`: Nx3 numpy array, physical km/s. In principle this can be empty if `correct=False`,
          but I haven't tried this.


## getting result

Using the `finalize` method. Call after calling `add_snap` for all snapshots.
Returns a tuple RA, DEC, z, where the angles are in degrees.
These are already rotated into the NGC footprint.
