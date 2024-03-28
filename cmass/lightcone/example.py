import numpy as np
import lc

# how many galaxies we randomly generate per volume
N = 3000000

# scale factors, need to be monotonically decreasing
snap_times = [0.7, 0.65, 0.6, ]

# run the constructor
l = lc.Lightcone(
    boss_dir='/tigress/lthiele/boss_dr12',
    Omega_m=0.3, zmin=0.40, zmax=0.65,
    snap_times=snap_times,
    verbose=True
)

# add some snapshots
rng = np.random.default_rng()
for snap_idx, a in enumerate(snap_times) :
    xgal = rng.random((N, 3)) * 3e3
    vgal = (rng.random((N, 3))-0.5)*300
    vhlo = (rng.random((N, 3))-0.5)*300
    l.add_snap(snap_idx, xgal, vgal, vhlo)

ra, dec, z = l.finalize()
