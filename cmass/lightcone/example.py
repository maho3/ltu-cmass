import numpy as np
from matplotlib import pyplot as plt
import lc

# how many galaxies we randomly generate per volume
N = 3000000

# where mask .ply files and redshift histogram are stored
boss_dir = '../../data/obs'

# scale factors, need to be monotonically decreasing
snap_times = [0.7, 0.65, 0.6, ]

# make the mask (this is a bit expensive, so we do it outside the lightcone so it can be re-used)
print('Starting mask loading...')
m = lc.Mask(boss_dir=boss_dir)
print('...finished mask loading')

# run the constructor
l = lc.Lightcone(
    boss_dir=boss_dir, mask=m,
    Omega_m=0.3, zmin=0.40, zmax=0.65,
    snap_times=snap_times,
    verbose=True
)

# add some snapshots
rng = np.random.default_rng()
for snap_idx, a in enumerate(snap_times):
    xgal = rng.random((N, 3)) * 3e3
    vgal = (rng.random((N, 3))-0.5)*300
    vhlo = (rng.random((N, 3))-0.5)*300
    l.add_snap(snap_idx, xgal, vgal, vhlo)

ra, dec, z = l.finalize()

# check if mask is working correctly
fig, ax = plt.subplots(figsize=(10, 10))
choose = (z > 0.5) * (z < 0.6)
ax.plot(ra[choose], dec[choose], linestyle='none', marker='o', markersize=0.1)
ax.set_xlabel('RA [deg]')
ax.set_ylabel('DEC [deg]')
plt.show()
