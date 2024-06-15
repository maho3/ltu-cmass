import numpy as np
from matplotlib import pyplot as plt
import lc

def split_galid (gid) :
    return np.divmod(gid, 2**((gid.itemsize-1)*8))

# how many galaxies we randomly generate per volume
N = 300000

# where mask .ply files and redshift histogram are stored
boss_dir = './testdata'

# scale factors, need to be monotonically decreasing
snap_times = [0.7, 0.65, 0.6, ]

# make the mask (this is a bit expensive, so we do it outside the lightcone so it can be re-used)
print('Starting mask loading...')
m = lc.Mask(boss_dir=boss_dir, veto=False)
print('...finished mask loading')

# run the constructor
l = lc.Lightcone(
    boss_dir=boss_dir, mask=m,
    Omega_m=0.3, zmin=0.40, zmax=0.65,
    snap_times=snap_times,
    verbose=True,

    # NOTE that True is the default setting. This stitches the snapshots before RSD is applied.
    #      This means that the final redshifts overlap a little bit between snapshots, because
    #      they contain the peculiar velocities.
    #      I think the default setting makes a bit more sense, but it shouldn't matter much.
    stitch_before_RSD=True,
)

# add some snapshots
rng = np.random.default_rng()
for snap_idx, a in enumerate(snap_times):
    xgal = rng.random((N, 3)) * 3e3
    vgal = (rng.random((N, 3))-0.5)*300
    vhlo = (rng.random((N, 3))-0.5)*300
    l.add_snap(snap_idx, xgal, vgal, vhlo)

ra, dec, z, galid = l.finalize()

galsnap, galidx = split_galid(galid)
print(f'{galsnap.min()} <= galsnap <= {galsnap.max()}')
print(f'{galidx.min()} <= galidx <= {galidx.max()}')

# check if mask is working correctly
fig, ax = plt.subplots(figsize=(20, 10), ncols=2)
ax_mask = ax[0]
ax_snap = ax[1]
choose = (z > 0.5) * (z < 0.6)

ax_mask.plot(ra[choose], dec[choose], linestyle='none', marker='o', markersize=0.1)
ax_mask.set_xlabel('RA [deg]')
ax_mask.set_ylabel('DEC [deg]')

ax_snap.plot(galsnap, z, linestyle='none', marker='o', markersize=0.1)
ax_snap.set_xlabel('snapshot index')
ax_snap.set_ylabel('redshift')
plt.show()
