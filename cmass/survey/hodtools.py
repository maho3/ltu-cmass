import numpy as np
from ..bias.apply_hod import load_snapshot, run_snapshot


class HODEngine():
    def __init__(self, cfg, snap_times, simdir):
        print('init HOD engine')
        self.cfg = cfg
        self.snap_times = snap_times
        self.simdir = simdir

    def __call__(self, snap_idx, hlo_idx, z):
        print('calling HOD engine')
        a = self.snap_times[snap_idx]
        if isinstance(z, float):
            z = np.full(len(hlo_idx), z)

        # Load snapshot
        # Note that we do not need to noise the halos' positons here (even if
        # cfg.bias.hod.noise_uniform is set to True), since we return the
        # difference between the galaxy's position and that of the host halo,
        # so the noise will cancel out.
        hpos, hvel, hmass, hmeta = load_snapshot(self.simdir, a)
        print('loaded')

        # Only keep those selected for the lightcone
        hpos = hpos[hlo_idx]
        hvel = hvel[hlo_idx]
        hmass = hmass[hlo_idx]
        for key in hmeta:
            hmeta[key] = hmeta[key][hlo_idx]
        hmeta['redshift'] = z

        # Run HOD
        gpos, gvel, gmeta = run_snapshot(
            hpos, hvel, hmass, a, self.cfg, hmeta)

        # Get and return desired properties
        ghost = gmeta['hostid']
        dgpos = gpos - hpos[ghost]
        dgvel = gvel - hvel[ghost]
        ghost = ghost.astype(np.uint64)
        dgpos = dgpos.astype(np.float64)
        dgvel = dgvel.astype(np.float64)

        return ghost, dgpos, dgvel
