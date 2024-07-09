"""
Stitches together snapshots to create an extrapolated lightcone and
applies BOSS survey mask.
"""

import os
import numpy as np
import logging
import h5py
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf
from ..utils import (get_source_path, timing_decorator)
from ..nbody.tools import parse_nbody_config
from ..lightcone import lc


def load_galaxies(source_dir, a, seed):
    filepath = pjoin(source_dir, 'hod', f'galaxies{seed}.h5')
    with h5py.File(filepath, 'r') as f:
        key = f'{a:.6f}'
        pos = f[key]['pos'][...]
        vel = f[key]['vel'][...]
        hostid = f[key]['hostid'][...]
    return pos, vel, hostid


def load_halo_velocities(source_dir, a):
    filepath = pjoin(source_dir, 'halos.h5')
    with h5py.File(filepath, 'r') as f:
        key = f'{a:.6f}'
        vel = f[key]['vel'][...]
    return vel


def save_lightcone(outdir, ra, dec, z, galsnap, galidx, hod_seed=0):
    outfile = pjoin(outdir, 'obs', f'lightcone{hod_seed}.h5')
    logging.info(f'Saving lightcone to {outfile}')
    with h5py.File(outfile, 'w') as f:
        f.create_dataset('ra', data=ra)            # Right ascension [deg]
        f.create_dataset('dec', data=dec)          # Declination [deg]
        f.create_dataset('z', data=z)              # Redshift
        f.create_dataset('galsnap', data=galsnap)  # Snapshot index
        f.create_dataset('galidx', data=galidx)    # Galaxy index


def split_galsnap_galidx(gid):
    return np.divmod(gid, 2**((gid.itemsize-1)*8))


def stitch_lightcone(lightcone, source_path, snap_times, hod_seed):
    for i, a in enumerate(snap_times):
        logging.info(f'Loading snapshot at a={a:.6f}...')

        # Load galaxies
        gpos, gvel, hostid = load_galaxies(
            source_path, a, hod_seed)

        # Load halos
        hvel = load_halo_velocities(source_path, a)

        # Get host velocites
        hostvel = hvel[hostid]
        lightcone.add_snap(i, gpos, gvel, hostvel)

    ra, dec, z, galid = lightcone.finalize()

    # Split galid into galsnap and galidx
    galsnap, galidx = split_galsnap_galidx(galid)
    return ra, dec, z, galsnap, galidx


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'nbody', 'bias', 'survey'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))
    source_path = get_source_path(cfg, cfg.sim)
    hod_seed = cfg.bias.hod.seed  # for indexing different hod realizations

    # Check that we are in snapshot_mode
    if not (hasattr(cfg.nbody, 'snapshot_mode') and cfg.nbody.snapshot_mode):
        raise ValueError('snapshot_mode config is false, but ngc_lightcone'
                         ' is only for snapshot mode.')

    # Load mask
    logging.info(f'Loading mask from {cfg.survey.boss_dir}')
    maskobs = lc.Mask(boss_dir=cfg.survey.boss_dir, veto=False)

    # Setup lightcone constructor
    snap_times = sorted(cfg.nbody.asave)[::-1]  # decreasing order
    lightcone = lc.Lightcone(
        boss_dir=cfg.survey.boss_dir,
        mask=maskobs,
        Omega_m=cfg.nbody.cosmo[0],
        zmin=0.4,
        zmax=0.7,
        snap_times=snap_times,
        verbose=True,
        stitch_before_RSD=True,
    )

    logging.info(f'Stitching snapshots a={snap_times}')
    ra, dec, z, galsnap, galidx = stitch_lightcone(
        lightcone, source_path, snap_times, hod_seed)

    # Save
    os.makedirs(pjoin(source_path, 'obs'), exist_ok=True)
    save_lightcone(source_path, ra, dec, z, galsnap, galidx,
                   hod_seed=hod_seed)


if __name__ == "__main__":
    main()
