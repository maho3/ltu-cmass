"""
Stitches together snapshots to create an extrapolated lightcone and
applies CMASS NGC survey mask and selection effects.

Input:
    - halos.h5
        - vel: halo velocities
    - hod/galaxies{hod_seed}.h5
        - pos: halo positions
        - vel: halo velocities
        - hostid: host halo ID

Output:
    - obs/lightcone{hod_seed}.h5
        - ra: right ascension
        - dec: declination
        - z: redshift
        - galsnap: snapshot index
        - galidx: galaxy index

NOTE:
    - This only works for snapshot mode, wherein lightcone evolution is
    mimicked by stitching snapshots together. For the non-snapshot mode
    alternative, use 'ngc_selection.py'.
    - The fiber collisions are applied in-sync with resampling.
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
from .tools import save_lightcone, load_galaxies
try:
    from ..lightcone import lc
except ImportError:
    raise ImportError(
        'Lightcone extrapolation not compiled. Please `make` the '
        'lightcone package in cmass/lightcone')


def load_halo_velocities(source_dir, a):
    filepath = pjoin(source_dir, 'halos.h5')
    with h5py.File(filepath, 'r') as f:
        key = f'{a:.6f}'
        vel = f[key]['vel'][...]
    return vel


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
    maskobs = lc.Mask(boss_dir=cfg.survey.boss_dir, veto=True)

    # Get path to lightcone module (where n(z) is saved)
    nz_dir = os.path.dirname(lc.__file__)

    # Setup lightcone constructor
    snap_times = sorted(cfg.nbody.asave)[::-1]  # decreasing order
    lightcone = lc.Lightcone(
        boss_dir=nz_dir,
        mask=maskobs,
        Omega_m=cfg.nbody.cosmo[0],
        zmin=0.4,
        zmax=0.7,
        snap_times=snap_times,
        verbose=True,
        stitch_before_RSD=True,
        augment=0,
        seed=42
    )

    logging.info(f'Stitching snapshots a={snap_times}')
    ra, dec, z, galsnap, galidx = stitch_lightcone(
        lightcone, source_path, snap_times, hod_seed)

    # Save
    outdir = pjoin(source_path, 'lightcone')
    os.makedirs(outdir, exist_ok=True)
    save_lightcone(outdir, ra, dec, z, galsnap, galidx,
                   hod_seed=hod_seed)


if __name__ == "__main__":
    main()
