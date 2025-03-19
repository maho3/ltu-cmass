"""
Stitches together halo snapshots to create an extrapolated lightcone, applies
a redshift-dependent HOD model and
applies CMASS NGC survey mask and selection effects.

Input:
    - halos.h5
        - a: scale factors
        - pos: halo positions
        - vel: halo velocities
        - mass: halo masses

Output:
    - lightcone/hod{hod_seed}_aug{augmentation_seed}.h5
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
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from ..utils import get_source_path, timing_decorator, save_cfg
from ..nbody.tools import parse_nbody_config
from .tools import save_lightcone
from .hodtools import HODEngine
from ..bias.apply_hod import load_snapshot
from ..bias.tools.hod import parse_hod

try:
    from ..lightcone import lc
except ImportError:
    raise ImportError(
        'Lightcone extrapolation not compiled. Please `make` the '
        'lightcone package in cmass/lightcone')


def split_galsnap_galidx(gid):
    return np.divmod(gid, 2**((gid.itemsize-1)*8))


def stitch_lightcone(lightcone, source_path, snap_times):
    # Load snapshots
    for snap_idx, a in enumerate(snap_times):
        logging.info(f'Loading snapshot at a={a:.6f}...')
        hpos, hvel, _, _ = load_snapshot(source_path, a)
        lightcone.add_snap(snap_idx, hpos, hvel)

    # Run lightcone
    ra, dec, z, galid = lightcone.finalize()

    # Split galid into galsnap and galidx
    galsnap, galidx = split_galsnap_galidx(galid)
    return ra, dec, z, galsnap, galidx


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'multisnapshot', 'nbody', 'bias', 'survey'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    hod_seed = cfg.bias.hod.seed  # for indexing different hod realizations
    aug_seed = cfg.survey.aug_seed  # for rotating and shuffling
    is_North = cfg.survey.is_North  # whther to use NGC or SGC mask
    if not is_North:
        raise NotImplementedError(
            'SGC mask not implemented yet in multisnapshot mode.')

    # Load mask
    logging.info(f'Loading mask from {cfg.survey.boss_dir}')
    maskobs = lc.Mask(boss_dir=cfg.survey.boss_dir, veto=True)

    # Setup lightcone constructor
    snap_times = sorted(cfg.nbody.asave)[::-1]  # decreasing order
    zmin, zmax = 0.4, 0.7  # ngc redshift range
    snap_times = [a for a in snap_times if (zmin < (1/a - 1) < zmax)]
    lightcone = lc.Lightcone(
        boss_dir=None,
        mask=maskobs,
        Omega_m=cfg.nbody.cosmo[0],
        zmin=zmin,
        zmax=zmax,
        snap_times=snap_times,
        verbose=True,
        augment=0,
        seed=42
    )

    # Setup HOD model function
    hod_fct = HODEngine(cfg, snap_times, source_path)
    lightcone.set_hod(hod_fct)

    logging.info(f'Stitching snapshots a={snap_times}')
    ra, dec, z, galsnap, galidx = stitch_lightcone(
        lightcone, source_path, snap_times)

    # Save
    if is_North:
        outdir = join(source_path, 'ngc_lightcone')
    else:
        outdir = join(source_path, 'sgc_lightcone')
    os.makedirs(outdir, exist_ok=True)
    save_lightcone(
        outdir,
        ra=ra, dec=dec, z=z,
        galsnap=galsnap,
        galidx=galidx,
        hod_seed=hod_seed,
        aug_seed=aug_seed
    )
    save_cfg(source_path, cfg, field='survey')


if __name__ == "__main__":
    main()
