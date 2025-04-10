"""
Stitches together halo snapshots to create an extrapolated lightcone, applies
a redshift-dependent HOD model, and applies survey masks and selection effects.
Allows for NGC, SGC, and MTNG-like lightcones.

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
    - This script has superseeded lightcone.py and selection.py which stitched
    together NGC lightcones from snapshots of galaxies. This script now jointly
    applies HOD and stitches the lightcone, allowing for z-dependent HODs.
    - This script allows for NGC, SGC, and MTNG-like lightcones.
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

    # Conform to [0, 2pi] and [-pi/2, pi/2]
    ra = np.mod(ra, 360)
    dec = np.mod(dec + 90, 180) - 90

    # Split galid into galsnap and galidx
    galsnap, galidx = split_galsnap_galidx(galid)
    return ra, dec, z, galsnap, galidx


def check_saturation(z, nz_dir, zmin, zmax, geometry):
    if geometry == 'ngc':
        cap = 'North'
    elif geometry == 'sgc':
        cap = 'South'
    elif geometry == 'mtng':
        cap = 'MTNG'
    else:
        raise ValueError(geometry)

    filepath = join(
        nz_dir, f'nz_DR12v5_CMASS_{cap}_zmin{zmin:.4f}_zmax{zmax:.4f}.dat')
    nzobs = np.loadtxt(filepath, usecols=(0,))
    zbins = np.linspace(zmin, zmax, len(nzobs)+1)

    # Check if n(z) is saturated (within 1-sigma of observed n(z))
    nz = np.histogram(z, bins=zbins)[0]
    saturated = np.all(nz >= nzobs - np.sqrt(nzobs))
    return saturated


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

    geometry = cfg.survey.geometry  # whether to use NGC, SGC, or MTNG mask
    geometry = geometry.lower()

    # Load mask
    if geometry == 'ngc':
        maskobs = lc.Mask(boss_dir=cfg.survey.boss_dir,
                          veto=True, is_north=True)
        remap_case = 0
    elif geometry == 'sgc':
        maskobs = lc.Mask(boss_dir=cfg.survey.boss_dir,
                          veto=True, is_north=False)
        remap_case = 3
    elif geometry == 'mtng':
        maskobs = None
        remap_case = 2
    else:
        raise ValueError(
            'Invalid geometry {geometry}. Choose from NGC, SGC, or MTNG.')

    # Get path to lightcone module (where n(z) is saved)
    nz_dir = os.path.dirname(lc.__file__)

    # Setup lightcone constructor
    if 'z_range' in cfg.survey:
        zmin, zmax = cfg.survey.z_range
    else:
        zmin, zmax = 0.4, 0.7
    snap_times = sorted(cfg.nbody.asave)[::-1]  # decreasing order
    snap_times = [a for a in snap_times if (zmin < (1/a - 1) < zmax)]
    lightcone = lc.Lightcone(
        boss_dir=nz_dir if cfg.survey.fix_nz else None,
        mask=maskobs,
        BoxSize=cfg.nbody.L,
        Omega_m=cfg.nbody.cosmo[0],
        zmin=zmin,
        zmax=zmax,
        snap_times=snap_times,
        verbose=True,
        augment=aug_seed,
        remap_case=remap_case,
        seed=42,
        is_north=geometry == 'ngc'
    )

    # Setup HOD model function
    hod_fct = HODEngine(cfg, snap_times, source_path)
    lightcone.set_hod(hod_fct)

    logging.info(f'Stitching snapshots a={snap_times}')
    ra, dec, z, galsnap, galidx = stitch_lightcone(
        lightcone, source_path, snap_times)

    # Check if n(z) is saturated
    saturated = check_saturation(z, nz_dir, zmin, zmax, geometry)

    # Save
    if geometry == 'ngc':
        outdir = join(source_path, 'ngc_lightcone')
    elif geometry == 'sgc':
        outdir = join(source_path, 'sgc_lightcone')
    elif geometry == 'mtng':
        outdir = join(source_path, 'mtng_lightcone')
    os.makedirs(outdir, exist_ok=True)
    save_lightcone(
        outdir,
        ra=ra, dec=dec, z=z,
        galsnap=galsnap,
        galidx=galidx,
        hod_seed=hod_seed,
        aug_seed=aug_seed,
        saturated=saturated,
        config=cfg
    )
    save_cfg(source_path, cfg, field='survey')


if __name__ == "__main__":
    main()
