"""
~~~ DEPRECATED IN FAVOR OF hodlightcone.py ~~~

Stitches together snapshots to create an extrapolated lightcone and
applies CMASS NGC survey mask and selection effects.

Input:
    - halos.h5
        - vel: halo velocities
    - galaxies/hod{hod_seed}.h5
        - pos: halo positions
        - vel: halo velocities
        - hostid: host halo ID

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
import h5py
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from ..utils import get_source_path, timing_decorator, save_cfg
from ..nbody.tools import parse_nbody_config
from .tools import save_lightcone, load_galaxies
try:
    from ..lightcone import lc
except ImportError:
    raise ImportError(
        'Lightcone extrapolation not compiled. Please `make` the '
        'lightcone package in cmass/lightcone')


def load_halo_velocities(source_dir, a):
    filepath = join(source_dir, 'halos.h5')
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
        cfg, ['meta', 'sim', 'multisnapshot', 'nbody', 'bias', 'survey'])

    # Build run config
    cfg = parse_nbody_config(cfg)
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
        augment=aug_seed,
        seed=42
    )

    logging.info(f'Stitching snapshots a={snap_times}')
    ra, dec, z, galsnap, galidx = stitch_lightcone(
        lightcone, source_path, snap_times, hod_seed)

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
