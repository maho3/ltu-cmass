"""
~~~ DEPRECATED IN FAVOR OF hodlightcone.py ~~~

Places a cubic simulation in the footprint of the MTNG lightcone. Measures
ra/dec/z. However, it does not apply survey systematics (geometry, fiber
collisions, etc).

Input:
    - galaxies/hod{hod_seed}.h5
        - pos: halo positions
        - vel: halo velocities

Output:
    - lightcone/hod{hod_seed}_aug{augmentation_seed}.h5
        - ra: right ascension
        - dec: declination
        - z: redshift
        - galsnap: snapshot index
        - galidx: galaxy index

NOTE:
    - This only works for non-snapshot mode, wherein lightcone evolution is
    ignored. For the snapshot mode alternative, use 'ngc_lightcone.py'.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # noqa, must be set before jax

import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf

from .tools import (
    xyz_to_sky, random_rotate_translate, save_lightcone, load_galaxies)
from ..utils import get_source_path, timing_decorator, save_cfg
from ..nbody.tools import parse_nbody_config
from .selection import custom_cuts, reweight


@timing_decorator
def reweight_mtng(rdz, wdir='./data'):
    n_z = np.load(
        join(wdir, 'obs',
             'n-z_MTNG.npy'),
        allow_pickle=True).item()
    be, hobs = n_z['be'], n_z['h']

    return reweight(rdz, wdir, be=be, hobs=hobs)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'multisnapshot', 'nbody', 'bias', 'survey'])

    # Build run config
    if cfg.multisnapshot:
        raise ValueError('This script only works for single-snapshot mode.')

    cfg = parse_nbody_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    hod_seed = cfg.bias.hod.seed  # for indexing different hod realizations
    aug_seed = cfg.survey.aug_seed  # for rotating and shuffling

    # Load galaxies
    pos, vel, _ = load_galaxies(source_path, cfg.nbody.af, hod_seed)

    # [Optionally] rotate and shuffle cubic volume
    pos, vel = random_rotate_translate(
        pos, L=cfg.nbody.L, vel=vel, seed=aug_seed)

    # Calculate sky coordinates
    rdz = xyz_to_sky(pos, vel, cfg.nbody.cosmo)

    # Custom cuts
    rdz = custom_cuts(rdz, cfg)

    # Reweight (Todo: fiber collisions should iterate with this?)
    rdz = reweight_mtng(rdz, cfg.meta.wdir)

    # Save
    outdir = join(source_path, 'mtng_lightcone')
    os.makedirs(outdir, exist_ok=True)
    save_lightcone(
        outdir,
        ra=rdz[:, 0], dec=rdz[:, 1], z=rdz[:, 2],
        galsnap=np.zeros(len(rdz), dtype=int),
        galidx=np.arange(len(rdz)),
        hod_seed=hod_seed,
        aug_seed=aug_seed)
    # save_cfg(source_path, cfg, field='survey')


if __name__ == "__main__":
    main()
