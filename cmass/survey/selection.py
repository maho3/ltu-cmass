"""
~~~ DEPRECATED IN FAVOR OF hodlightcone.py ~~~

Reshapes a cubic simulation into a lightcone footprint, measures ra/dec/z,
and applies CMASS NGC/SGC survey mask and selection effects.

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
    - The fiber collisions are applied before resampling, and not in-sync with
    it. This often leads to a slightly lower n(z) in some z-bins than desired.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # noqa, must be set before jax

import numpy as np
import logging
from os.path import join
import jax
from cuboid_remap import Cuboid, remap_Lbox
import hydra
from omegaconf import DictConfig, OmegaConf

from .tools import (
    xyz_to_sky, sky_to_xyz, rotate_to_z, random_rotate_translate,
    apply_mask,
    save_lightcone, load_galaxies)
from ..utils import get_source_path, timing_decorator, save_cfg, clean_up
from ..nbody.tools import parse_nbody_config


@timing_decorator
def remap(ppos, pvel, L, u1, u2, u3):
    # remap the particles to the cuboid
    new_size = list(L*np.array(remap_Lbox(u1, u2, u3)))
    logging.info(f'Remapping from {[L]*3} to {new_size}.')

    c = Cuboid(u1, u2, u3)
    ppos = jax.vmap(c.Transform)(ppos/L)*L
    pvel = jax.vmap(c.TransformVelocity)(pvel)
    return np.array(ppos), np.array(pvel)


@timing_decorator
def move_to_footprint(pos, vel, mid_rdz, cosmo, L):
    pos = pos.copy()  # comoving positions [Mpc/h]
    vel = vel.copy()  # physical velocities [km/s]

    # shift to origin
    pos -= pos.mean(axis=0)

    # find footprint center in comoving coordinates, conditioned on cosmo
    mid_xyz = sky_to_xyz(mid_rdz, cosmo)

    # rotate to same orientation as footprint
    _, rot_inv = rotate_to_z(mid_xyz, cosmo)
    pos = rot_inv.apply(pos)
    vel = rot_inv.apply(vel)

    # shift to center of footprint
    pos += mid_xyz

    return pos, vel


@timing_decorator
def custom_cuts(rdz, cfg):
    logging.info('Applying custom cuts...')
    if 'ra_range' in cfg.survey:
        ra_range = cfg.survey.ra_range
        mask = (rdz[:, 0] > ra_range[0]) & (rdz[:, 0] < ra_range[1])
        rdz = rdz[mask]
    if 'dec_range' in cfg.survey:
        dec_range = cfg.survey.dec_range
        mask = (rdz[:, 1] > dec_range[0]) & (rdz[:, 1] < dec_range[1])
        rdz = rdz[mask]
    if 'z_range' in cfg.survey:
        z_range = cfg.survey.z_range
        mask = (rdz[:, 2] > z_range[0]) & (rdz[:, 2] < z_range[1])
        rdz = rdz[mask]
    return rdz


@timing_decorator
def reweight(rdz, wdir='./data', is_North=True, be=None, hobs=None):
    if (be is None) or (hobs is None):
        # load observed n(z)
        n_z = np.load(
            join(wdir, 'obs',
                 'n-z_DR12v5_CMASS_North.npy' if is_North else 'n-z_DR12v5_CMASS_South.npy'),
            allow_pickle=True).item()
        be, hobs = n_z['be'], n_z['h']

    # load simulated n(z)
    hsim, _ = np.histogram(rdz[:, -1], bins=be)
    hobs, hsim = hobs.astype(int), hsim.astype(int)
    for i in range(len(hsim)):
        if hsim[i] < hobs[i]:
            logging.warning(
                f'hsim ({hsim[i]}) < hobs ({hobs[i]}) in bin: '
                f'{be[i]:.5f}<z<{be[i+1]:.5f},\n'
                'More observed galaxies than simulated. '
                'Reweighting may not be accurate.')

    # sample at most as many as observed
    hsamp = np.minimum(hobs, hsim)

    # mask
    mask = np.zeros(len(rdz), dtype=bool)
    bind = np.digitize(rdz[:, -1], be) - 1
    for i in range(len(be) - 1):
        in_bin = np.argwhere(bind == i).flatten()
        in_samp = np.random.choice(in_bin, size=hsamp[i], replace=False)
        mask[in_samp] = True

    return rdz[mask]


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
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
    is_North = cfg.survey.is_North  # whther to use NGC or SGC mask

    # Load galaxies
    pos, vel, _ = load_galaxies(source_path, cfg.nbody.af, hod_seed)

    # [Optionally] rotate and shuffle cubic volume
    pos, vel = random_rotate_translate(
        pos, L=cfg.nbody.L, vel=vel, seed=aug_seed)

    # Apply cuboid remapping
    pos, vel = remap(
        pos, vel, cfg.nbody.L,
        cfg.survey.u1, cfg.survey.u2, cfg.survey.u3)

    # Rotate and shift to align with CMASS
    pos, vel = move_to_footprint(
        pos, vel, cfg.survey.mid_rdz, cfg.nbody.cosmo, cfg.nbody.L)

    # Calculate sky coordinates
    rdz = xyz_to_sky(pos, vel, cfg.nbody.cosmo)

    # Apply mask
    rdz = apply_mask(rdz, cfg.meta.wdir, is_North, cfg.survey.fibermode)

    # Custom cuts
    rdz = custom_cuts(rdz, cfg)

    # Reweight (Todo: fiber collisions should iterate with this?)
    rdz = reweight(rdz, cfg.meta.wdir, is_North)

    # Save
    if is_North:
        outdir = join(source_path, 'ngc_lightcone')
    else:
        outdir = join(source_path, 'sgc_lightcone')
    os.makedirs(outdir, exist_ok=True)
    save_lightcone(
        outdir,
        ra=rdz[:, 0], dec=rdz[:, 1], z=rdz[:, 2],
        galsnap=np.zeros(len(rdz), dtype=int),
        galidx=np.arange(len(rdz)),
        hod_seed=hod_seed,
        aug_seed=aug_seed)
    save_cfg(source_path, cfg, field='survey')


if __name__ == "__main__":
    main()
