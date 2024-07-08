"""
Moves a cubic halo catalog to the CMASS footprint, and applies custom cuts.

Requires:
    - pymangle
    - astropy

Input:
    - pos: (N, 3) array of galaxy positions
    - vel: (N, 3) array of galaxy velocities
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # noqa, must be set before jax

import numpy as np
import logging
from os.path import join as pjoin
import jax
from cuboid_remap import Cuboid, remap_Lbox
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict


from .tools import (
    xyz_to_sky, sky_to_xyz, rotate_to_z, random_rotate_translate,
    BOSS_angular, BOSS_veto, BOSS_redshift, BOSS_fiber)
from ..utils import (get_source_path, timing_decorator, load_params)


def parse_config(cfg):
    with open_dict(cfg):
        # Cosmology
        cfg.nbody.cosmo = load_params(cfg.nbody.lhid, cfg.meta.cosmofile)
    return cfg


@timing_decorator
def load_halos_sim(source_dir, seed):
    pos = np.load(pjoin(source_dir, 'halo_pos.npy'))
    vel = np.load(pjoin(source_dir, 'halo_vel.npy'))
    mass = np.load(pjoin(source_dir, 'halo_mass.npy'))
    return pos, vel, mass


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
    pos, vel = pos.copy(), vel.copy()

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
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'nbody', 'bias', 'survey'])

    # Build run config
    cfg = parse_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    source_path = get_source_path(cfg, cfg.sim)

    # Load halos
    pos, vel, mass = load_halos_sim(source_path, cfg.bias.hod.seed)

    # [Optionally] rotate and shuffle cubic volume
    pos, vel = random_rotate_translate(
        pos, L=cfg.nbody.L, vel=vel, seed=cfg.survey.rot_seed)

    # Apply cuboid remapping
    pos, vel = remap(
        pos, vel, cfg.nbody.L,
        cfg.survey.u1, cfg.survey.u2, cfg.survey.u3)

    # Rotate and shift to align with CMASS
    pos, vel = move_to_footprint(
        pos, vel, cfg.survey.mid_rdz, cfg.nbody.cosmo, cfg.nbody.L)

    # Calculate sky coordinates
    rdz = xyz_to_sky(pos, vel, cfg.nbody.cosmo)

    # Append
    rdzm = np.concatenate([rdz, mass[:, None]], axis=1)

    # Custom cuts
    rdzm = custom_cuts(rdzm, cfg)

    # ra, dec, redshift
    np.save(pjoin(source_path, 'halo_rdzm.npy'), rdzm)


if __name__ == "__main__":
    main()
