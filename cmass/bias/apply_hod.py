"""
Sample an HOD realization from the halo catalog using the Zheng+(2007) model.

Requires:
    - nbodykit

Input:
    - pos: halo positions
    - vel: halo velocities
    - mass: halo masses
    - seed: random seed for sampling HOD parameters

Output:
    - pos: galaxy positions
    - vel: galaxy velocities
"""

import numpy as np
import argparse
import logging
import os
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import nbodykit.lab as nblab
from nbodykit.hod import Zheng07Model
from .tools.hod import thetahod_literature
from ..utils import get_source_path, timing_decorator, load_params


def parse_config(cfg):
    with open_dict(cfg):
        # HOD parameters
        cfg.bias.hod.theta = get_hod_params(cfg.bias.hod.seed)

        # Cosmology
        cfg.nbody.cosmo = load_params(cfg.nbody.lhid, cfg.meta.cosmofile)
    return cfg


def get_hod_params(seed=0):
    theta = thetahod_literature('reid2014_cmass')
    # sample theta based on priors set by Reid+(2014)
    if seed != 0:
        np.random.seed(seed)
        hod_lower_bound = np.array([12.0, 0.1, 13.0, 13.0, 0.])
        hod_upper_bound = np.array([14.0, 0.6, 15.0, 15.0, 1.5])
        keys = ['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha']
        theta = np.random.uniform(hod_lower_bound, hod_upper_bound, size=(5))
        theta = dict(zip(keys, theta))
    return theta


@timing_decorator
def load_cuboid(source_dir):
    pos = np.load(pjoin(source_dir, 'halo_cuboid_pos.npy'))
    vel = np.load(pjoin(source_dir, 'halo_cuboid_vel.npy'))
    mass = np.load(pjoin(source_dir, 'halo_mass.npy'))
    return pos, vel, mass


@timing_decorator
def populate_hod(
        pos, vel, mass,
        theta, cosmo, redshift, mdef, L,
        seed=0):

    # create a structured array to hold the halo catalog
    dtype = [('Position', (np.float32, 3)),
             ('Velocity', (np.float32, 3)),
             ('Mass', np.float32)]
    halos = np.empty(len(pos), dtype=dtype)
    halos['Position'] = pos
    halos['Velocity'] = vel
    halos['Mass'] = 10**mass

    source = nblab.ArrayCatalog(halos)
    source.attrs['BoxSize'] = L*np.array([np.sqrt(2), 1, 1/np.sqrt(2)])

    cosmology = nblab.cosmology.Planck15.clone(
        h=cosmo[2],
        Omega0_b=cosmo[1],
        Omega0_cdm=cosmo[0] - cosmo[1],
        m_ncdm=None,
        n_s=cosmo[3])

    # We don't need to match sigma8, because HOD is invariant.
    # cosmology = cosmology.match(sigma8=cosmo[4])

    halos = nblab.HaloCatalog(
        source,
        cosmo=cosmology,
        redshift=redshift,
        mdef=mdef,
    )
    hod = halos.populate(Zheng07Model, seed=seed, **theta)
    return hod


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = parse_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    logging.info('Loading halos...')
    source_path = get_source_path(cfg, cfg.sim)
    pos, vel, mass = load_cuboid(source_path)

    logging.info('Populating HOD...')
    hod = populate_hod(
        pos, vel, mass,
        cfg.bias.hod.theta, cfg.nbody.cosmo, 0, 'vir', cfg.nbody.L,
        seed=cfg.bias.hod.seed
    )

    pos, vel = np.array(hod['Position']), np.array(hod['Velocity'])

    savepath = pjoin(source_path, 'hod')
    os.makedirs(savepath, exist_ok=True)

    logging.info(f'Saving to {savepath}/hod{cfg.bias.hod.seed}...')
    # galaxy positions [Mpc/h]
    np.save(pjoin(savepath, f'hod{cfg.bias.hod.seed}_pos.npy'), pos)
    # galaxy velocities [km/s]
    np.save(pjoin(savepath, f'hod{cfg.bias.hod.seed}_vel.npy'), vel)

    logging.info('Done!')


if __name__ == '__main__':
    main()
