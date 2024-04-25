"""
Sample an HOD realization from the halo catalog using the Zheng+(2007) model.

Requires:
    - halotools
    - astropy

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
import logging
import os
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import astropy.cosmology as cosmology
from .tools.hod import (thetahod_literature,
                        build_halo_catalog, build_HOD_model)
from ..utils import get_source_path, timing_decorator, load_params


def parse_config(cfg):
    with open_dict(cfg):
        # HOD parameters
        cfg.bias.hod.theta = get_hod_params(cfg.bias.hod.seed)

        # Cosmology
        cfg.nbody.cosmo = load_params(
            cfg.nbody.lhid, cfg.meta.cosmofile)
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
        theta = [float(x) for x in theta]
        theta = dict(zip(keys, theta))
    return theta


@ timing_decorator
def load_cuboid(source_dir):
    pos = np.load(pjoin(source_dir, 'halo_cuboid_pos.npy'))
    vel = np.load(pjoin(source_dir, 'halo_cuboid_vel.npy'))
    mass = np.load(pjoin(source_dir, 'halo_mass.npy'))
    return pos, vel, mass


@ timing_decorator
def populate_hod(
    pos, vel, mass,
    cosmo, cfg, seed=0, mdef='vir'
):
    if isinstance(cosmo, list):
        cosmo = cosmology.FlatLambdaCDM(
            H0=cosmo[2]*100,
            Om0=cosmo[0],
            Ob0=cosmo[1],
        )  # sigma8 and ns are not needed

    BoxSize = cfg.nbody.L*np.array([np.sqrt(2), 1, 1/np.sqrt(2)])
    catalog = build_halo_catalog(
        pos, vel, mass, cfg.nbody.zf, BoxSize, cosmo,
        mdef=mdef
    )

    hod_params = cfg.bias.hod.theta

    hod = build_HOD_model(
        cosmo, cfg.nbody.zf, hod_model='zheng07', mdef=mdef,
        **hod_params
    )

    hod.populate_mock(catalog, seed=seed)

    galcat = hod.mock.galaxy_table.as_array()

    return galcat


@ timing_decorator
@ hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'sim', 'nbody', 'bias'])

    # Build run config
    cfg = parse_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    logging.info('Loading halos...')
    source_path = get_source_path(cfg, cfg.sim)
    pos, vel, mass = load_cuboid(source_path)

    logging.info('Populating HOD...')
    hod = populate_hod(
        pos, vel, mass,
        cfg.nbody.cosmo, cfg,
        seed=cfg.bias.hod.seed
    )

    gpos = np.array([hod['x'], hod['y'], hod['z']]).T
    gvel = np.array([hod['vx'], hod['vy'], hod['vz']]).T
    meta = {'galtype': hod['galtype'], 'hostid': hod['hostid']}

    savepath = pjoin(source_path, 'hod')
    os.makedirs(savepath, exist_ok=True)

    logging.info(f'Saving to {savepath}/hod{cfg.bias.hod.seed}...')
    # galaxy positions [Mpc/h]
    np.save(pjoin(savepath, f'hod{cfg.bias.hod.seed}_pos.npy'), gpos)
    # galaxy velocities [km/s]
    np.save(pjoin(savepath, f'hod{cfg.bias.hod.seed}_vel.npy'), gvel)
    # galaxy metadata
    np.savez(pjoin(savepath, f'hod{cfg.bias.hod.seed}_meta.npz'), **meta)

    logging.info('Done!')


if __name__ == '__main__':
    main()
