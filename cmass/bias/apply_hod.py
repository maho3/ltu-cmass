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

import nbodykit.lab as nblab
from nbodykit.hod import Zheng07Model
from .tools.hod import thetahod_literature
from ..utils import (attrdict, get_global_config, get_source_path,
                     setup_logger, timing_decorator, load_params)


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='apply_hod')


def build_config():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-L', type=int, default=3000)  # side length of box in Mpc/h
    parser.add_argument(
        '-N', type=int, default=384)  # number of grid points on one side
    parser.add_argument(
        '--lhid', type=int, required=True)  # which cosmology to use
    parser.add_argument(
        '--simtype', type=str, default='borg2lpt')  # which base simulation
    args = parser.parse_args()

    theta = get_hod_params(args.seed)  # HOD parameters
    cosmo = load_params(args.lhid, glbcfg['cosmofile'])

    return attrdict(
        L=args.L, N=args.N,
        lhid=args.lhid, seed=args.seed, simtype=args.simtype,
        theta=theta, cosmo=cosmo
    )


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


def main():
    cfg = build_config()
    logging.info(f'Running with config: {cfg}')

    logging.info('Loading halos...')
    source_dir = get_source_path(
        glbcfg["wdir"], f"borg{cfg.order}lpt", cfg.L, cfg.N)
    pos, vel, mass = load_cuboid(source_dir)

    logging.info('Populating HOD...')
    hod = populate_hod(
        pos, vel, mass,
        cfg.theta, cfg.cosmo, 0, 'vir', cfg.L,
        seed=cfg.seed
    )

    pos, vel = np.array(hod['Position']), np.array(hod['Velocity'])

    savepath = pjoin(source_dir, 'hod')
    os.makedirs(savepath, exist_ok=True)
    logging.info(f'Saving to {savepath}/hod{cfg.seed}...')
    np.save(pjoin(savepath, f'hod{cfg.seed}_pos.npy'), pos)
    np.save(pjoin(savepath, f'hod{cfg.seed}_vel.npy'), vel)

    logging.info('Done!')


if __name__ == '__main__':
    main()
