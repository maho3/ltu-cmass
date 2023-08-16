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
from nbodykit import cosmology
from ..tools.BOSS_FM import thetahod_literature
from ..tools.utils import get_global_config, setup_logger, timing_decorator


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='apply_hod')


@timing_decorator
def load_halos(source_dir):
    pos = np.load(pjoin(source_dir, 'halo_cuboid_pos.npy'))
    vel = np.load(pjoin(source_dir, 'halo_cuboid_vel.npy'))
    mass = np.load(pjoin(source_dir, 'halo_mass.npy'))
    print(pos.max(axis=0), pos.min(axis=0))
    return pos, vel, mass


@timing_decorator
def populate_hod(
        pos, vel, mass,
        theta, cosmo, redshift, mdef,
        seed=0):
    # sample theta based on priors set by Reid+(2014)
    if seed != 0:
        np.random.seed(seed)
        hod_lower_bound = np.array([12.0, 0.1, 13.0, 13.0, 0.])
        hod_upper_bound = np.array([14.0, 0.6, 15.0, 15.0, 1.5])
        keys = ['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha']
        theta = np.random.uniform(hod_lower_bound, hod_upper_bound, size=(5))
        theta = dict(zip(keys, theta))
        print(theta)

    # create a structured array to hold the halo catalog
    dtype = [('Position', (np.float32, 3)),
             ('Velocity', (np.float32, 3)),
             ('Mass', np.float32)]
    halos = np.empty(len(pos), dtype=dtype)
    halos['Position'] = pos
    halos['Velocity'] = vel
    halos['Mass'] = 10**mass

    source = nblab.ArrayCatalog(halos)
    source.attrs['BoxSize'] = [4242.64068712, 3000.,
                               2121.32034356]  # calculated from remap_Lbox
    halos = nblab.HaloCatalog(
        source,
        cosmo=cosmo,
        redshift=redshift,
        mdef=mdef,
    )
    hod = halos.populate(Zheng07Model, seed=seed, **theta)
    return hod


def main():
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--simtype', type=str, default='borg2lpt')
    args = parser.parse_args()

    logging.info(f'Running with lhid={args.lhid}, seed={args.seed}...')
    logging.info('Loading halos...')
    source_dir = pjoin(
        glbcfg['wdir'], f'{args.simtype}/L3000-N384',
        f'{args.lhid}')
    pos, vel, mass = load_halos(source_dir)

    logging.info('Populating HOD...')
    theta = thetahod_literature('reid2014_cmass')
    hod = populate_hod(
        pos, vel, mass,
        theta, cosmology.Planck15, 0, 'vir',
        seed=args.seed
    )

    pos, vel = np.array(hod['Position']), np.array(hod['Velocity'])

    savepath = pjoin(source_dir, 'hod')
    os.makedirs(savepath, exist_ok=True)
    logging.info(f'Saving to {savepath}/hod{args.seed}...')
    np.save(pjoin(savepath, f'hod{args.seed}_pos.npy'), pos)
    np.save(pjoin(savepath, f'hod{args.seed}_vel.npy'), vel)

    logging.info('Done!')


if __name__ == '__main__':
    main()
