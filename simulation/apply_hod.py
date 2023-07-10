import numpy as np
import argparse
import logging
from os.path import join as pjoin

import nbodykit.lab as nblab
from nbodykit.hod import Zheng07Model
from nbodykit import cosmology

from tools.BOSS_FM import thetahod_literature
from tools.utils import get_global_config, get_logger, timing_decorator

logger = get_logger(__name__)


@timing_decorator
def load_halos(source_dir):
    pos = np.load(pjoin(source_dir, 'halo_cuboid_pos.npy'))
    vel = np.load(pjoin(source_dir, 'halo_cuboid_vel.npy'))
    mass = np.load(pjoin(source_dir, 'halo_mass.npy'))
    return pos, vel, mass


@timing_decorator
def populate_hod(
        pos, vel, mass,
        theta, cosmo, redshift, mdef,
        seed=0):
    # create a structured array to hold the halo catalog
    dtype = [('Position', ('f32', 3)),
             ('Velocity', ('f32', 3)),
             ('Mass', 'f32')]
    halos = np.empty(len(x), dtype=dtype)
    halos['Position'] = pos
    halos['Velocity'] = vel
    halos['Mass'] = mass

    source = nblab.ArrayCatalog(halos)
    halos = nblab.HaloCatalog(
        source,
        cosmo=cosmo,
        redshift=redshift,
        mdef=mdef,
    )
    hod = halos.populate(Zheng07Model, seed=seed, **theta)
    return hod


@timing_decorator
def main():
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    source_dir = pjoin(
        glbcfg['wdir'], 'borg-quijote/latin_hypercube_HR-L3000-N384',
        f'{args.lhid}')
    pos, vel, mass = load_halos(source_dir)

    theta = thetahod_literature('reid2014_cmass')
    hod = populate_hod(
        pos, vel, mass,
        theta, cosmology.Planck15, 0, 'vir',
        seed=args.seed
    )

    pos, vel, mass = hod['Position'], hod['Velocity'], hod['Mass']

    logging.info(f'Saving to {source_dir}/hod/hod{args.seed}...')
    np.save(pjoin(source_dir, 'hod', f'hod{args.seed}_pos.npy'), pos)
    np.save(pjoin(source_dir, 'hod', f'hod{args.seed}_vel.npy'), vel)
    np.save(pjoin(source_dir, 'hod', f'hod{args.seed}_mass.npy'), mass)

    logging.info('Done!')


if __name__ == '__main__':
    main()
