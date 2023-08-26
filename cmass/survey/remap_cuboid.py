"""
Remaps a cubic box to a cuboid, designed to fit the survey geometry.

Requires:
    - jax
    - https://github.com/maho3/cuboid_remap_jax

Input:
    - xtrues: (N, 3) array of point set positions in cube
    - vtrues: (N, 3) array of point set velocities in cube

Output:
    - xtrues: (N, 3) array of point set positions in cuboid
    - vtrues: (N, 3) array of point set velocities in cuboid
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '16'  # noqa, must be set before jax

import numpy as np
import argparse
import logging
import jax
from os.path import join as pjoin
from cuboid_remap import Cuboid, remap_Lbox
from ..utils import attrdict, get_global_config, setup_logger, timing_decorator


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='remap_as_cuboid')


def build_config():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--simtype', type=str, default='borg2lpt')
    args = parser.parse_args()

    L = 3000           # length of box in Mpc/h
    N = 384            # number of grid points on one side
    # lattice vectors
    u1, u2, u3 = (1, 1, 0), (0, 0, 1), (1, 0, 0)

    return attrdict(
        L=L, N=N,
        u1=u1, u2=u2, u3=u3,
        lhid=args.lhid, simtype=args.simtype,
    )


@timing_decorator
def remap(ppos, pvel, L, u1, u2, u3):
    # remap the particles to the cuboid
    new_size = list(L*np.array(remap_Lbox(u1, u2, u3)))
    logging.info(f'Remapping from {[L]*3} to {new_size}.')

    c = Cuboid(u1, u2, u3)
    ppos = jax.vmap(c.Transform)(ppos/L)*L
    pvel = jax.vmap(c.TransformVelocity)(pvel)
    return ppos, pvel


def main():
    cfg = build_config()
    logging.info(f'Running with config: {cfg}')

    logging.info('Loading halo cube...')
    source_dir = pjoin(
        glbcfg['wdir'], f'{cfg.simtype}/L3000-N384',
        f'{cfg.lhid}')

    hpos = np.load(pjoin(source_dir, 'halo_pos.npy'))
    hvel = np.load(pjoin(source_dir, 'halo_vel.npy'))

    logging.info('Remapping to cuboid...')
    hpos, hvel = remap(hpos, hvel, cfg.L, cfg.u1, cfg.u2, cfg.u3)

    logging.info('Saving cuboid...')
    np.save(pjoin(source_dir, 'halo_cuboid_pos.npy'), hpos)
    np.save(pjoin(source_dir, 'halo_cuboid_vel.npy'), hvel)

    logging.info('Done!')


if __name__ == '__main__':
    main()