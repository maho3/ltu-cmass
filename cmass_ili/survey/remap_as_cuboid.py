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
os.environ['OPENBLAS_NUM_THREADS'] = '16'  # noqa

import numpy as np
import argparse
import logging
import jax
from os.path import join as pjoin
from cuboid_remap import Cuboid
from ..tools.utils import get_global_config, setup_logger, timing_decorator


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='remap_as_cuboid')


@timing_decorator
def remap(ppos, pvel):
    # remap the particles to the cuboid
    Lbox = 3000
    u1, u2, u3 = (1, 1, 0), (0, 0, 1), (1, 0, 0)

    c = Cuboid(u1, u2, u3)
    ppos = jax.vmap(c.Transform)(ppos/Lbox)*Lbox
    pvel = jax.vmap(c.TransformVelocity)(pvel)
    return ppos, pvel


def main():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--simtype', type=str, default='borg-quijote')
    args = parser.parse_args()

    logging.info(f'Running with lhid={args.lhid}...')

    logging.info('Loading halo cube...')
    source_dir = pjoin(
        glbcfg['wdir'], f'{args.simtype}/latin_hypercube_HR-L3000-N384',
        f'{args.lhid}')

    xtrues = np.load(pjoin(source_dir, 'halo_pos.npy'))
    vtrues = np.load(pjoin(source_dir, 'halo_vel.npy'))

    logging.info('Remapping to cuboid...')
    xtrues, vtrues = remap(xtrues, vtrues)

    logging.info('Saving cuboid...')
    np.save(pjoin(source_dir, 'halo_cuboid_pos.npy'), xtrues)
    np.save(pjoin(source_dir, 'halo_cuboid_vel.npy'), vtrues)

    logging.info('Done!')


if __name__ == '__main__':
    main()
