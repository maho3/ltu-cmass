import numpy as np
from scipy.spatial import cKDTree
import argparse
import logging
import jax
from os.path import join as pjoin
from cuboid_remap import Cuboid

from tools.freecode import TruncatedPowerLaw, sample_3d
from tools.utils import get_global_config, get_logger, timing_decorator

logger = logging.getLogger(__name__)


def remap(ppos, pvel):
    # remap the particles to the cuboid
    Lbox = 3000
    u1, u2, u3 = (1, 1, 0), (0, 0, 1), (1, 0, 0)

    c = Cuboid(u1, u2, u3)
    ppos = jax.vmap(c.Transform)(ppos/Lbox)*Lbox
    pvel = jax.vmap(c.TransformVelocity)(pvel)
    return ppos, pvel


def main():
    # Load global configuration
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    args = parser.parse_args()

    logging.info(f'Running with lhid={args.lhid}...')

    logging.info('Loading halo cube...')
    source_dir = pjoin(
        glbcfg['wdir'], 'borg-quijote/latin_hypercube_HR-L3000-N384',
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
