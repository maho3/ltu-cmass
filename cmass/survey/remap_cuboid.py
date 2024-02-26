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
import logging
import jax
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf
from cuboid_remap import Cuboid, remap_Lbox
from ..utils import get_source_path, timing_decorator


@timing_decorator
def remap(ppos, pvel, cfg):
    L = cfg.nbody.L
    u1, u2, u3 = cfg.survey.u1, cfg.survey.u2, cfg.survey.u3

    # remap the particles to the cuboid
    new_size = list(L*np.array(remap_Lbox(u1, u2, u3)))
    logging.info(f'Remapping from {[L]*3} to {new_size}.')

    c = Cuboid(u1, u2, u3)
    ppos = jax.vmap(c.Transform)(ppos/L)*L
    pvel = jax.vmap(c.TransformVelocity)(pvel)
    return ppos, pvel


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    logging.info('Loading halo cube...')
    source_path = get_source_path(cfg, cfg.sim)

    hpos = np.load(pjoin(source_path, 'halo_pos.npy'))
    hvel = np.load(pjoin(source_path, 'halo_vel.npy'))

    logging.info('Remapping to cuboid...')
    hpos, hvel = remap(hpos, hvel, cfg)

    logging.info('Saving cuboid...')
    np.save(pjoin(source_path, 'halo_cuboid_pos.npy'), hpos)
    np.save(pjoin(source_path, 'halo_cuboid_vel.npy'), hvel)

    logging.info('Done!')


if __name__ == '__main__':
    main()
