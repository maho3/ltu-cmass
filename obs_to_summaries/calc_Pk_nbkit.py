import numpy as np
import argparse
import logging
from os.path import join as pjoin

import nbodykit.lab as nblab
from nbodykit.hod import Zheng07Model
from nbodykit import cosmology

from tools.BOSS_FM import BOSS_angular, BOSS_veto, BOSS_redshift
from tools.utils import get_global_config, get_logger, timing_decorator

logger = get_logger(__name__)


def load_galaxies_sim(source_dir, seed):
    pos = np.load(pjoin(source_dir, 'hod', f'hod{seed}_pos.npy'))
    vel = np.load(pjoin(source_dir, 'hod', f'hod{seed}_vel.npy'))
    return pos, vel


def apply_mask(pos, veto):
    return pos[veto]


@timing_decorator
def main():
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])


if __name__ == "__main__":
    main()
