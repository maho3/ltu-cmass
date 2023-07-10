import numpy as np
import argparse
import logging
from os.path import join as pjoin
from astropy.stats import scott_bin_width

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


def xyz_to_sky(pos, vel, cosmo):
    return nblab.transform.CartesianToSky(pos, cosmo)


def sky_to_xyz(rdz, cosmo):
    return nblab.transform.SkyToCartesian(rdz, cosmo)


@timing_decorator
def apply_mask(rdz):
    # Apply veto mask
    len_rdz = len(rdz)
    mask = BOSS_redshift(rdz[:, -1])
    rdz = rdz[mask]

    inpoly = BOSS_angular(*rdz[:, :-1].T)
    inveto = BOSS_veto(*rdz[:, :-1].T)
    mask = inpoly & (~inveto)
    rdz = rdz[mask]

    logging.info(f'Fraction of galaxies kept: {len(rdz) / len_rdz:.3f}')
    return rdz


@timing_decorator
def reweight(rdz):
    _, be = scott_bin_width(rdz[:, -1], True)
    bind = np.digitize(rdz[:, -1], bins)


@timing_decorator
def main():
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    source_dir = pjoin(
        glbcfg['wdir'], 'borg-quijote/latin_hypercube_HR-L3000-N384',
        f'{args.lhid}')

    # Load galaxies
    pos, vel = load_galaxies_sim(source_dir, args.seed)

    # Calculate sky coordinates
    rdz = xyz_to_sky(pos, vel, cosmology.Planck15)

    # Apply mask
    rdz = apply_mask(rdz)

    # Reweight

    # Save
    np.save(pjoin(source_dir, 'rdz.npy'), rdz)


if __name__ == "__main__":
    main()
