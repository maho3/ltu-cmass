"""
Applies BOSS survey mask to a lightcone-shaped volume of galaxies.

Requires:
    - nbodykit
    - pymangle
    - astropy

Input:
    - pos: (N, 3) array of galaxy positions
    - vel: (N, 3) array of galaxy velocities
"""

import os
import numpy as np
import argparse
import logging
from os.path import join as pjoin
from scipy.spatial.transform import Rotation as R

import nbodykit.lab as nblab
from nbodykit import cosmology

from tools.BOSS_FM import BOSS_angular, BOSS_veto, BOSS_redshift
from tools.utils import get_global_config, get_logger, timing_decorator

# logger = get_logger(__name__)


@timing_decorator
def load_galaxies_sim(source_dir, seed):
    pos = np.load(pjoin(source_dir, 'hod', f'hod{seed}_pos.npy'))
    vel = np.load(pjoin(source_dir, 'hod', f'hod{seed}_vel.npy'))
    return pos, vel


@timing_decorator
def rotate(pos, vel):
    cmass_cen = [-924.42673929,  -44.04583784,
                 750.98510587]  # mean of comoving range
    r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]).as_matrix()
    pos, vel = pos@r, vel@r
    pos -= (pos.max(axis=0) + pos.min(axis=0))/2
    pos += cmass_cen
    return pos, vel


def xyz_to_sky(pos, vel, cosmo):
    return nblab.transform.CartesianToSky(pos, cosmo)


@timing_decorator
def apply_mask(rdz):
    logging.info('Applying redshift cut...')
    len_rdz = len(rdz)
    mask = BOSS_redshift(rdz[:, -1])
    rdz = rdz[mask]

    logging.info('Applying angular mask...')
    inpoly = BOSS_angular(*rdz[:, :-1].T)

    logging.info('Applying veto mask...')
    inveto = BOSS_veto(*rdz[:, :-1].T)
    mask = inpoly & (~inveto)
    rdz = rdz[mask]

    logging.info(f'Fraction of galaxies kept: {len(rdz) / len_rdz:.3f}')
    return rdz


@timing_decorator
def reweight(rdz):
    n_z = np.load(
        pjoin('data', 'obs', 'n-z_DR12v5_CMASS_North.npy'),
        allow_pickle=True).item()
    be, hobs = n_z['be'], n_z['h']

    hsim, _ = np.histogram(rdz[:, -1], bins=be)
    samp_weight = hobs / hsim
    bind = np.digitize(rdz[:, -1], be) - 1
    mask = np.random.rand(len(rdz)) < samp_weight[bind]
    return rdz[mask]


def main():
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--simtype', type=str, default='borg-quijote')
    args = parser.parse_args()

    source_dir = pjoin(
        glbcfg['wdir'], f'{args.simtype}/latin_hypercube_HR-L3000-N384',
        f'{args.lhid}')

    # Load galaxies
    pos, vel = load_galaxies_sim(source_dir, args.seed)

    # Rotate to align with CMASS
    pos, vel = rotate(pos, vel)

    # Calculate sky coordinates
    rdz = xyz_to_sky(pos, vel, cosmology.Planck15).T

    # Apply mask
    rdz = apply_mask(rdz)

    # Reweight
    rdz = reweight(rdz)

    # Save
    os.makedirs(pjoin(source_dir, 'obs'), exist_ok=True)
    np.save(pjoin(source_dir, 'obs', f'rdz{args.seed}.npy'), rdz)


if __name__ == "__main__":
    main()
