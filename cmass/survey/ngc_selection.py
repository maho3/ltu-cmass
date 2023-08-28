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

from .tools import BOSS_angular, BOSS_veto, BOSS_redshift
from ..utils import (attrdict, get_global_config, setup_logger,
                     timing_decorator, load_params)


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='apply_survey_cut')


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

    cosmo = load_params(args.lhid, glbcfg['cosmofile'])

    return attrdict(
        L=args.L, N=args.N,
        lhid=args.lhid, seed=args.seed, simtype=args.simtype,
        cosmo=cosmo
    )


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
    cosmology = nblab.cosmology.Planck15.clone(
        h=cosmo[2],
        Omega0_b=cosmo[1],
        Omega0_cdm=cosmo[0] - cosmo[1],
        m_ncdm=None,
        n_s=cosmo[3])

    # We don't need to match sigma8, because sky transform is invariant.
    # cosmology = cosmology.match(sigma8=cosmo[4])

    return nblab.transform.CartesianToSky(pos, cosmology, velocity=vel).T


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
    cfg = build_config()

    source_dir = pjoin(
        glbcfg['wdir'], f'{cfg.simtype}/L{cfg.L}-N{cfg.N}',
        f'{cfg.lhid}')

    # Load galaxies
    pos, vel = load_galaxies_sim(source_dir, cfg.seed)

    # Rotate to align with CMASS
    pos, vel = rotate(pos, vel)

    # Calculate sky coordinates
    rdz = xyz_to_sky(pos, vel, cfg.cosmo)

    # Apply mask
    rdz = apply_mask(rdz)

    # Reweight
    rdz = reweight(rdz)

    # Save
    os.makedirs(pjoin(source_dir, 'obs'), exist_ok=True)
    np.save(pjoin(source_dir, 'obs', f'rdz{cfg.seed}.npy'), rdz)


if __name__ == "__main__":
    main()
