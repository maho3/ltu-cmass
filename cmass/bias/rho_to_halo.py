"""
Sample halos from the density field using a bias model. Assumes
a continuous Poissonian distribution of halos by interpolating
between the grid points of the density field

Requires:
    - scipy
    - sklearn
    - jax
    - pmwd

Input:
    - rho: density field
    - ppos: particle positions
    - pvel: particle velocities
    - popt: bias parameters
    - medges: mass bin edges

Output:
    - hpos: halo positions
    - hvel: halo velocities
    - hmass: halo masses
"""

import numpy as np
import argparse
import logging
from os.path import join as pjoin
from scipy.interpolate import interp1d
from .tools.halos import (pad_3d, TruncatedPowerLaw, sample_3d)
from .tools.halos import (sample_velocities_density, sample_velocities_kNN,
                          sample_velocities_CIC)
from ..utils import (attrdict, get_global_config, setup_logger,
                     timing_decorator, load_params)


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='rho_to_halo')


def build_config():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lhid', type=int, required=True)  # which cosmology to use
    parser.add_argument(
        '--simtype', type=str, default='borg2lpt')  # which nbody sim to use
    parser.add_argument(
        '--veltype', type=str, default='density')  # method to infer velocities
    args = parser.parse_args()

    L = 3000           # length of box in Mpc/h
    N = 384            # number of grid points on one side

    cosmo = load_params(args.lhid, glbcfg['cosmofile'])

    return attrdict(
        L=L, N=N, cosmo=cosmo,
        lhid=args.lhid, simtype=args.simtype, veltype=args.veltype
    )


@timing_decorator
def load_nbody(source_dir):
    rho = np.load(pjoin(source_dir, 'rho.npy'))
    ppos = np.load(pjoin(source_dir, 'ppos.npy'))
    pvel = np.load(pjoin(source_dir, 'pvel.npy'))
    return rho, ppos, pvel


def load_bias_params(bias_path, lhid):
    # load the bias parameters for using the 1 Gpc Quijote sims
    popt = np.load(pjoin(bias_path, f'{lhid}.npy'))
    medges = np.load(pjoin(bias_path, 'medges.npy'))
    return popt, medges


@timing_decorator
def sample_counts(rho, popt):
    # sample the halo counts from the bias model
    law = TruncatedPowerLaw()
    return np.stack([law._get_mean_ngal(rho, *popt[i]) for i in range(10)],
                    axis=-1)


@timing_decorator
def sample_positions(hsamp):
    # sample the halo positions from the halo count field
    hpos = []
    for i in range(10):
        xtrue, _, _ = sample_3d(
            hsamp[..., i],
            np.sum(hsamp[..., i]).astype(int),
            3000, 0, np.zeros(3))
        hpos.append(xtrue.T)
    return hpos


@timing_decorator
def sample_masses(Nsamp, medges):
    """Linearly interpolate between different mass bins and sample."""

    # calculate the cdf from high-to-low mass
    cdf = np.array([0., *np.cumsum(Nsamp)/np.sum(Nsamp)])
    hmass = []
    for i in range(len(Nsamp)):
        x = np.random.rand(Nsamp[i])
        x = (cdf[i+1] - cdf[i])*x + cdf[i]
        m = interp1d(cdf, medges, kind='quadratic')(x)
        hmass.append(m)
    return hmass


def main():
    # Build run config
    cfg = build_config()
    logging.info(f'Running with config: {cfg}')

    bias_path = pjoin(glbcfg['wdir'], 'quijote/bias_fit/LH_n=128')
    popt, medges = load_bias_params(bias_path, cfg.lhid)

    logging.info('Loading 3 Gpc sims...')
    source_dir = pjoin(
        glbcfg['wdir'], f'{cfg.simtype}/L{cfg.L}-N{cfg.N}', f'{cfg.lhid}')
    rho, ppos, pvel = load_nbody(source_dir)

    logging.info('Sampling power law...')
    hcount = sample_counts(rho, popt)

    logging.info('Sampling halo positions as a Poisson field...')
    hpos = sample_positions(hcount)

    logging.info('Calculating velocities...')
    if cfg.veltype == 'density':
        # estimate halo velocities from matter density field
        hvel = sample_velocities_density(
            hpos, rho, cfg.L, cfg.cosmo[0], cfg.L/cfg.N)
    elif cfg.veltype == 'CIC':
        # estimate halo velocities from CIC-interpolated particle velocities
        hvel = sample_velocities_CIC(hpos, ppos, pvel, cfg.L, cfg.N, cfg.N//16)
    elif cfg.veltype == 'kNN':
        # estimate halo velocities from kNN-interpolated particle velocities
        ppos, pvel = pad_3d(ppos, pvel, Lbox=cfg.L, Lpad=10)
        hvel = sample_velocities_kNN(hpos, ppos, pvel)
    else:
        raise NotImplementedError(
            f'Velocity type {cfg.veltype} not implemented.')

    logging.info('Sampling masses...')
    hmass = sample_masses([len(x) for x in hpos], medges)

    logging.info('Combine...')
    hpos = np.concatenate(hpos, axis=0)
    hvel = np.concatenate(hvel, axis=0)
    hmass = np.concatenate(hmass, axis=0)

    logging.info('Saving cube...')
    np.save(pjoin(source_dir, 'halo_pos.npy'), hpos)
    np.save(pjoin(source_dir, 'halo_vel.npy'), hvel)
    np.save(pjoin(source_dir, 'halo_mass.npy'), hmass)

    logging.info('Done!')


if __name__ == '__main__':
    main()
