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

import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa, must go before jax

import numpy as np
import logging
import hydra
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf, open_dict
from os.path import join as pjoin
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from .tools.halo_models import TruncatedPowerLaw
from .tools.halo_sampling import (pad_3d, sample_3d,
                                  sample_velocities_density,
                                  sample_velocities_kNN,
                                  sample_velocities_CIC)
from ..utils import get_source_path, timing_decorator, load_params


def parse_config(cfg):
    with open_dict(cfg):
        cfg.nbody.cosmo = load_params(cfg.nbody.lhid, cfg.meta.cosmofile)
    return cfg


@timing_decorator
def load_nbody(source_dir):
    # density contrast
    rho = np.load(pjoin(source_dir, 'rho.npy'))
    fvel, ppos, pvel = None, None, None
    if os.path.exists(pjoin(source_dir, 'fvel.npy')):
        # velocity field [km/s]
        fvel = np.load(pjoin(source_dir, 'fvel.npy'))
    if os.path.exists(pjoin(source_dir, 'ppos.npy')):
        # particle positions [Mpc/h]
        ppos = np.load(pjoin(source_dir, 'ppos.npy'))
        # particle velocities [km/s]
        pvel = np.load(pjoin(source_dir, 'pvel.npy'))
    return rho, fvel, ppos, pvel


def load_bias_params(bias_path):
    # load the bias parameters for Truncated Power Law
    popt = np.load(pjoin(bias_path, 'halo_bias.npy'))
    medges = np.load(pjoin(bias_path, 'halo_medges.npy'))
    return popt, medges


@timing_decorator
def sample_counts(rho, popt):
    # sample the halo counts from the bias model
    law = TruncatedPowerLaw()
    return np.stack([law.predict(rho, popt[i]) for i in range(10)],
                    axis=-1)


@timing_decorator
def sample_positions(hsamp, cfg):
    # sample the halo positions from the halo count field
    hpos = []
    for i in range(10):
        Nbin = np.random.poisson(np.sum(hsamp[..., i]))
        if Nbin == 0:
            hpos.append([])
            continue
        xtrue, _, _ = sample_3d(
            hsamp[..., i], Nbin,
            cfg.nbody.L, 0, np.zeros(3))
        hpos.append(xtrue.T)
    return hpos


@timing_decorator
def sample_masses(Nsamp, medg, order=1):
    """Interpolate the mass PDF and sample it continuously."""
    mcen = (medg[1:] + medg[:-1])/2

    # don't interpolate unresolved bins at low/high mass
    mask = np.array(Nsamp) > 0
    l, r = mask.argmax(), mask.size - (mask[::-1].argmax())
    maskedg, maskcen, maskN = medg[l:r], mcen[l:r], Nsamp[l:r]

    # interpolate the mass PDF
    pdf = IUS(maskcen, np.log(maskN), k=order, ext=0)

    # sample the CDF at high resolution
    be = np.linspace(maskedg[0], maskedg[-1], 1000)
    ipdf = [0] + [quad(lambda x: np.exp(pdf(x)), be[i], be[i+1])[0]
                  for i in range(len(be)-1)]
    cdf = np.cumsum(ipdf)
    cdf /= cdf[-1]

    # invert the CDF
    invcdf = IUS(cdf, be, k=order, ext=3)

    # calculate percentiles for each mass bin edge
    perc = [0] + [quad(lambda x: np.exp(pdf(x)), medg[i], medg[i+1])[0]
                  for i in range(len(medg)-1)]
    perc = np.cumsum(perc)
    perc /= perc[-1]

    # sample the invcdf
    hmass = []
    for i in range(len(Nsamp)):
        if Nsamp[i] == 0:
            hmass.append([])
            continue
        u = np.random.uniform(low=perc[i], high=perc[i+1], size=Nsamp[i])
        m = invcdf(u)
        hmass.append(m)
    return hmass


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Build run config
    cfg = parse_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    source_path = get_source_path(cfg, cfg.sim)
    bcfg = deepcopy(cfg)
    bcfg.nbody.suite = bcfg.bias.halo.base_suite
    bias_path = get_source_path(bcfg, cfg.sim)

    logging.info('Loading bias parameters...')
    popt, medges = load_bias_params(bias_path)

    logging.info('Loading sims...')
    rho, fvel, ppos, pvel = load_nbody(source_path)

    logging.info('Sampling power law...')
    hcount = sample_counts(rho, popt)

    logging.info('Sampling halo positions as a Poisson field...')
    hpos = sample_positions(hcount, cfg)

    logging.info('Calculating velocities...')
    if cfg.bias.halo.vel == 'density':
        # estimate halo velocities from matter density field
        hvel = sample_velocities_density(hpos, rho, cfg)
    elif cfg.bias.halo.vel == 'CIC':
        # estimate halo velocities from CIC-interpolated particle velocities
        hvel = sample_velocities_CIC(hpos, cfg, fvel)
    elif cfg.bias.halo.vel == 'kNN':
        # estimate halo velocities from kNN-interpolated particle velocities
        if (ppos is None) or (pvel is None):
            raise ValueError('No particles found for kNN interpolation.')
        ppos, pvel = pad_3d(ppos, pvel, Lbox=cfg.L, Lpad=10)
        hvel = sample_velocities_kNN(hpos, ppos, pvel)
    else:
        raise NotImplementedError(
            f'Velocity type {cfg.bias.halo.vel} not implemented.')

    logging.info('Sampling masses...')
    hmass = sample_masses([len(x) for x in hpos], medges)

    logging.info('Combine...')

    def combine(x):
        x = [t for t in x if len(t) > 0]
        return np.concatenate(x, axis=0)
    hpos, hvel, hmass = map(combine, [hpos, hvel, hmass])

    logging.info('Saving cube...')
    np.save(pjoin(source_path, 'halo_pos.npy'), hpos)  # halo positions [Mpc/h]
    np.save(pjoin(source_path, 'halo_vel.npy'), hvel)  # halo velocities [km/s]
    np.save(pjoin(source_path, 'halo_mass.npy'), hmass)  # halo masses [Msun/h]

    logging.info('Done!')


if __name__ == '__main__':
    main()
