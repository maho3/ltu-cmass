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
from ..utils import (get_source_path, timing_decorator,
                     load_params, cosmo_to_colossus)
import colossus.cosmology.cosmology as csm


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
def sample_velocities(hpos, cfg, rho=None, fvel=None, ppos=None, pvel=None):
    if cfg.bias.halo.vel == 'density':
        # estimate halo velocities from matter density field
        hvel = sample_velocities_density(
            hpos, rho, L=cfg.nbody.L, Omega_m=cfg.nbody.cosmo[0],
            smooth_R=2*cfg.nbody.L/cfg.nbody.N)
    elif cfg.bias.halo.vel == 'CIC':
        # estimate halo velocities from CIC-interpolated particle velocities
        hvel = sample_velocities_CIC(hpos, cfg, fvel, rho, ppos, pvel)
    elif cfg.bias.halo.vel == 'kNN':
        # estimate halo velocities from kNN-interpolated particle velocities
        # Not used often
        if (ppos is None) or (pvel is None):
            raise ValueError('No particles found for kNN interpolation.')
        ppos, pvel = pad_3d(ppos, pvel, Lbox=cfg.L, Lpad=10)
        hvel = sample_velocities_kNN(hpos, ppos, pvel)
    else:
        raise NotImplementedError(
            f'Velocity type {cfg.bias.halo.vel} not implemented.')
    return hvel


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


def load_IC(source_path, cpars):
    filepath = pjoin(source_path, 'rho_z50.npy')
    rhoic = np.load(filepath)

    # correct for growth factor
    cosmo = cosmo_to_colossus(cpars)
    corr = cosmo.growthFactorUnnormalized(z=99)
    corr /= cosmo.growthFactorUnnormalized(z=50)
    return rhoic*corr*0.72  # additional hard-coded factor


def batch_cube(x, Nsub, width, stride):
    """Batches a cube x into Nsub sub-cubes per side, each of size Nbatch with stride dN."""
    batches = []
    for i in range(Nsub):
        for j in range(Nsub):
            for k in range(Nsub):
                batches.append(x[i*stride:i*stride+width,
                                 j*stride:j*stride+width,
                                 k*stride:k*stride+width])
    return np.stack(batches, axis=0)


def apply_charm(rho, rho_IC, charm_cfg, L, cosmo):
    """Apply CHARM, accounting for the pre-trained resolution."""

    # Load CHARM
    from .charm.integrate_ltu_cmass import get_model_interface
    run_config_name = charm_cfg
    charm_interface = get_model_interface(run_config_name)

    # Hard-code the pre-trained CHARM configuration
    Npix = 128  # pre-trained resolution
    pad = 4  # CHARM padding
    Lcharm = 1000  # CHARM box size
    Npad = 128+2*pad  # padded resolution

    N = rho.shape[0]  # input resolution
    assert N % Npix == 0, 'Input must be divisible by Npix'  # TODO: generalize

    Nsub = N//Npix  # number of sub-boxes

    # Pad the input density field
    rho_pad = np.pad(rho, pad, mode='wrap')
    rho_IC_pad = np.pad(rho_IC, pad, mode='wrap')

    # Split the inputs into batches
    batch_rho = batch_cube(rho, Nsub, Npix, Npix)
    batch_rho_pad = batch_cube(rho_pad, Nsub, Npad, Npix)
    batch_rho_IC = batch_cube(rho_IC, Nsub, Npix, Npix)
    batch_rho_IC_pad = batch_cube(rho_IC_pad, Nsub, Npad, Npix)

    # Run CHARM on each batch and append outputs
    hposs, hmasss = [], []
    for i in range(len(batch_rho)):
        logging.info(f'Processing CHARM batch {i+1}/{len(batch_rho)}...')
        hpos, hmass = charm_interface.process_input_density(
            rho_m_zg=batch_rho[i],
            rho_m_zIC=batch_rho_IC[i],
            df_test_pad_zg=batch_rho_pad[i],
            df_test_pad_zIC=batch_rho_IC_pad[i],
            cosmology_array=np.array(cosmo),
            BoxSize=Lcharm
        )
        mask = hmass > 13
        hposs.append(hpos[mask])
        hmasss.append(hmass[mask])

    # Shift the positions to the original box
    l = 0
    for i in range(Nsub):
        for j in range(Nsub):
            for k in range(Nsub):
                hposs[l] += np.array([i, j, k])*Lcharm
                l += 1

    return np.concatenate(hposs), np.concatenate(hmasss)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'sim', 'nbody', 'bias'])

    # Build run config
    cfg = parse_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    source_path = get_source_path(cfg, cfg.sim)
    bcfg = deepcopy(cfg)
    bcfg.nbody.suite = bcfg.bias.halo.base_suite
    bcfg.nbody.L = bcfg.bias.halo.L
    bcfg.nbody.N = bcfg.bias.halo.N
    bias_path = get_source_path(bcfg, cfg.sim)

    logging.info('Loading bias parameters...')
    popt, medges = load_bias_params(bias_path)

    logging.info('Loading sims...')
    rho, fvel, ppos, pvel = load_nbody(source_path)

    if cfg.bias.halo.model == "CHARM":
        logging.info('Using CHARM model...')
        # load initial conditions at z=50, correct to z=99
        rho_IC = load_IC(source_path, cfg.nbody.cosmo)

        # apply CHARM model
        hpos, hmass = apply_charm(
            rho, rho_IC,
            cfg.bias.halo.config_charm,
            cfg.nbody.L, cfg.nbody.cosmo
        )

        # halos are initially put on a grid, perturb their positions
        voxL = cfg.nbody.L/cfg.nbody.N
        hpos += np.random.uniform(
            low=-voxL/2,
            high=voxL/2,
            size=hpos.shape
        )  # TODO: Should this use `sample_3d`?

        # ensure periodicity
        hpos = hpos % cfg.nbody.L

        # Limit to M>1e13
        mask = hmass > 13
        hpos, hmass = hpos[mask], hmass[mask]

        # conform to mass-bin format of other bias models TODO: refactor?
        hpos, hmass = [hpos], [hmass]

    else:
        logging.info('Sampling power law...')
        hcount = sample_counts(rho, popt)

        logging.info('Sampling halo positions as a Poisson field...')
        hpos = sample_positions(hcount, cfg)

        logging.info('Sampling masses...')
        hmass = sample_masses([len(x) for x in hpos], medges)

    logging.info('Calculating velocities...')
    hvel = sample_velocities(hpos, cfg, rho, fvel, ppos, pvel)

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
