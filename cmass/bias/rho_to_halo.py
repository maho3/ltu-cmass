"""
Sample halos from the density field using a bias model. Assumes
a continuous Poissonian distribution of halos by interpolating
between the grid points of the density field


Input:
    - nbody.h5
        - rho: density contrast field
        - fvel: velocity field
        - pos: particle positions [optional]
        - vel: particle velocities [optional]
    - Pre-trained TruncatedPowerLaw or CHARM (included)

Output:
    - halos.h5
        - pos: halo positions
        - vel: halo velocities
        - mass: halo masses

NOTE:
    - Works with LIMD bias models or CHARM
"""

import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa, must go before jax

import numpy as np
import logging
import hydra
import h5py
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
from os.path import join as pjoin
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from .tools.halo_models import TruncatedPowerLaw
from .tools.halo_sampling import (
    pad_3d, sample_3d,
    sample_velocities_density,
    sample_velocities_kNN,
    sample_velocities_CIC)
from ..utils import (
    get_source_path, timing_decorator)
from ..nbody.tools import parse_nbody_config


def load_bias_params(bias_path):
    # load the bias parameters for Truncated Power Law
    with h5py.File(pjoin(bias_path, 'bias.h5'), 'r') as f:
        popt = f['popt'][...]
        medges = f['medges'][...]
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
            hpos=hpos, rho=rho, L=cfg.nbody.L, Omega_m=cfg.nbody.cosmo[0],
            smooth_R=2*cfg.nbody.L/cfg.nbody.N)
    elif cfg.bias.halo.vel == 'CIC':
        # estimate halo velocities from CIC-interpolated particle velocities
        hvel = sample_velocities_CIC(
            hpos=hpos, fvel=fvel, L=cfg.nbody.L, rho=rho,
            N=cfg.nbody.N, cosmo=cfg.nbody.cosmo, z=cfg.nbody.zf)
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


def load_transfer(source_path):
    filepath = pjoin(source_path, 'transfer.h5')
    with h5py.File(filepath, 'r') as f:
        return f['rho'][...]


def batch_cube(x, Nsub, width, stride):
    """Batches a cube x into Nsub sub-cubes per side, with width and stride."""
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

    # Combine the outputs
    hposs, hmasss = np.concatenate(hposs), np.concatenate(hmasss)

    # halos are initially put on a grid, perturb their positions
    voxL = L/N
    hposs += np.random.uniform(
        low=-voxL/2,
        high=voxL/2,
        size=hposs.shape
    )  # TODO: Should this use `sample_3d`?

    # ensure periodicity
    hposs = hposs % L

    # Limit to M>1e13
    mask = hmasss > 13
    hposs, hmasss = hposs[mask], hmasss[mask]

    # conform to mass-bin format of other bias models TODO: refactor?
    hposs, hmasss = [hposs], [hmasss]

    return hposs, hmasss


def apply_limd(rho, cfg):
    # Load bias parameters
    bcfg = deepcopy(cfg)
    bcfg.nbody.suite = bcfg.bias.halo.base_suite
    bcfg.nbody.L = bcfg.bias.halo.L
    bcfg.nbody.N = bcfg.bias.halo.N
    bias_path = get_source_path(bcfg, cfg.sim)

    logging.info('Loading bias parameters...')
    popt, medges = load_bias_params(bias_path)

    # Sample halo counts
    logging.info('Sampling power law...')
    hcount = sample_counts(rho, popt)

    # Sample halo positions
    logging.info('Sampling halo positions as a Poisson field...')
    hpos = sample_positions(hcount, cfg)

    # Sample halo masses
    logging.info('Sampling masses...')
    hmass = sample_masses([len(x) for x in hpos], medges)

    return hpos, hmass


@timing_decorator
def run_snapshot(rho, fvel, cfg, rho_transfer=None, ppos=None, pvel=None):
    if cfg.bias.halo.model == "CHARM":
        logging.info('Using CHARM model...')

        # apply CHARM model
        hpos, hmass = apply_charm(
            rho, rho_transfer,
            cfg.bias.halo.config_charm,
            cfg.nbody.L, cfg.nbody.cosmo
        )
    elif cfg.bias.halo.model == "LIMD":
        logging.info('Using LIMD model...')
        hpos, hmass = apply_limd(rho, cfg)
    else:
        raise NotImplementedError(
            f'Model {cfg.bias.halo.model} not implemented.')

    logging.info('Calculating velocities...')
    hvel = sample_velocities(hpos, cfg, rho, fvel, ppos, pvel)

    logging.info('Combine...')

    def combine(x):
        x = [t for t in x if len(t) > 0]
        return np.concatenate(x, axis=0)
    hpos, hvel, hmass = map(combine, [hpos, hvel, hmass])

    return hpos, hvel, hmass


def load_snapshot(source_path, a):
    with h5py.File(pjoin(source_path, 'nbody.h5'), 'r') as f:
        group = f[f'{a:.6f}']
        rho = group['rho'][...]
        fvel = group['fvel'][...]
        if 'ppos' in group:
            ppos = group['ppos'][...]
            pvel = group['pvel'][...]
        else:
            ppos, pvel = None, None
    return rho, fvel, ppos, pvel


def delete_outputs(outdir):
    outpath = pjoin(outdir, 'halos.h5')
    if os.path.isfile(outpath):
        os.remove(outpath)


def save_snapshot(outdir, a, hpos, hvel, hmass):
    with h5py.File(pjoin(outdir, 'halos.h5'), 'a') as f:
        group = f.create_group(f'{a:.6f}')
        group.create_dataset('pos', data=hpos)  # halo positions [Mpc/h]
        group.create_dataset('vel', data=hvel)  # halo velocities [km/s]
        group.create_dataset('mass', data=hmass)  # halo masses [Msun/h]


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'sim', 'nbody', 'bias'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))
    source_path = get_source_path(cfg, cfg.sim)

    # Delete existing outputs
    delete_outputs(source_path)

    # Load transfer fn density (for CHARM)
    rho_transfer = None
    if cfg.bias.halo.model == 'CHARM':
        rho_transfer = load_transfer(source_path)

    if cfg.nbody.snapshot_mode:
        for i, a in enumerate(cfg.nbody.asave):
            logging.info(f'Running snapshot {i} at a={a:.6f}...')
            rho, fvel, ppos, pvel = load_snapshot(source_path, a)

            # Apply bias model
            hpos, hvel, hmass = run_snapshot(
                rho, fvel, cfg, rho_transfer, ppos, pvel)

            logging.info(f'Saving halo catalog to {source_path}')
            save_snapshot(source_path, a, hpos, hvel, hmass)
    else:
        # Load single snapshot
        logging.info('Loading single snapshot...')
        rho, fvel, ppos, pvel = load_snapshot(source_path, cfg.nbody.af)

        # Apply bias model
        hpos, hvel, hmass = run_snapshot(
            rho, fvel, cfg, rho_transfer, ppos, pvel)

        logging.info(f'Saving halo catalog to {source_path}')
        save_snapshot(source_path, cfg.nbody.af, hpos, hvel, hmass)

    logging.info('Done!')


if __name__ == '__main__':
    main()
