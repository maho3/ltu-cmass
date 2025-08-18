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
from os.path import join
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from .tools.halo_models import TruncatedPowerLaw
from .tools.halo_sampling import (
    pad_3d, sample_3d,
    sample_velocities_density,
    sample_velocities_kNN,
    sample_velocities_CIC)
from ..utils import get_source_path, timing_decorator, save_cfg, clean_up
from ..nbody.tools import parse_nbody_config


def load_bias_params(bias_path, a):
    # load the bias parameters for Truncated Power Law
    with h5py.File(join(bias_path, 'bias.h5'), 'r') as f:
        key = f'{a:.6f}'
        popt = f[key]['popt'][...]
        medges = f[key]['medges'][...]
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
    filepath = join(source_path, 'transfer.h5')
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


def apply_charm(rho, fvel, charm_cfg, L, cosmo):
    """Apply CHARM, accounting for the pre-trained resolution."""

    # Load CHARM
    from charm.infer_halos_from_PM import get_model_interface
    charm_interface = get_model_interface()

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
    fvel_pad = np.pad(fvel, [(pad, pad)]*3+[(0, 0)], mode='wrap')

    # Split the inputs into batches
    batch_rho = batch_cube(rho, Nsub, Npix, Npix)
    batch_rho_pad = batch_cube(rho_pad, Nsub, Npad, Npix)
    batch_fvel = batch_cube(fvel, Nsub, Npix, Npix)
    batch_fvel_pad = batch_cube(fvel_pad, Nsub, Npad, Npix)

    # Run CHARM on each batch and append outputs
    hposs, hmasss, hvels, hconcs = [], [], [], []
    for i in range(len(batch_rho)):
        logging.info(f'Processing CHARM batch {i+1}/{len(batch_rho)}...')
        hpos, hmass, hvel, hconc = charm_interface.process_input_density(
            rho_m_zg=batch_rho[i],
            rho_m_vel_zg=np.stack([batch_fvel[i, ..., j]
                                  for j in range(3)], axis=0),
            rho_m_pad_zg=batch_rho_pad[i],
            rho_m_vel_pad_zg=np.stack(
                [batch_fvel_pad[i, ..., j] for j in range(3)], axis=0),
            cosmology_array=np.array(cosmo),
            BoxSize=Lcharm
        )
        mask = hmass > np.log10(5e12)  # charm minimum mass threshold
        hposs.append(hpos[mask])
        hmasss.append(hmass[mask])
        hvels.append(hvel[mask])
        hconcs.append(hconc[mask])

    # Shift the positions to the original box
    l = 0
    for i in range(Nsub):
        for j in range(Nsub):
            for k in range(Nsub):
                hposs[l] += np.array([i, j, k])*Lcharm
                l += 1

    # Combine the outputs
    hposs, hmasss, hvels, hconcs = map(
        np.concatenate, [hposs, hmasss, hvels, hconcs])

    # ensure periodicity
    hposs = hposs % L

    # save misc halo metadata
    meta = {'concentration': hconcs}

    return hposs, hmasss, hvels, meta


def apply_limd(rho, fvel, cfg, ppos=None, pvel=None):
    # Load bias parameters
    bcfg = deepcopy(cfg)
    bcfg.nbody.suite = bcfg.bias.halo.base_suite
    bcfg.nbody.L = bcfg.bias.halo.L
    bcfg.nbody.N = bcfg.bias.halo.N
    bias_path = get_source_path(
        bcfg.meta.wdir, bcfg.nbody.suite, cfg.sim,
        bcfg.nbody.L, bcfg.nbody.N, bcfg.nbody.lhid
    )

    logging.info('Loading bias parameters...')
    popt, medges = load_bias_params(bias_path, cfg.nbody.af)

    # Sample halo counts
    logging.info('Sampling power law...')
    hcount = sample_counts(rho, popt)

    # Sample halo positions
    logging.info('Sampling halo positions as a Poisson field...')
    hpos = sample_positions(hcount, cfg)

    # Sample halo masses
    logging.info('Sampling masses...')
    hmass = sample_masses([len(x) for x in hpos], medges)

    # Sample velocities
    logging.info('Sampling velocities...')
    hvel = sample_velocities(hpos, cfg, rho, fvel, ppos=ppos, pvel=pvel)

    # Combine the outputs of different mass bins
    def combine(x):
        x = [t for t in x if len(t) > 0]
        return np.concatenate(x, axis=0)
    hpos, hvel, hmass = map(combine, [hpos, hvel, hmass])

    return hpos, hmass, hvel


@timing_decorator
def run_snapshot(rho, fvel, a, cfg, ppos=None, pvel=None):
    if cfg.bias.halo.model == "CHARM":
        logging.info('Using CHARM model...')

        # apply CHARM model
        hpos, hmass, hvel, meta = apply_charm(
            rho,
            fvel*a/1e3,  # CHARM vel normalization (physical velocities Mm/s)
            cfg.bias.halo.config_charm,
            cfg.nbody.L, cfg.nbody.cosmo
        )
    elif cfg.bias.halo.model == "LIMD":
        logging.info('Using LIMD model...')
        hpos, hmass, hvel = apply_limd(rho, fvel, cfg, ppos, pvel)
        meta = {}
    else:
        raise NotImplementedError(
            f'Model {cfg.bias.halo.model} not implemented.')

    return hpos, hvel, hmass, meta


def load_snapshot(source_path, a):
    with h5py.File(join(source_path, 'nbody.h5'), 'r') as f:
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
    outpath = join(outdir, 'halos.h5')
    if os.path.isfile(outpath):
        os.remove(outpath)


def save_snapshot(outdir, a, hpos, hvel, hmass, **meta):

    with h5py.File(join(outdir, 'halos.h5'), 'a') as f:
        group = f.create_group(f'{a:.6f}')
        group.create_dataset('pos', data=hpos)  # comoving positions [Mpc/h]
        group.create_dataset('vel', data=hvel)  # physical velocities [km/s]
        group.create_dataset('mass', data=hmass)  # halo masses [Msun/h]

        # save other halo metadata (e.g. concentration)
        for key, val in meta.items():
            group.create_dataset(key, data=val)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'multisnapshot', 'nbody', 'bias'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )

    # Delete existing outputs
    delete_outputs(source_path)

    for i, a in enumerate(cfg.nbody.asave):
        logging.info(f'Running snapshot {i} at a={a:.6f}...')
        rho, fvel, ppos, pvel = load_snapshot(source_path, a)

        # Apply bias model
        hpos, hvel, hmass, meta = run_snapshot(
            rho, fvel, a,
            cfg, ppos, pvel
        )

        logging.info(f'Saving halo catalog to {source_path}')
        save_snapshot(source_path, a, hpos, hvel, hmass, **meta)

    save_cfg(source_path, cfg, field='bias')
    logging.info('Done!')


if __name__ == '__main__':
    main()
