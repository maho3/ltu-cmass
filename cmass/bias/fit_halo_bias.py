"""
Fits a TruncatedPowerLaw bias model to the density field vs. halo counts in
the Quijote simulations.

Input:
    - nbody.h5
        - rho: density contrast field
    - Quijote halo catalogs

Output:
    - bias.h5
        - popt: best-fit parameters for each mass bin
        - medges: mass bin edges
"""

import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa, must go before jax
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'  # noqa, must go before jax
os.environ['JAX_ENABLE_X64'] = '1'  # noqa

import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import tqdm
import h5py

from .tools.quijote import load_quijote_halos
from .tools.halo_models import TruncatedPowerLaw
from .rho_to_halo import load_snapshot
from ..utils import get_source_path, timing_decorator, save_cfg
from ..nbody.tools import parse_nbody_config


@timing_decorator
def load_halo_histogram(cfg):
    # setup metadata
    snapdir = join(
        cfg.meta.wdir,
        cfg.fit.path_to_qhalos,
        f'{cfg.nbody.lhid}')

    L = cfg.nbody.L
    N, Nm = cfg.nbody.N, cfg.fit.Nm
    mmin, mmax = cfg.fit.logMmin, cfg.fit.logMmax

    # load quijote halos
    pos_h, mass, _, _ = load_quijote_halos(snapdir, z=cfg.nbody.zf)

    # offset quijote halos by half a voxel (see issue #8)
    pos_h = (pos_h + L/(2*N)) % L

    # compute histogram
    posm = np.concatenate([pos_h, np.log10(mass)[:, None]], axis=1)
    h, edges = np.histogramdd(
        posm,
        (N,)*3+(Nm,),
        range=[(0, L)]*3+[(mmin, mmax)]
    )
    return h, edges[-1]


@timing_decorator
def load_rho(cfg):
    N, z = cfg.nbody.N, cfg.nbody.zf
    if cfg.fit.use_rho_quijote:
        rho_path = join(
            cfg.meta.wdir,
            cfg.fit.path_to_qrhos,
            f'{cfg.nbody.lhid}',
            f'df_m_{N}_z={z}.npy')
        return np.load(rho_path)
    else:
        source_path = get_source_path(
            cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
            cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
        )
        source_cfg = OmegaConf.load(join(source_path, 'config.yaml'))

        # check that the source rho is the same as the one we want
        if source_cfg.nbody.zf != z:
            raise ValueError(
                f"Source redshift {source_cfg.nbody.zf} does not match "
                f"target redshift {z}.")
        rho, _, _, _ = load_snapshot(source_cfg, cfg.nbody.af)
        return rho


def fit_mass_bin(rho, hcounts, verbose=False, attempts=5):
    law = TruncatedPowerLaw()
    return law.fit(rho.flatten(), hcounts.flatten(),
                   verbose=verbose, attempts=attempts)


@timing_decorator
def fit_bias_params(rho, hcounts, verbose=True, attempts=5):
    # fit the bias parameters for using the 1 Gpc Quijote sims
    logging.info('Fitting power law...')
    Nm = hcounts.shape[-1]
    params = []
    for i in tqdm.trange(Nm, desc='Fitting mass bins', disable=not verbose):
        pi, _ = fit_mass_bin(rho, hcounts[..., i],
                             verbose=verbose, attempts=attempts)
        params.append(pi)
    return np.stack(params, axis=0)


def save_bias(source_path, a, medges, popt):
    with h5py.File(join(source_path, 'bias.h5'), 'w') as f:
        group = f.create_group(f'{a:.6f}')
        group.create_dataset('popt', data=popt)
        group.create_dataset('medges', data=medges)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'sim', 'nbody', 'fit'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    hcounts, medges = load_halo_histogram(cfg)

    rho = load_rho(cfg)

    popt = fit_bias_params(rho, hcounts, cfg.fit.verbose, cfg.fit.attempts)

    logging.info('Saving...')
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    save_bias(source_path, cfg.nbody.af, medges, popt)
    save_cfg(source_path, cfg, field='fit')
    logging.info('Done!')


if __name__ == '__main__':
    main()
