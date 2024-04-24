"""
Fits a TruncatedPowerLaw bias model to the density field vs. halo counts in
the Quijote simulations.

Requires:
    - scipy
    - astropy

Input:
    - posh: (N, 3) array of halo positions
    - mass: (N,) array of halo masses
    - rho: (N, N, N) array of density field

Output:
    - popt: (10, 3) array of bias parameters
"""

import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa, must go before jax
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'  # noqa, must go before jax
os.environ['JAX_ENABLE_X64'] = '1'  # noqa

import numpy as np
import logging
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf
import tqdm

from .tools.quijote import load_quijote_halos
from .tools.halo_models import TruncatedPowerLaw
from ..utils import get_source_path, timing_decorator


@timing_decorator
def load_halo_histogram(cfg):
    # setup metadata
    snapdir = pjoin(
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
        rho_path = pjoin(
            cfg.meta.wdir,
            cfg.fit.path_to_qrhos,
            f'{cfg.nbody.lhid}',
            f'df_m_{N}_z={z}.npy')
    else:
        source_path = get_source_path(cfg, cfg.sim)
        source_cfg = OmegaConf.load(pjoin(source_path, 'config.yaml'))

        # check that the source rho is the same as the one we want
        if source_cfg.nbody.zf != z:
            raise ValueError(
                f"Source redshift {source_cfg.nbody.zf} does not match "
                f"target redshift {z}.")

        rho_path = pjoin(source_path, 'rho.npy')
    return np.load(rho_path)


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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'sim', 'nbody', 'fit'])

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    hcounts, medges = load_halo_histogram(cfg)

    rho = load_rho(cfg)

    popt = fit_bias_params(rho, hcounts, cfg.fit.verbose, cfg.fit.attempts)

    logging.info('Saving...')
    source_path = get_source_path(cfg, cfg.sim)
    np.save(pjoin(source_path, 'halo_bias.npy'), popt)
    np.save(pjoin(source_path, 'halo_medges.npy'), medges)
    logging.info('Done!')


if __name__ == '__main__':
    main()
