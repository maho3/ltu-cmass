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

import numpy as np
import argparse
import logging
from os.path import join as pjoin
import multiprocessing as mp
from functools import partial

from .tools.quijote import load_quijote_halos, TruncatedPowerLaw
from ..utils import attrdict, get_global_config, setup_logger, timing_decorator


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='fit_halo_bias')


def build_config():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lhid', type=int, required=True)  # which cosmology to use
    args = parser.parse_args()

    return attrdict(
        lhid=args.lhid
    )


@timing_decorator
def load_hhalos(snapdir):
    # load quijote halos and compute histogram
    Lbox = 1000
    xbins, mbins = 128, 10

    pos_h, mass, _, _ = load_quijote_halos(snapdir)
    posm = np.concatenate([pos_h, np.log10(mass)[:, None]], axis=1)
    h, edges = np.histogramdd(
        posm,
        (xbins,)*3+(mbins,),
        range=[(0, Lbox)]*3+[(12.8, 15.8)]
    )
    return (h, edges)


def fit_mass_bin(ind, rho, halos):
    law = TruncatedPowerLaw()
    return law.fit(rho.flatten(), halos[..., ind].flatten())


@timing_decorator
def fit_bias_params(rho, hcounts):
    # fit the bias parameters for using the 1 Gpc Quijote sims
    logging.info('Fitting power law...')
    helper = partial(fit_mass_bin, rho=rho, halos=hcounts)
    with mp.Pool(10) as pool:
        popt = np.stack(list(pool.map(helper, range(10))))
    return popt


def main():
    cfg = build_config()

    logging.info(f'Running with lhid={cfg.lhid}...')
    halo_path = pjoin(
        glbcfg['wdir'],
        f'quijote/source/Halos/latin_hypercube/{cfg.lhid}')
    hcounts, edges = load_hhalos(halo_path)
    rho_path = pjoin(
        glbcfg['wdir'],
        f'quijote/source/density_field/latin_hypercube/{cfg.lhid}',
        'df_m_128_z=0.npy')
    rho = np.load(rho_path)

    popt = fit_bias_params(rho, hcounts)

    logging.info('Saving...')
    save_path = pjoin(
        glbcfg['wdir'],
        f'quijote/bias_fit/LH_n=128/{cfg.lhid}.npy'
    )
    np.save(save_path, popt)

    logging.info('Done!')


if __name__ == '__main__':
    main()
