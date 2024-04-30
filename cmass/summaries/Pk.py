"""
A script to compute the 1D power spectrum of the galaxies in a survey
geometry using nbodykit.

Requires:
    - nbodykit
    - astropy
    - pandas

Input:
    - rdz: (N, 3) array of galaxy ra, dec, and redshifts

Output:
    - k_gal: wavenumbers
    - Pk0_gal: power spectrum monopole
    - Pk2_gal: power spectrum quadrupole
    - Pk4_gal: power spectrum hexadecapole
"""

import os
import numpy as np
import logging
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf
import astropy
from pypower import CatalogFFTPower

from .tools import load_galaxies_obs, load_randoms
from ..survey.tools import sky_to_xyz
from ..utils import get_source_path, timing_decorator, cosmo_to_astropy


@timing_decorator
def compute_Pk(
    grdz, rrdz, cosmo,
    gweights=None, rweights=None,
    Ngrid=256, kmin=0., kmax=2, dk=0.005,
):
    if gweights is None:
        gweights = np.ones(len(grdz))
    if rweights is None:
        rweights = np.ones(len(rrdz))

    if isinstance(cosmo, list):
        cosmo = cosmo_to_astropy(cosmo)

    # convert ra, dec, z to cartesian coordinates
    gpos = sky_to_xyz(grdz, cosmo)
    rpos = sky_to_xyz(rrdz, cosmo)

    # note: FKP weights are automatically included in CatalogFFTPower

    # compute the power spectra multipoles
    kedges = np.arange(kmin, kmax, dk)
    poles = CatalogFFTPower(
        data_positions1=gpos, data_weights1=gweights,
        randoms_positions1=rpos, randoms_weights1=rweights,
        nmesh=Ngrid, resampler='tsc', interlacing=2,
        ells=(0, 2, 4), edges=kedges,
        position_type='pos', dtype=np.float32,
        wrap=False
    ).poles

    k = poles.k
    p0k = poles(ell=0, complex=False, remove_shotnoise=True)
    p2k = poles(ell=2, complex=False)
    p4k = poles(ell=4, complex=False)
    return k, p0k, p2k, p4k


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    source_path = get_source_path(cfg, cfg.sim)

    # check if we are using a filter
    use_filter = hasattr(cfg, 'filter')
    if use_filter:
        logging.info(f'Using filtered obs from {cfg.filter.filter_name}...')
        grdz, gweights = load_galaxies_obs(
            source_path, cfg.bias.hod.seed, cfg.filter.filter_name)
    else:
        grdz, gweights = load_galaxies_obs(source_path, cfg.bias.hod.seed)

    rrdz = load_randoms(cfg.meta.wdir)

    # fixed because we don't know true cosmo
    cosmo = astropy.cosmology.Planck15

    # compute P(k)
    k, p0k, p2k, p4k = compute_Pk(
        grdz, rrdz, cosmo,
        gweights=gweights
    )

    # save results
    outpath = pjoin(source_path, 'Pk')
    os.makedirs(outpath, exist_ok=True)
    if not use_filter:
        outname = f'Pk{cfg.bias.hod.seed}.npz'
    else:
        outname = f'Pk{cfg.bias.hod.seed}_{cfg.filter.filter_name}.npz'
    outpath = pjoin(outpath, outname)
    logging.info(f'Saving P(k) to {outpath}...')
    np.savez(outpath, k_gal=k, p0k_gal=p0k,
             p2k_gal=p2k, p4k_gal=p4k)


if __name__ == "__main__":
    main()
