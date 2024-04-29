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

from .tools import get_nofz, load_galaxies_obs
from ..survey.tools import BOSS_area, gen_randoms, sky_to_xyz
from ..utils import get_source_path, timing_decorator, cosmo_to_astropy


@timing_decorator
def load_randoms(wdir):
    path = pjoin(wdir, 'obs', 'random0_DR12v5_CMASS_North_PRECOMPUTED.npy')
    if os.path.exists(path):
        return np.load(path)
    randoms = gen_randoms()
    np.save(path, randoms)
    return randoms


@timing_decorator
def compute_Pk(
    grdz, rrdz, cosmo, survey_area,
    gweights=None, rweights=None,
    P0=1e5, Ngrid=256, dk=0.005,
    kmin=0., kmax=2,
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

    # calculate FKP weights
    fsky = survey_area / (360.**2 / np.pi)
    ng_of_z = get_nofz(grdz[:, -1], fsky, cosmo=cosmo)
    nbar_g = ng_of_z(grdz[:, -1])
    nbar_r = ng_of_z(rrdz[:, -1])
    gfkp = 1./(1. + nbar_g * P0)
    rfkp = 1./(1. + nbar_r * P0)

    # total weight = completeness weight * FKP weight
    gweights *= gfkp
    rweights *= rfkp

    # compute the power spectra multipoles
    kedges = np.arange(0, 0.5, 0.005)
    poles = CatalogFFTPower(
        data_positions1=gpos, data_weights1=gweights,
        randoms_positions1=rpos, randoms_weights1=rweights,
        nmesh=Ngrid, resampler='tsc', interlacing=2,
        ells=(0, 2, 4), edges=kedges,
        position_type='pos', dtype='f4').poles

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

    survey_area = BOSS_area(cfg.meta.wdir)  # sky coverage area of BOSS survey

    # compute P(k)
    k, p0k, p2k, p4k = compute_Pk(
        grdz, rrdz, cosmo, survey_area,
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
