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

import nbodykit.lab as nblab
from nbodykit import cosmology

from .tools import get_nofz, sky_to_xyz, load_galaxies_obs
from ..survey.tools import BOSS_area, gen_randoms
from ..utils import get_source_path, timing_decorator


@timing_decorator
def load_randoms(wdir):
    path = pjoin(wdir, 'obs', 'random0_DR12v5_CMASS_North_PRECOMPUTED.npy')
    if os.path.exists(path):
        return np.load(path)
    randoms = gen_randoms()
    np.save(path, randoms)
    return randoms


@timing_decorator
def compute_Pk(grdz, rrdz, cosmo, area, weights=None):
    if weights is None:
        weights = np.ones(len(grdz))

    P0 = 1e4
    Ngrid = 360
    dk = 0.005
    Nr = len(rrdz)
    w_r = np.ones(Nr)

    gpos = sky_to_xyz(grdz, cosmo)
    rpos = sky_to_xyz(rrdz, cosmo)

    fsky = area / (360.**2 / np.pi)
    ng_of_z = get_nofz(grdz[:, -1], fsky, cosmo=cosmo)
    nbar_g = ng_of_z(grdz[:, -1])
    nbar_r = ng_of_z(rrdz[:, -1])

    _gals = nblab.ArrayCatalog({
        'Position': gpos,
        'NZ': nbar_g,
        'WEIGHT': weights,
        'WEIGHT_FKP': 1./(1. + nbar_g * P0)
    })

    _rands = nblab.ArrayCatalog({
        'Position': rpos,
        'NZ': nbar_r,
        'WEIGHT': w_r,
        'WEIGHT_FKP': 1./(1. + nbar_r * P0)
    })

    fkp = nblab.FKPCatalog(_gals, _rands)
    mesh = fkp.to_mesh(Nmesh=Ngrid, nbar='NZ', fkp_weight='WEIGHT_FKP',
                       comp_weight='WEIGHT', window='tsc')

    # compute the multipoles
    r = nblab.ConvolvedFFTPower(mesh, poles=[0, 2, 4], dk=dk, kmin=0.)

    k_gal = r.poles['k']
    p0k_gal = r.poles['power_0'].real - r.attrs['shotnoise']
    p2k_gal = r.poles['power_2'].real
    p4k_gal = r.poles['power_4'].real
    return k_gal, p0k_gal, p2k_gal, p4k_gal


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'nbody', 'bias'])

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    source_path = get_source_path(cfg, cfg.sim)

    # check if we are using a filter
    use_filter = hasattr(cfg, 'filter')
    if use_filter:
        rdz, weights = load_galaxies_obs(
            source_path, cfg.bias.hod.seed, cfg.filter.filter_name)
    else:
        rdz, weights = load_galaxies_obs(source_path, cfg.bias.hod.seed)

    randoms = load_randoms(cfg.meta.wdir)

    cosmo = cosmology.Planck15  # fixed because we don't know true cosmology

    area = BOSS_area(cfg.meta.wdir)  # sky coverage area of BOSS survey

    # compute P(k)
    k_gal, p0k_gal, p2k_gal, p4k_gal = compute_Pk(
        rdz, randoms, cosmo, area, weights=weights)

    # save results
    outpath = pjoin(source_path, 'Pk')
    os.makedirs(outpath, exist_ok=True)
    if not use_filter:
        outname = f'Pk{cfg.bias.hod.seed}.npz'
    else:
        outname = f'Pk{cfg.bias.hod.seed}_{cfg.filter.filter_name}.npz'
    outpath = pjoin(outpath, outname)
    logging.info(f'Saving P(k) to {outpath}...')
    np.savez(outpath, k_gal=k_gal, p0k_gal=p0k_gal,
             p2k_gal=p2k_gal, p4k_gal=p4k_gal)


if __name__ == "__main__":
    main()
