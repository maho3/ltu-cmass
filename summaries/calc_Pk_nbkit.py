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
import argparse
import logging
from os.path import join as pjoin
from astropy.io import fits
import pandas as pd

import nbodykit.lab as nblab
from nbodykit import cosmology

from tools.BOSS_FM import BOSS_angular, BOSS_veto, BOSS_redshift, BOSS_area, \
    get_nofz
from tools.utils import get_global_config, get_logger, timing_decorator

# logger = get_logger(__name__)


@timing_decorator
def load_galaxies_obs(source_dir, seed):
    rdz = np.load(pjoin(source_dir, 'obs', f'rdz{seed}.npy'))
    return rdz


@timing_decorator
def load_randoms():
    fname = 'data/obs/random0_DR12v5_CMASS_North.fits'
    fields = ['RA', 'DEC', 'Z']
    with fits.open(fname) as hdul:
        randoms = np.array([hdul[1].data[x] for x in fields]).T
        randoms = pd.DataFrame(randoms, columns=fields)

    n_z = np.load(pjoin('data', 'obs', 'n-z_DR12v5_CMASS_North.npy'),
                  allow_pickle=True).item()
    be, hobs = n_z['be'], n_z['h']
    cutoffs = np.cumsum(hobs) / np.sum(hobs)
    w = np.diff(be[:2])[0]

    prng = np.random.uniform(size=len(randoms))
    randoms['Z'] = be[:-1][cutoffs.searchsorted(prng)]
    randoms['Z'] += w * np.random.uniform(size=len(randoms))

    # further selection functions
    mask = BOSS_angular(randoms['RA'], randoms['DEC'])
    randoms = randoms[mask]
    mask = BOSS_redshift(randoms['Z'])
    randoms = randoms[mask]
    mask = (~BOSS_veto(randoms['RA'], randoms['DEC'], verbose=True))
    randoms = randoms[mask]

    return randoms.values


@timing_decorator
def load_randoms_precomputed():
    savepath = pjoin(
        'data', 'obs', 'random0_DR12v5_CMASS_North_PRECOMPUTED.npy')
    return np.load(savepath)


def sky_to_xyz(rdz, cosmo):
    return nblab.transform.SkyToCartesian(*rdz.T, cosmo)


@timing_decorator
def compute_Pk(grdz, rrdz, cosmo, weights=None):
    if weights is None:
        weights = np.ones(len(grdz))

    P0 = 1e4
    Ngrid = 360
    dk = 0.005
    Nr = len(rrdz)
    w_r = np.ones(Nr)

    gpos = sky_to_xyz(grdz, cosmo)
    rpos = sky_to_xyz(rrdz, cosmo)

    fsky = BOSS_area() / (360.**2 / np.pi)
    ng_of_z = get_nofz(grdz[:, -1], fsky, cosmo=cosmo)
    nbar_g = ng_of_z(grdz[:, -1])
    nbar_r = ng_of_z(rrdz[:, -1])

    _gals = nblab.ArrayCatalog({
        'Position': gpos,
        'NZ': nbar_g,
        'WEIGHT': weights,  # w_g,
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


def main():
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    source_dir = pjoin(
        glbcfg['wdir'], 'borg-quijote/latin_hypercube_HR-L3000-N384',
        f'{args.lhid}')
    rdz = load_galaxies_obs(source_dir, args.seed)
    # randoms = load_randoms()
    randoms = load_randoms_precomputed()

    cosmo = cosmology.Planck15

    # compute P(k)
    k_gal, p0k_gal, p2k_gal, p4k_gal = compute_Pk(rdz, randoms, cosmo)

    # save results
    outpath = pjoin(source_dir, 'Pk')
    os.makedirs(outpath, exist_ok=True)
    outpath = pjoin(outpath, f'Pk{args.seed}.npz')
    logging.info(f'Saving P(k) to {outpath}...')
    np.savez(outpath, k_gal=k_gal, p0k_gal=p0k_gal,
             p2k_gal=p2k_gal, p4k_gal=p4k_gal)


if __name__ == "__main__":
    main()
