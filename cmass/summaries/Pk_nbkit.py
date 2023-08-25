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

import nbodykit.lab as nblab
from nbodykit import cosmology

from .tools import (get_nofz, sky_to_xyz, load_galaxies_obs,
                    load_randoms_precomputed)
from ..survey.tools import BOSS_area
from ..utils import attrdict, get_global_config, setup_logger, timing_decorator


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='calc_Pk_nbkit')


def build_config():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--simtype', type=str, default='borg-quijote')
    args = parser.parse_args()

    L = 3000           # length of box in Mpc/h
    N = 384            # number of grid points on one side

    return attrdict(
        L=L, N=N,
        lhid=args.lhid, seed=args.seed, simtype=args.simtype
    )


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


def main():
    cfg = build_config()
    logging.info(f'Running with config: {cfg}')

    source_dir = pjoin(
        glbcfg['wdir'], f'{cfg.simtype}/L{cfg.L}-N{cfg.N}',
        f'{cfg.lhid}')
    rdz = load_galaxies_obs(source_dir, cfg.seed)

    randoms = load_randoms_precomputed()

    cosmo = cosmology.Planck15  # fixed because we don't know true cosmology

    # compute P(k)
    k_gal, p0k_gal, p2k_gal, p4k_gal = compute_Pk(rdz, randoms, cosmo)

    # save results
    outpath = pjoin(source_dir, 'Pk')
    os.makedirs(outpath, exist_ok=True)
    outpath = pjoin(outpath, f'Pk{cfg.seed}.npz')
    logging.info(f'Saving P(k) to {outpath}...')
    np.savez(outpath, k_gal=k_gal, p0k_gal=p0k_gal,
             p2k_gal=p2k_gal, p4k_gal=p4k_gal)


if __name__ == "__main__":
    main()
