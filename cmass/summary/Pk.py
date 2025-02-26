"""
A script to compute the 1D power spectrum of the galaxies in a survey
geometry using nbodykit.

Input:
    - lightcone/hod{hod_seed}_aug{augmentation_seed}.h5
        - ra: right ascension
        - dec: declination
        - z: redshift

Output:
    - summary/hod{hod_seed}_aug{augmentation_seed}.h5
        - Pk: power spectrum
            - k: wavenumbers
            - p0k: power spectrum monopole
            - p2k: power spectrum quadrupole
            - p4k: power spectrum hexadecapole
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import astropy
from pypower import CatalogFFTPower

from .tools import load_lightcone, save_summary, load_randoms
from ..survey.tools import sky_to_xyz
from ..utils import get_source_path, timing_decorator, cosmo_to_astropy


@timing_decorator
def compute_Pk(
    grdz, rrdz, cosmo,
    gweights=None, rweights=None,
    Ngrid=256, kmin=0., kmax=2, dk=0.005, return_wedges=False
):
    if gweights is None:
        gweights = np.ones(len(grdz))
    if rweights is None:
        rweights = np.ones(len(rrdz))

    cosmo = cosmo_to_astropy(cosmo)

    # convert ra, dec, z to cartesian coordinates
    gpos = sky_to_xyz(grdz, cosmo)
    rpos = sky_to_xyz(rrdz, cosmo)

    # note: FKP weights are automatically included in CatalogFFTPower

    # compute the power spectra multipoles
    kedges = np.arange(kmin, kmax, dk)
    power = CatalogFFTPower(
        data_positions1=gpos, data_weights1=gweights,
        randoms_positions1=rpos, randoms_weights1=rweights,
        nmesh=Ngrid, resampler='tsc', interlacing=2,
        ells=(0, 2, 4), edges=kedges,
        position_type='pos', dtype=np.float32,
        wrap=False
    )
    if return_wedges:
        wedges = power.wedges
        k = wedges.k
        muavg = wedges.muavg
        wedge_values = [power.wedges(mu=mu, return_k=False, complex=False) for mu in muavg]
        return k, muavg, wedge_values
    else:
        poles = power.poles
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

    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    is_North = cfg.survey.is_North  # whther to use NGC or SGC mask

    # check if we are using a filter
    use_filter = hasattr(cfg, 'filter')
    if use_filter:
        logging.info(f'Using filtered obs from {cfg.filter.filter_name}...')
        grdz, gweights = load_lightcone(
            source_path,
            hod_seed=cfg.bias.hod.seed,
            aug_seed=cfg.survey.aug_seed,
            filter_name=cfg.filter.filter_name,
            is_North=is_North)
    else:
        grdz, gweights = load_lightcone(
            source_path,
            hod_seed=cfg.bias.hod.seed,
            aug_seed=cfg.survey.aug_seed,
            is_North=is_North
        )

    rrdz = load_randoms(cfg.meta.wdir)

    # fixed because we don't know true cosmo
    cosmo = astropy.cosmology.Planck18

    # compute P(k)
    if cfg.summary.wedges:
        k, mu, pwedges =  compute_Pk(
            grdz, rrdz, cosmo,
            gweights=gweights, return_wedges=True
        )
    else:
        k, p0k, p2k, p4k = compute_Pk(
            grdz, rrdz, cosmo,
            gweights=gweights
        )

    # save results
    if is_North:
        outpath = join(source_path, 'ngc_summary')
    else:
        outpath = join(outpath, 'sgc_summary')
    os.makedirs(outpath, exist_ok=True)
    outname = f'hod{cfg.bias.hod.seed:05}_aug{cfg.survey.aug_seed:05}'
    if use_filter:
        outname += f'_{cfg.filter.filter_name}'
    outname += '.h5'
    outpath = join(outpath, outname)
    logging.info(f'Saving P(k) to {outpath}...')
    if cfg.summary.wedges:
        save_summary(outpath, 'Pk_wedges', k=k, mu=mu, pwedges=pwedges)
    else:
        save_summary(outpath, 'Pk', k=k, p0k=p0k, p2k=p2k, p4k=p4k)


if __name__ == "__main__":
    main()
