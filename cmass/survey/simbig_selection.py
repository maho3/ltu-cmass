"""
Ingests SGC lightcones and filters them to match the SIMBIG selection function
as defined in arxiv:2211.00660

Input:
    - sgc_lightcone/hod{hod_seed}_aug{augmentation_seed}.h5
        - ra: right ascension
        - dec: declination
        - z: redshift
        - galsnap: snapshot index
        - galidx: galaxy index

Output:
    - simbig_lightcone/hod{hod_seed}_aug{augmentation_seed}.h5
        - ra: right ascension
        - dec: declination
        - z: redshift
        - galsnap: snapshot index
        - galidx: galaxy index

NOTE:
    - This script requires hodlightcone.py to be run for the SGC lightcone.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from ..utils import get_source_path, timing_decorator, save_cfg
from ..nbody.tools import parse_nbody_config
from .tools import save_lightcone, load_lightcone
from ..bias.tools.hod import parse_hod


def _in_simbig_selection(ra, dec, z):
    assert len(ra) == len(dec) == len(z)

    # SIMBIG selection function (arxiv:2211.00660)
    ramin, ramax = -25. + 360, 28.
    decmin, decmax = - 6., np.inf
    zmin, zmax = 0.45, 0.6
    return (  # ra check accounts for wrap-around
        (((0 < ra) & (ra < ramax)) | (((ramin < ra) & (ra < 360)))) &
        (decmin < dec) & (dec < decmax) &
        (zmin < z) & (z < zmax)
    )


def _mask(x, m):
    if x is None:
        return None
    return x[m]


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'multisnapshot', 'nbody', 'bias', 'survey'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    hod_seed = cfg.bias.hod.seed  # for indexing different hod realizations
    aug_seed = cfg.survey.aug_seed  # for rotating and shuffling

    indir = join(source_path, 'sgc_lightcone')
    outdir = join(source_path, 'simbig_lightcone')

    # Load the SGC lightcone
    logging.info(f'Loading SGC lightcone from {indir}')
    ra, dec, z, galsnap, galidx, _, attrs = \
        load_lightcone(indir, hod_seed, aug_seed)
    logging.info(f'Loaded {len(ra)} galaxies from the SGC lightcone.')

    # Apply the SIMBIG selection function
    mask = _in_simbig_selection(ra, dec, z)
    ra, dec, z, galsnap, galidx = \
        map(_mask, (ra, dec, z, galsnap, galidx), (mask,) * 5)
    logging.info(
        f'After filtering, {len(ra)} galaxies remain in the SimBIG lightcone.')

    # Remove attributes which will be overwritten
    for key in ['config', 'cosmo_names', 'cosmo_params',
                'HOD_model', 'HOD_seed', 'HOD_names', 'HOD_params']:
        attrs.pop(key, None)

    # Save the filtered lightcone
    os.makedirs(outdir, exist_ok=True)
    save_lightcone(
        outdir,
        ra=ra, dec=dec, z=z,
        galsnap=galsnap,
        galidx=galidx,
        hod_seed=hod_seed,
        aug_seed=aug_seed,
        config=cfg,
        **attrs
    )
    save_cfg(source_path, cfg, field='survey')


if __name__ == "__main__":
    main()
