"""
Applies a filter to the rdz data

Input:
    - lightcone/hod{hod_seed}_aug{augmentation_seed}.h5
        - ra: right ascension
        - dec: declination
        - z: redshift

Output:
    - filter/hod{hod_seed}_aug{augmentation_seed}_{filter_name}.h5
        - ra: right ascension
        - dec: declination
        - z: redshift
        - weight: filter weight
"""

import os
import logging
from os.path import join as pjoin
import hydra
import importlib
from omegaconf import DictConfig, OmegaConf, open_dict
from ..utils import (get_source_path, timing_decorator, load_params)
from ..summary.tools import load_lightcone
from ..survey.tools import save_lightcone


def parse_config(cfg):
    with open_dict(cfg):
        # Cosmology
        cfg.nbody.cosmo = load_params(cfg.nbody.lhid, cfg.meta.cosmofile)
    return cfg


def get_filter(filter_name):

    logging.info(f'Applying filter: {filter_name}')

    return importlib.import_module(f'.filter_lib.{filter_name}',
                                   package='cmass.filter').filter


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'nbody', 'bias', 'survey', 'filter'])

    # Build run config
    cfg = parse_config(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg.filter))

    source_path = get_source_path(cfg, cfg.sim)

    # load rdz
    rdz, _ = load_lightcone(
        source_path, cfg.bias.hod.seed, cfg.survey.aug_seed)

    # import the filter function
    filter = get_filter(cfg.filter.filter_name)

    # apply filter
    rdz, weight = filter(rdz, **cfg.filter.filter_args)

    # Save
    outdir = pjoin(source_path, 'filter')
    os.makedirs(outdir, exist_ok=True)
    suffix = f'_{cfg.filter.filter_name}'
    save_lightcone(
        outdir,
        ra=rdz[:, 0], dec=rdz[:, 1], z=rdz[:, 2],
        weight=weight,
        hod_seed=cfg.bias.hod.seed,
        aug_seed=cfg.survey.aug_seed,
        suffix=suffix
    )


if __name__ == "__main__":
    main()
