"""
Applies a filter to the rdz data

Input:
    - rdz: (N, 3) array of observed (ra, dec ,z)

"""

import os
import numpy as np
import logging
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from ..utils import (get_source_path, timing_decorator, load_params)
import importlib


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
    # logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg.filter))

    source_path = get_source_path(cfg, cfg.sim)

    # load rdz
    rdz = np.load(pjoin(source_path, 'obs', f'rdz{cfg.bias.hod.seed}.npy'))

    # import the filter function
    filter = get_filter(cfg.filter.filter_name)

    # apply filter
    filtered_rdz, weight = filter(rdz, **cfg.filter.filter_args)

    # Save

    os.makedirs(pjoin(source_path, 'obs', 'filtered'), exist_ok=True)
    np.save(pjoin(source_path, 'obs/filtered',
                  f'rdz{cfg.bias.hod.seed}_{cfg.filter.filter_name}.npy'),
            filtered_rdz)
    np.save(pjoin(source_path, 'obs/filtered',
                  f'rdz{cfg.bias.hod.seed}_{cfg.filter.filter_name}_weight.npy'),
            weight)


if __name__ == "__main__":
    main()
