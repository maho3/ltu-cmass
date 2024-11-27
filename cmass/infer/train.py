"""
A script to train ML models on existing suites of simulations.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py
from collections import defaultdict

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config
from ..bias.apply_hod import parse_hod

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.embedding import FCN
device='cpu'


def get_cosmo(source_path):
    cfg = OmegaConf.load(join(source_path, 'config.yaml'))
    return np.array(cfg.nbody.cosmo)

def get_halo_Pk(source_path, a):
    diag_file = join(source_path, 'diag', 'halos.h5')
    if not os.path.exists(diag_file):
        return {}
    
    summ = {}
    with h5py.File(diag_file, 'r') as f:
        for key in ['Pk_k', 'Pk', 'zPk_k', 'zPk']:
            if key in f[a]:  # save summary if its available
                summ[key] = f[a][key][:]
    summ['cosmo'] = get_cosmo(source_path)
    return summ

def load_halo_summaries(suitepath, a):
    logging.info(f'Looking for halo summaries at {suitepath}')

    simpaths = os.listdir(suitepath)
    summaries, parameters, meta = defaultdict(list), defaultdict(list), defaultdict(list)
    for lhid in simpaths:
        summ = get_halo_Pk(join(suitepath, lhid), a)
        for key in ['Pk', 'zPk']:
            if key in summ:
                summaries[key].append(summ[key])
                parameters[key].append(summ['cosmo'])
                meta[key].append(summ[key+'_k'])
    summaries = {key: np.array(val) for key, val in summaries.items()}
    parameters = {key: np.array(val) for key, val in parameters.items()}
    meta = {key: np.array(val) for key, val in meta.items()}

    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} / {len(simpaths)} {key}'
            ' summaries')  
    return summaries, parameters, meta



@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    cfg = parse_nbody_config(cfg)
    suite_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, 0, check=False
    )[:-2]  # remove the last directory

    if cfg.infer.halo:
        summaries, parameters, meta = load_halo_summaries(suite_path, cfg.infer.a)
        


if __name__ == "__main__":
    main()
