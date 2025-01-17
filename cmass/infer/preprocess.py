"""
A script to train ML models on existing suites of simulations.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from tqdm import tqdm

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config
from .tools import split_experiments
from .loaders import get_cosmo, get_hod
from .loaders import load_Pk, load_lc_Pk, preprocess_Pk


def aggregate(summlist, paramlist, idlist):
    summaries = defaultdict(list)
    parameters = defaultdict(list)
    ids = defaultdict(list)
    for summ, param, id in zip(summlist, paramlist, idlist):
        for key in summ:
            summaries[key].append(summ[key])
            parameters[key].append(param)
            ids[key].append(id)
    return summaries, parameters, ids


def load_summaries(suitepath, tracer, Nmax, a=None):
    if tracer not in ['halo', 'galaxy', 'ngc_lightcone', 'sgc_lightcone',
                      'mtng_lightcone']:
        raise ValueError(f'Unknown tracer: {tracer}')

    logging.info(f'Looking for {tracer} summaries at {suitepath}')

    # get list of simulation paths
    simpaths = os.listdir(suitepath)
    simpaths.sort(key=lambda x: int(x))  # sort by lhid
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]

    # load summaries
    summlist, paramlist, idlist = [], [], []
    Ntot = 0
    for lhid in tqdm(simpaths):
        # specify paths to diagnostics
        sourcepath = join(suitepath, lhid)
        diagpath = join(sourcepath, 'diag')
        if tracer == 'galaxy':
            diagpath = join(diagpath, 'galaxies')
        elif 'lightcone' in tracer:
            diagpath = join(diagpath, f'{tracer}')
        if not os.path.isdir(diagpath):
            continue

        # for each diagnostics file
        if tracer == 'halo':
            filelist = ['halos.h5']
        else:
            filelist = os.listdir(diagpath)
        Ntot += len(filelist)
        for f in filelist:
            diagfile = join(diagpath, f)
            # load summaries  # TODO: load other summaries
            if 'lightcone' in tracer:
                summ = load_lc_Pk(diagfile)
            else:
                summ = load_Pk(diagfile, a)
            # load parameters
            if len(summ) > 0:
                params = get_cosmo(sourcepath)
                if tracer != 'halo':  # add HOD params
                    try:  # sometimes, HOD params not saved right
                        params = np.concatenate(
                            [params, get_hod(diagfile)], axis=0)
                    except (OSError, KeyError):
                        print(tracer)
                        continue
                summlist.append(summ)
                paramlist.append(params)
                idlist.append(lhid)

    summaries, parameters, ids = aggregate(summlist, paramlist, idlist)
    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} / {Ntot} {key}'
            ' summaries')
    return summaries, parameters, ids


def split_train_test(x, theta, ids, test_frac, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x, theta, ids = map(np.array, [x, theta, ids])
    unique_ids = np.unique(ids)
    np.random.shuffle(unique_ids)
    cutoff = int(len(unique_ids) * (1 - test_frac))
    ids_train, ids_test = unique_ids[:cutoff], unique_ids[cutoff:]

    train_mask = np.isin(ids, ids_train)
    test_mask = np.isin(ids, ids_test)

    x_train, x_test = x[train_mask], x[test_mask]
    theta_train, theta_test = theta[train_mask], theta[test_mask]

    return x_train, x_test, theta_train, theta_test


def run_preprocessing(summaries, parameters, ids, exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'

    # check that there's data
    for summ in exp.summary:
        if (summ not in summaries) or (len(summaries[summ]) == 0):
            logging.warning(f'No data for {exp.summary}. Skipping...')
            return

    name = '+'.join(exp.summary)
    for kmax in exp.kmax:
        logging.info(f'Running preprocessing for {name} with kmax={kmax}')
        exp_path = join(model_path, f'kmax-{kmax}')
        xs = []
        for summ in exp.summary:
            x, theta, id = summaries[summ], parameters[summ], ids[summ]
            if 'Pk0' in summ:
                x = preprocess_Pk(x, kmax, monopole=True)
            elif 'Pk' in summ:
                norm_key = summ[:-1] + '0'  # monopole (Pk0 or zPk0)
                if norm_key in summaries:
                    x = preprocess_Pk(
                        x, kmax, monopole=False, norm=summaries[norm_key])
                else:
                    raise ValueError(
                        f'Need monopole for normalization of {summ}')
            else:
                raise NotImplementedError  # TODO: implement other summaries
            xs.append(x)
        if not np.all([len(x) == len(xs[0]) for x in xs]):
            raise ValueError(
                f'Inconsistent lengths of summaries for {name}. Check that all '
                'summaries have been computed for the same simulations.')
        x = np.concatenate(xs, axis=-1)

        # split train/test
        x_train, x_test, theta_train, theta_test = split_train_test(
            x, theta, id, cfg.infer.test_frac, cfg.infer.seed)
        logging.info(f'Split: {len(x_train)} training, {len(x_test)} testing')

        # save training/test data
        logging.info(f'Saving training/test data to {exp_path}')
        os.makedirs(exp_path, exist_ok=True)
        np.save(join(exp_path, 'x_train.npy'), x_train)
        np.save(join(exp_path, 'theta_train.npy'), theta_train)
        np.save(join(exp_path, 'x_test.npy'), x_test)
        np.save(join(exp_path, 'theta_test.npy'), theta_test)


@ timing_decorator
@ hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = parse_nbody_config(cfg)
    suite_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, 0, check=False
    )[:-2]  # get to the suite directory
    model_dir = join(cfg.meta.wdir, cfg.nbody.suite, cfg.sim, 'models')
    if cfg.infer.save_dir is not None:
        model_dir = cfg.infer.save_dir
    if cfg.infer.exp_index is not None:
        cfg.infer.experiments = split_experiments(cfg.infer.experiments)
        cfg.infer.experiments = [cfg.infer.experiments[cfg.infer.exp_index]]

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    if cfg.infer.halo:
        logging.info('Running halo preprocessing...')
        summaries, parameters, ids = load_summaries(
            suite_path, 'halo', cfg.infer.Nmax, a=cfg.nbody.af)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'halo', '+'.join(exp.summary))
            run_preprocessing(summaries, parameters, ids, exp, cfg, save_path)
    else:
        logging.info('Skipping halo preprocessing...')

    if cfg.infer.galaxy:
        logging.info('Running galaxies preprocessing...')
        summaries, parameters, ids = load_summaries(
            suite_path, 'galaxy', cfg.infer.Nmax, a=cfg.nbody.af)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'galaxy', '+'.join(exp.summary))
            run_preprocessing(summaries, parameters, ids, exp, cfg, save_path)
    else:
        logging.info('Skipping galaxy preprocessing...')

    if cfg.infer.ngc_lightcone:
        logging.info('Running ngc_lightcone preprocessing...')
        summaries, parameters, ids = load_summaries(
            suite_path, 'ngc_lightcone', cfg.infer.Nmax)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'ngc_lightcone', '+'.join(exp.summary))
            run_preprocessing(summaries, parameters, ids, exp, cfg, save_path)
    else:
        logging.info('Skipping ngc_lightcone preprocessing...')

    if cfg.infer.sgc_lightcone:
        logging.info('Running sgc_lightcone preprocessing...')
        summaries, parameters, ids = load_summaries(
            suite_path, 'sgc_lightcone', cfg.infer.Nmax)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'sgc_lightcone', '+'.join(exp.summary))
            run_preprocessing(summaries, parameters, ids, exp, cfg, save_path)
    else:
        logging.info('Skipping sgc_lightcone preprocessing...')

    if cfg.infer.mtng_lightcone:
        logging.info('Running mtng_lightcone preprocessing...')
        summaries, parameters, ids = load_summaries(
            suite_path, 'mtng_lightcone', cfg.infer.Nmax)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'mtng_lightcone', '+'.join(exp.summary))
            run_preprocessing(summaries, parameters, ids, exp, cfg, save_path)
    else:
        logging.info('Skipping mtng_lightcone preprocessing...')


if __name__ == "__main__":
    main()
