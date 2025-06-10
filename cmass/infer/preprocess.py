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
import optuna

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config
from .tools import split_experiments
from .loaders import (
    preprocess_Pk, preprocess_Bk, _construct_hod_prior,
    _load_single_simulation_summaries, _get_log10nbar)


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


def load_summaries(suitepath, tracer, Nmax, a=None, only_cosmo=False):
    if tracer not in ['halo', 'galaxy', 'ngc_lightcone', 'sgc_lightcone',
                      'mtng_lightcone', 'simbig_lightcone']:
        raise ValueError(f'Unknown tracer: {tracer}')

    logging.info(f'Looking for {tracer} summaries at {suitepath}')

    # get list of simulation paths
    simpaths = os.listdir(suitepath)
    simpaths.sort(key=lambda x: int(x))  # sort by lhid
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]

    # load summaries
    summlist, paramlist, idlist = [], [], []
    for lhid in tqdm(simpaths):
        sourcepath = join(suitepath, lhid)
        summs, params = _load_single_simulation_summaries(
            sourcepath, tracer, a=a, only_cosmo=only_cosmo)
        summlist += summs
        paramlist += params
        idlist += [lhid] * len(summs)

    # get parameter names
    hodprior = None
    if (tracer != 'halo') & (not only_cosmo):  # add HOD params
        example_config_file = join(suitepath, simpaths[0], 'config.yaml')
        hodprior = _construct_hod_prior(example_config_file)

    # aggregate summaries (merges all summaries into a single dict)
    summaries, parameters, ids = aggregate(summlist, paramlist, idlist)
    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} {key} summaries')
    return summaries, parameters, ids, hodprior


def split_train_val_test(x, theta, ids, val_frac, test_frac, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x, theta, ids = map(np.array, [x, theta, ids])

    # split by lhid
    unique_ids = np.unique(ids)
    np.random.shuffle(unique_ids)
    s1, s2 = int(val_frac * len(unique_ids)), int(test_frac * len(unique_ids))
    ui_val = unique_ids[:s1]
    ui_test = unique_ids[s1:s1+s2]
    ui_train = unique_ids[s1+s2:]

    # mask
    train_mask = np.isin(ids, ui_train)
    val_mask = np.isin(ids, ui_val)
    test_mask = np.isin(ids, ui_test)
    x_train, x_val, x_test = x[train_mask], x[val_mask], x[test_mask]
    theta_train, theta_val, theta_test = theta[train_mask], theta[val_mask], theta[test_mask]
    ids_train, ids_val, ids_test = ids[train_mask], ids[val_mask], ids[test_mask]

    return ((x_train, x_val, x_test), (theta_train, theta_val, theta_test),
            (ids_train, ids_val, ids_test))


def setup_optuna(exp_path, name, n_startup_trials):
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=n_startup_trials,
    )
    _ = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        storage='sqlite:///'+join(exp_path, 'optuna_study.db'),
        study_name=name,
        load_if_exists=True
    )


def run_preprocessing(summaries, parameters, ids, hodprior, exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'

    # check that there's data
    for summ in exp.summary:
        if "Eq" in summ:
            summ = summ.replace("Eq", "")
        if summ == 'nbar':  # this comes for free with any summaries
            continue
        if (summ not in summaries) or (len(summaries[summ]) == 0):
            logging.warning(f'No data for {exp.summary}. Skipping...')
            return

    name = '+'.join(exp.summary)
    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]

    for kmin in kmin_list:
        for kmax in kmax_list:
            logging.info(
                f'Running preprocessing for {name} with {kmin} <= k <= {kmax}')
            exp_path = join(model_path, f'kmin-{kmin}_kmax-{kmax}')
            xs = []

            for summ in exp.summary:
                # Handle all the different summaries
                if summ == 'nbar':
                    continue  # we handle this separately
                eq_bool = "Eq" in summ
                summ = summ.replace("Eq", "") if eq_bool else summ
                x, theta, id = summaries[summ], parameters[summ], ids[summ]
                # Preprocess the summaries
                if 'Pk0' in summ:
                    x = preprocess_Pk(x, kmax, monopole=True, kmin=kmin,
                                      correct_shot=cfg.infer.correct_shot)
                elif 'Pk' in summ:
                    norm_key = summ[:-1] + '0'  # monopole (Pk0 or zPk0)
                    if norm_key in summaries:
                        x = preprocess_Pk(
                            x, kmax, monopole=False, norm=summaries[norm_key],
                            kmin=kmin)
                    else:
                        raise ValueError(
                            f'Need monopole for normalization of {summ}')
                elif 'Bk' in summ:
                    x = preprocess_Bk(x, kmax, log=True,
                                      equilateral_only=eq_bool, kmin=kmin,
                                      correct_shot=cfg.infer.correct_shot)
                elif 'Qk' in summ:
                    x = preprocess_Bk(x, kmax, log=False,
                                      equilateral_only=eq_bool, kmin=kmin,
                                      correct_shot=cfg.infer.correct_shot)
                else:
                    raise NotImplementedError  # TODO: implement other summaries
                xs.append(x)
            if 'nbar' in exp.summary:  # add nbar
                xs.append(_get_log10nbar(summaries['Pk0']))

            if not np.all([len(x) == len(xs[0]) for x in xs]):
                raise ValueError(
                    f'Inconsistent lengths of summaries for {name}. Check that all '
                    'summaries have been computed for the same simulations.')
            x = np.concatenate(xs, axis=-1)

            # split train/test
            ((x_train, x_val, x_test), (theta_train, theta_val, theta_test),
             (ids_train, ids_val, ids_test)) = split_train_val_test(
                x, theta, id,
                cfg.infer.val_frac, cfg.infer.test_frac, cfg.infer.seed)
            logging.info(f'Split: {len(x_train)} training, '
                         f'{len(x_val)} validation, {len(x_test)} testing')

            # save training/test data
            logging.info(f'Saving training/test data to {exp_path}')
            os.makedirs(exp_path, exist_ok=True)
            np.save(join(exp_path, 'x_train.npy'), x_train)
            np.save(join(exp_path, 'x_val.npy'), x_val)
            np.save(join(exp_path, 'x_test.npy'), x_test)
            np.save(join(exp_path, 'theta_train.npy'), theta_train)
            np.save(join(exp_path, 'theta_val.npy'), theta_val)
            np.save(join(exp_path, 'theta_test.npy'), theta_test)
            np.save(join(exp_path, 'ids_train.npy'), ids_train)
            np.save(join(exp_path, 'ids_val.npy'), ids_val)
            np.save(join(exp_path, 'ids_test.npy'), ids_test)
            if hodprior is not None:
                np.savetxt(join(exp_path, 'hodprior.csv'), hodprior,
                           delimiter=',', fmt='%s')
            # np.savetxt(join(exp_path, 'param_names.txt'), names, fmt='%s')

            # initialize Optuna study (to avoid overwriting during parallelization)
            setup_optuna(exp_path, name, cfg.infer.n_startup_trials)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
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

    if cfg.infer.halo or cfg.infer.galaxy:
        logging.info(f"Training: scale factor a =  {cfg.nbody.af}")

    for tracer in ['halo', 'galaxy',
                   'ngc_lightcone', 'sgc_lightcone', 'mtng_lightcone',
                   'simbig_lightcone']:
        if not getattr(cfg.infer, tracer):
            logging.info(f'Skipping {tracer} preprocessing...')
            continue

        logging.info(f'Running {tracer} preprocessing...')
        summaries, parameters, ids, hodprior = load_summaries(
            suite_path, tracer, cfg.infer.Nmax, a=cfg.nbody.af,
            only_cosmo=cfg.infer.only_cosmo)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, tracer, '+'.join(exp.summary))
            run_preprocessing(summaries, parameters, ids,
                              hodprior, exp, cfg, save_path)


if __name__ == "__main__":
    main()
