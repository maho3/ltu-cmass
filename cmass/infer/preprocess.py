"""
A script to train ML models on existing suites of simulations.
"""

import os
import numpy as np
import logging
from os.path import join, isfile
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from tqdm import tqdm
import optuna
import multiprocessing

from ..utils import get_source_path, timing_decorator, clean_up
from ..nbody.tools import parse_nbody_config
from .tools import split_experiments
from .loaders import (
    preprocess_Pk, preprocess_Bk, _construct_hod_prior, _construct_noise_prior,
    _load_single_simulation_summaries, _get_log10nbar, _get_log10nz)


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


def _load_summaries_worker(lhid, suitepath, tracer, a,
                           include_hod, include_noise):
    """
    Helper function to load data for a single simulation.
    """
    sourcepath = join(suitepath, lhid)
    summs, params = _load_single_simulation_summaries(
        sourcepath, tracer, a=a,
        include_hod=include_hod, include_noise=include_noise
    )
    ids = [lhid] * len(summs)
    return summs, params, ids


def load_summaries(suitepath, tracer, Nmax, a=None,
                   include_hod=False, include_noise=False):
    """
    Loads summaries from a suite of simulations in parallel.
    """
    if tracer not in ['halo', 'galaxy', 'ngc_lightcone', 'sgc_lightcone',
                      'mtng_lightcone', 'simbig_lightcone']:
        raise ValueError(f'Unknown tracer: {tracer}')

    logging.info(f'Looking for {tracer} summaries at {suitepath}')

    simpaths = os.listdir(suitepath)
    simpaths.sort(key=lambda x: int(x))
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]

    # Create a list of arguments for each worker task
    tasks = [(lhid, suitepath, tracer, a, include_hod, include_noise)
             for lhid in simpaths]

    # Use available CPUs, but no more than 16
    num_processes = min(os.cpu_count(), 16)

    # Load summaries in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        async_results = [
            pool.apply_async(_load_summaries_worker, args=task) for task in tasks
        ]
        results = [res.get() for res in tqdm(async_results)]

    # Unpack the parallel results into flat lists
    summlist, paramlist, idlist = [], [], []
    for s_chunk, p_chunk, id_chunk in results:
        summlist.extend(s_chunk)
        paramlist.extend(p_chunk)
        idlist.extend(id_chunk)

    # Get and save hod/noise priors from the first simulation
    hodprior, noiseprior = None, None
    if simpaths:
        if (tracer != 'halo') and include_hod:
            example_config_file = join(suitepath, simpaths[0], 'config.yaml')
            hodprior = _construct_hod_prior(example_config_file)
        if include_noise:
            noiseprior = _construct_noise_prior(
                join(suitepath, simpaths[0]), tracer)

    # Aggregate summaries into a single dictionary
    summaries, parameters, ids = aggregate(summlist, paramlist, idlist)
    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} {key} summaries')

    return summaries, parameters, ids, hodprior, noiseprior


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
        multivariate=True,
        constant_liar=True,
    )
    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        storage='sqlite:///'+join(exp_path, 'optuna_study.db'),
        study_name=name,
        load_if_exists=True
    )
    return study


def run_preprocessing(summaries, parameters, ids, hodprior, noiseprior,
                      exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'

    # check that there's data
    for summ in exp.summary:
        for tag in ["Eq", "Sq", "Ss", "Is"]:
            if tag in summ:
                summ = summ.replace(tag, "")
        if summ in ['nbar', 'nz']:  # these come for free with any summaries
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
                if summ in ['nbar', 'nz']:
                    continue  # we handle these separately

                base = summ
                for tag in ["Eq", "Sq", "Ss",  "Is"]:
                    if tag in summ:
                        base = base.replace(tag, "")
                        break

                x, theta, id = summaries[base], parameters[base], ids[base]
                # Preprocess the summaries
                if 'Pk' in summ:
                    norm_key = base[:-1] + '0'  # monopole (Pk0 or zPk0)
                    x = preprocess_Pk(
                        x, kmin=kmin, kmax=kmax,
                        norm=None if '0' in base else summaries[norm_key],
                        correct_shot=cfg.infer.correct_shot
                    )
                elif ('Bk' in summ) or ('Qk' in summ):
                    norm_key = base[:-1] + '0'  # monopole (Bk0 or zBk0)
                    x = preprocess_Bk(
                        x, kmin=kmin, kmax=kmax,
                        norm=None if '0' in base else summaries[norm_key],
                        mode=tag,
                        correct_shot=cfg.infer.correct_shot  # doesn't work currently
                    )
                else:
                    raise NotImplementedError  # TODO: implement other summaries
                xs.append((summ, x))
            if 'nz' in exp.summary:  # add n(z)
                xs.append(('nz', _get_log10nz(summaries['Pk0'])))
            if 'nbar' in exp.summary:  # add nbar
                xs.append(('nbar', _get_log10nbar(summaries['Pk0'])))

            labels, xs = zip(*xs)
            if not np.all([len(x) == len(xs[0]) for x in xs]):
                raise ValueError(
                    f'Inconsistent lengths of summaries for {name}. Check that all '
                    'summaries have been computed for the same simulations.')
            startidx = np.cumsum([0] + [x.shape[1] for x in xs])
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
            with open(join(exp_path, 'config.yaml'), 'w') as f:
                OmegaConf.save(cfg, f)
            np.save(join(exp_path, 'x_train.npy'), x_train)
            np.save(join(exp_path, 'x_val.npy'), x_val)
            np.save(join(exp_path, 'x_test.npy'), x_test)
            np.save(join(exp_path, 'theta_train.npy'), theta_train)
            np.save(join(exp_path, 'theta_val.npy'), theta_val)
            np.save(join(exp_path, 'theta_test.npy'), theta_test)
            np.save(join(exp_path, 'ids_train.npy'), ids_train)
            np.save(join(exp_path, 'ids_val.npy'), ids_val)
            np.save(join(exp_path, 'ids_test.npy'), ids_test)
            with open(join(exp_path, 'x_startidx.txt'), 'w') as f:
                f.write(','.join(labels) + '\n')
                f.write(','.join(map(str, startidx.tolist())) + '\n')
            if hodprior is not None:
                np.savetxt(join(exp_path, 'hodprior.csv'), hodprior,
                           delimiter=',', fmt='%s')
            if noiseprior is not None:
                with open(join(exp_path, 'noiseprior.yaml'), 'w') as f:
                    OmegaConf.save(noiseprior, f)
            # np.savetxt(join(exp_path, 'param_names.txt'), names, fmt='%s')

            # initialize Optuna study (to avoid overwriting during parallelization)
            if not isfile(join(exp_path, 'optuna_study.db')):
                _ = setup_optuna(exp_path, name, cfg.infer.n_startup_trials)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
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

    if cfg.infer.include_hod and cfg.bias.hod.from_samples:
        logging.warning(
            "Inferring HOD parameters with prior from file. "
            "ENSURE PRIOR MATCHES FILE SAMPLES TO AVOID MISMATCH."
        )

    for tracer in ['halo', 'galaxy',
                   'ngc_lightcone', 'sgc_lightcone', 'mtng_lightcone',
                   'simbig_lightcone']:
        if not getattr(cfg.infer, tracer):
            logging.info(f'Skipping {tracer} preprocessing...')
            continue

        logging.info(f'Running {tracer} preprocessing...')
        summaries, parameters, ids, hodprior, noiseprior = load_summaries(
            suite_path, tracer, cfg.infer.Nmax, a=cfg.nbody.af,
            include_hod=cfg.infer.include_hod,
            include_noise=cfg.infer.include_noise)
        for exp in cfg.infer.experiments:
            #save_path = join(model_dir, tracer, '+'.join(exp.summary))
            save_path = join(model_dir, tracer, cfg.sim, '+'.join(exp.summary)) # sim to compare pinocchio, fastpm...
            run_preprocessing(summaries, parameters, ids,
                              hodprior, noiseprior, exp, cfg, save_path)


if __name__ == "__main__":
    main()
