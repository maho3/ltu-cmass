"""
Preprocesses raw simulation summaries for training.

This script loads simulation summaries, applies specified transformations,
and saves the results for later use. The pipeline is configured via Hydra.

Key steps:
1. Loads summaries in parallel from a simulation suite.
2. For each experiment, concatenates summaries (e.g., Pk, Bk), applies k-space
   cuts, and optionally performs PCA dimensionality reduction.
3. Splits data into training, validation, and test sets based on simulation ID.
4. Saves the processed data, configuration, and priors to disk.
5. Initializes an Optuna study for hyperparameter optimization.
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

from ..utils import get_source_path, timing_decorator, clean_up
from ..nbody.tools import parse_nbody_config
from .tools import (split_experiments, iter_kcuts, kcut_dirname, resolve_kmax,
                    reparam_degeneracy_bounds, apply_degeneracy_reparam,
                    DEGEN_NAME_A, DEGEN_NEW_NAME_R, DEGEN_R_PRIOR_BOUNDS)
from .loaders import (
    preprocess_Pk, preprocess_Bk,
    _construct_hod_prior_from_summaries, _construct_noise_prior,
    _load_single_simulation_summaries, _get_log10nbar, _get_log10nz)


def aggregate(summlist, paramlist, idlist):
    summaries = defaultdict(list)
    parameters = defaultdict(list)
    ids = defaultdict(list)
    # Global sample position of each entry, per key. Keys are ragged: a sample
    # missing a summary (e.g. a partially-written diag file with Bk but no Pk)
    # is absent from that key but still present in others, so positions are not
    # interchangeable between keys.
    positions = defaultdict(list)
    for i, (summ, param, id) in enumerate(zip(summlist, paramlist, idlist)):
        for key in summ:
            summaries[key].append(summ[key])
            parameters[key].append(param)
            ids[key].append(id)
            positions[key].append(i)
    return summaries, parameters, ids, positions


def _load_summaries_worker(lhid, suitepath, tracer, a,
                           include_hod, include_noise,
                           subselect_cosmo=None):
    """
    Helper function to load data for a single simulation.
    """
    sourcepath = join(suitepath, lhid)
    summs, params = _load_single_simulation_summaries(
        sourcepath, tracer, a=a,
        include_hod=include_hod, include_noise=include_noise,
        subselect_cosmo=subselect_cosmo
    )
    ids = [lhid] * len(summs)
    return summs, params, ids


def load_summaries(suitepath, tracer, Nmax, a=None,
                   include_hod=False, include_noise=False,
                   subselect_cosmo=None):
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
    tasks = [(lhid, suitepath, tracer, a, include_hod, include_noise,
              subselect_cosmo)
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
            hodprior = _construct_hod_prior_from_summaries(
                join(suitepath, simpaths[0]), tracer)
        if include_noise:
            noiseprior = _construct_noise_prior(
                join(suitepath, simpaths[0]), tracer)

    # Aggregate summaries into a single dictionary
    summaries, parameters, ids, positions = aggregate(
        summlist, paramlist, idlist)
    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} {key} summaries')

    return summaries, parameters, ids, positions, hodprior, noiseprior


def split_train_val_test(x, theta, ids, val_frac, test_frac, seed=None):
    x, theta, ids = map(np.array, [x, theta, ids])

    # Assign each lhid to a split via a stable per-id hash so that adding or
    # removing lhids never changes the assignment of the remaining ones.
    unique_ids = np.unique(ids)
    # Draw a uniform value per lhid, keyed on (seed, lhid), then threshold.
    vals = np.array([
        np.random.default_rng([seed if seed is not None else 0, int(lhid)]).random()
        for lhid in unique_ids
    ])
    ui_val = unique_ids[vals < val_frac]
    ui_test = unique_ids[(vals >= val_frac) & (vals < val_frac + test_frac)]
    ui_train = unique_ids[vals >= val_frac + test_frac]

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


def _align_to_key(values, value_positions, target_positions,
                  value_key, target_key):
    """Reorder a per-sample aux array onto another key's sample ordering.

    summaries[key] only holds the samples that actually carried `key`, so two
    keys' lists are only positionally comparable when every sample carried
    both. Gathering through the global sample positions makes the alignment
    explicit and fails loudly when a sample is missing the aux value, rather
    than silently pairing an aux value with the wrong row.
    """
    lookup = dict(zip(value_positions, values))
    try:
        return np.asarray([lookup[i] for i in target_positions])
    except KeyError as e:
        raise ValueError(
            f"Cannot align '{value_key}' to '{target_key}': "
            f"{len(set(target_positions) - set(value_positions))} of "
            f"{len(target_positions)} '{target_key}' samples have no "
            f"'{value_key}' entry (sample {e} missing). This usually means "
            f"some summary files are incomplete -- check that every diag file "
            f"contains all expected datasets."
        ) from e


def run_preprocessing(summaries, parameters, ids, positions,
                      hodprior, noiseprior, exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'

    # check that there's data
    for summ in exp.summary:
        for tag in ["Eq", "Sq", "Ss", "Is", ""]:
            if tag in summ:
                summ = summ.replace(tag, "")
                break
        if summ in ['nbar', 'nz']:  # these come for free with any summaries
            continue
        if (summ not in summaries) or (len(summaries[summ]) == 0):
            logging.warning(f'No data for {exp.summary}. Skipping...')
            return

    name = '+'.join(exp.summary)

    for kmin, kmax in iter_kcuts(exp):
        logging.info(
            f'Running preprocessing for {name} with {kmin} <= k <= {kmax}')
        exp_path = join(model_path, kcut_dirname(kmin, kmax))
        xs = []
        for summ in exp.summary:
            # Handle all the different summaries
            if summ in ['nbar', 'nz']:
                continue  # we handle these separately

            base = summ
            for tag in ["Eq", "Sq", "Ss",  "Is", ""]:
                if tag in summ:
                    base = base.replace(tag, "")
                    break

            # kmax may differ between summaries (e.g. Pk to 0.6, Bk to 0.2)
            skmax = resolve_kmax(kmax, summ)

            x, theta, id = summaries[base], parameters[base], ids[base]
            base_key = base  # aux arrays below must align to this key
            # Preprocess the summaries
            if 'Pk' in summ:
                norm_key = base[:-1] + '0'  # monopole (Pk0 or zPk0)
                x = preprocess_Pk(
                    x, kmin=kmin, kmax=skmax,
                    norm=None if '0' in base else summaries[norm_key],
                    correct_shot=cfg.infer.correct_shot,
                    loglinear_start_idx=cfg.infer.loglinear_start_idx,
                )
            elif ('Bk' in summ) or ('Qk' in summ):
                norm_key = base[:-1] + '0'  # monopole (Bk0 or zBk0)
                x = preprocess_Bk(
                    x, kmin=kmin, kmax=skmax,
                    norm=None if '0' in base else summaries[norm_key],
                    mode=tag,
                    correct_shot=cfg.infer.correct_shot,  # doesn't work currently
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

        # noise out summaries that are not Pk0 or zPk0
        if cfg.infer.get('test_noised_summs', False):
            logging.info(
                "TESTING: Noise out all summaries that are not Pk0 or zPk0")
            for i, label in enumerate(labels):
                if label not in ['Pk0', 'zPk0']:
                    logging.info(f"Noise out summary: {label}")
                    start, end = startidx[i], startidx[i+1]
                    x[:, start:end] = np.random.standard_normal(
                        size=(x.shape[0], end-start)
                    ).astype(x.dtype)

        # reparameterize the eta_vb_centrals/noise_radial degeneracy into
        # polar (degen_r, degen_phi) coords, in place of theta's columns for
        # those two names
        hodprior_save = hodprior
        if cfg.infer.get('reparam_degeneracy', False):
            if not (cfg.infer.include_hod and cfg.infer.include_noise):
                raise ValueError(
                    'infer.reparam_degeneracy requires infer.include_hod '
                    'and infer.include_noise to both be True.')
            theta_names = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
            if cfg.infer.subselect_cosmo is not None:
                theta_names = [theta_names[i]
                              for i in cfg.infer.subselect_cosmo]
            theta_names += hodprior[:, 0].astype(str).tolist()
            theta_names += ['noise_radial', 'noise_transverse']
            bounds_a, bounds_b = reparam_degeneracy_bounds(
                hodprior, noiseprior)
            theta, theta_names = apply_degeneracy_reparam(
                theta, theta_names, bounds_a, bounds_b)

            # rename the matching hodprior row for the saved hodprior.csv,
            # with an assumed uniform prior on degen_r -- the actual induced
            # prior isn't derived yet (TODO)
            hodprior_save = hodprior.copy()
            row = np.where(hodprior_save[:, 0].astype(str) == DEGEN_NAME_A)[0][0]
            hodprior_save[row] = [DEGEN_NEW_NAME_R, 'uniform',
                                  *DEGEN_R_PRIOR_BOUNDS, None, None]

        # split train/test
        ((x_train, x_val, x_test), (theta_train, theta_val, theta_test),
         (ids_train, ids_val, ids_test)) = split_train_val_test(
            x, theta, id,
            cfg.infer.val_frac, cfg.infer.test_frac, cfg.infer.seed)
        logging.info(f'Split: {len(x_train)} training, '
                     f'{len(x_val)} validation, {len(x_test)} testing')

        # Create output directory
        logging.info(f'Saving training/test data to {exp_path}')
        os.makedirs(exp_path, exist_ok=True)

        # Precompress summaries
        if cfg.infer.pca_features is not None and cfg.infer.pca_features > 0:
            logging.info(
                f"Precompressing with PCA to {cfg.infer.pca_features} features")
            # Standardize the features
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)

            # PCA compress the features
            pca = PCA(n_components=cfg.infer.pca_features)
            pca.fit(x_train)
            x_train = pca.transform(x_train)
            x_val = pca.transform(x_val)
            x_test = pca.transform(x_test)
            joblib.dump((scaler, pca), join(exp_path, 'pca.pkl'))

        # save training/test data
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

        # Save split-wise aux arrays
        id_arr = np.asarray(id)
        train_mask = np.isin(id_arr, ids_train)
        val_mask = np.isin(id_arr, ids_val)
        test_mask = np.isin(id_arr, ids_test)

        # number densities
        nbar = _align_to_key(
            np.asarray(_get_log10nbar(summaries["Pk0"]))[:, -1],
            positions["Pk0"], positions[base_key], "Pk0", base_key)
        np.save(join(exp_path, "nbar_train.npy"), nbar[train_mask])
        np.save(join(exp_path, "nbar_val.npy"), nbar[val_mask])
        np.save(join(exp_path, "nbar_test.npy"), nbar[test_mask])

        if "noiseid" in summaries:
            # noise indices
            noise = _align_to_key(
                np.asarray(summaries["noiseid"]),
                positions["noiseid"], positions[base_key],
                "noiseid", base_key).reshape(-1, 1)
            np.save(join(exp_path, "noiseid_train.npy"), noise[train_mask])
            np.save(join(exp_path, "noiseid_val.npy"), noise[val_mask])
            np.save(join(exp_path, "noiseid_test.npy"), noise[test_mask])

        with open(join(exp_path, 'x_startidx.txt'), 'w') as f:
            f.write(','.join(labels) + '\n')
            f.write(','.join(map(str, startidx.tolist())) + '\n')
        if hodprior_save is not None:
            np.savetxt(join(exp_path, 'hodprior.csv'), hodprior_save,
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

    logging.info("Scale factor a =  ", cfg.nbody.af)
    # working dir where you have writing rights, ie to save preprocess splits
    wdir = cfg.meta.wdir
    # where the raw .h5 summaries are stored, only to read
    summ_dir = cfg.meta.summ_dir

    if summ_dir != wdir:
        logging.info(f"Loading from separate summary directory: {summ_dir}")

    suite_path = get_source_path(
        summ_dir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, 0, check=False
    )[:-2]  # get to the suite directory
    model_dir = join(cfg.meta.wdir, cfg.nbody.suite, cfg.sim, 'models')
    if cfg.infer.save_dir is not None:
        model_dir = cfg.infer.save_dir
    if cfg.infer.exp_index is not None:
        cfg.infer.experiments = split_experiments(cfg.infer.experiments)
        cfg.infer.experiments = [cfg.infer.experiments[cfg.infer.exp_index]]

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    if cfg.infer.include_hod and cfg.bias.hod.from_samples:
        logging.warning(
            "Inferring HOD parameters with prior from file. "
            "ENSURE PRIOR MATCHES FILE SAMPLES TO AVOID MISMATCH."
        )

    tracer = cfg.infer.tracer
    logging.info(f'Running {tracer} preprocessing...')
    if tracer in ['halo', 'galaxy']:
        logging.info(f"Training: scale factor a =  {cfg.nbody.af}")
    summaries, parameters, ids, positions, hodprior, noiseprior = load_summaries(
        suite_path, tracer, cfg.infer.Nmax, a=cfg.nbody.af,
        include_hod=cfg.infer.include_hod,
        include_noise=cfg.infer.include_noise,
        subselect_cosmo=cfg.infer.subselect_cosmo)
    for exp in cfg.infer.experiments:
        save_path = join(model_dir, tracer, '+'.join(exp.summary))
        run_preprocessing(summaries, parameters, ids, positions,
                          hodprior, noiseprior, exp, cfg, save_path)


if __name__ == "__main__":
    main()
