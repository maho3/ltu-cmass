"""
A script to train ML models on existing suites of simulations.

This script leverages Optuna for hyperparameter optimization of ML
models used for posterior inference. It supports both standard training and
cross-validation strategies to find the best model architectures and training
parameters. The models are trained on preprocessed simulation data and evaluated
based on their posterior log-probability.
"""
import os
import time
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra
from os.path import join
import logging
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

from .hyperparameters import sample_hyperparameters_optuna
from .preprocess import setup_optuna
from .train import (load_preprocessed_data,
                    run_training, run_training_with_precompression,
                    evaluate_posterior)
from ..nbody.tools import parse_nbody_config
from ..utils import timing_decorator, clean_up
from .tools import split_experiments


def objective(trial, cfg: DictConfig,
              x_train, theta_train, x_val, theta_val, x_test, theta_test,
              hodprior, noiseprior,
              startidx=None,
              validation_smoothing_method='none', ema_decay=0.9):

    # Sample hyperparameters
    mcfg = sample_hyperparameters_optuna(
        trial, cfg.net, cfg.infer.embedding_net)

    # run training
    start = time.time()
    train_fn = run_training_with_precompression if cfg.infer.precompress else run_training
    posterior, histories = train_fn(
        x_train=x_train, theta_train=theta_train, x_val=x_val, theta_val=theta_val,
        out_dir=None,
        cfg=cfg, mcfg=mcfg,
        hodprior=hodprior, noiseprior=noiseprior, verbose=False,
        start_idx=startidx,
        validation_smoothing_method=validation_smoothing_method,
        ema_decay=ema_decay
    )
    end = time.time()

    # Save results to trial
    trial.set_user_attr("timing", end - start)
    trial.set_user_attr("mcfg", OmegaConf.to_container(mcfg, resolve=True))

    # evaluate the posterior and return
    return evaluate_posterior(posterior, x_test, theta_test)


def objective_cval(trial, cfg: DictConfig,
                   x_train, theta_train, x_val, theta_val, x_test, theta_test,
                   hodprior, noiseprior,
                   n_splits, ids_train, ids_val, ids_test,  # for cross-val
                   startidx=None,  # for multihead embedding
                   validation_smoothing_method='none', ema_decay=0.9):
    """
    Cross-validation strategy: Split data into train/val/test folds.
    For each fold, train on the train+val data and evaluate on the test set.
    Optuna optimizes hyperparameters based on test set performance averaged
    over all folds.
    """
    # Aggregate splits
    x_all = np.vstack((x_train, x_val, x_test))
    theta_all = np.vstack((theta_train, theta_val, theta_test))
    ids_all = np.concatenate((ids_train, ids_val, ids_test))

    # Sample hyperparameters
    mcfg = sample_hyperparameters_optuna(
        trial, cfg.net, cfg.infer.embedding_net)

    # Set up output directory and record configuration
    trial.set_user_attr("mcfg", OmegaConf.to_container(mcfg, resolve=True))

    # Handle the train/test splits for N-fold cross-validation, ensuring that we
    # split based on the ids to avoid data leakage
    gss = GroupShuffleSplit(
        n_splits=n_splits, test_size=cfg.infer.test_frac,
        random_state=9)  # random is fixed for all trials
    scores_out = np.zeros((n_splits,))
    timings = np.zeros((n_splits,))

    for K, (train_valid_idx, test_idx) in enumerate(
            gss.split(x_all, theta_all, ids_all)):
        logging.info(f'Cross-validation fold {K+1}/{n_splits}...')

        start = time.time()
        x_train_valid = x_all[train_valid_idx]
        theta_train_valid = theta_all[train_valid_idx]
        ids_train_valid = ids_all[train_valid_idx]

        x_test_fold = x_all[test_idx]
        theta_test_fold = theta_all[test_idx]

        # Train/val split for this fold
        _cv_val_frac = cfg.infer.val_frac / (1 - cfg.infer.test_frac)
        gss_inner = GroupShuffleSplit(
            n_splits=1, test_size=_cv_val_frac, random_state=1)
        train_idx, val_idx = next(
            gss_inner.split(x_train_valid, theta_train_valid, ids_train_valid))

        x_train_fold, x_val_fold = x_train_valid[train_idx], x_train_valid[val_idx]
        theta_train_fold, theta_val_fold = theta_train_valid[train_idx], theta_train_valid[val_idx]

        # run training
        train_fn = run_training_with_precompression if cfg.infer.precompress else run_training
        posterior, histories = train_fn(
            x_train=x_train_fold, theta_train=theta_train_fold,
            x_val=x_val_fold, theta_val=theta_val_fold,
            out_dir=None, cfg=cfg, mcfg=mcfg,
            hodprior=hodprior, noiseprior=noiseprior, verbose=False,
            start_idx=startidx,
            validation_smoothing_method=validation_smoothing_method,
            ema_decay=ema_decay
        )

        # Evaluate loop score
        scores_out[K] = evaluate_posterior(
            posterior, x_test_fold, theta_test_fold)
        end = time.time()
        timings[K] = end - start

    # Save results to trial
    trial.set_user_attr("timing_splits", timings.tolist())
    trial.set_user_attr("log_prob_splits", scores_out.tolist())

    return scores_out.mean()

def run_experiment(exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)

    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]

    if cfg.infer.cross_val:
        logging.info('Optuna objective uses cross-validation.')
        objective_fn = objective_cval
    else:
        logging.info('Optuna objective does not use cross-validation')
        objective_fn = objective

    validation_smoothing_method = cfg.infer.get(
        'validation_smoothing_method', 'none').lower()
    ema_decay = cfg.infer.get('ema_decay', 0.9)

    for kmin in kmin_list:
        for kmax in kmax_list:
            logging.info(
                f'Running training for {name} with {kmin} <= k <= {kmax}')
            exp_path = join(model_path, f'kmin-{kmin}_kmax-{kmax}')

            # load training/test data
            (x_train, theta_train, ids_train,
             x_val, theta_val, ids_val,
             x_test, theta_test, ids_test,
             hodprior, noiseprior,
             startidx) = load_preprocessed_data(exp_path)

            cv_args = []
            if cfg.infer.cross_val:
                cv_args = (cfg.infer.n_splits, ids_train, ids_val, ids_test)

            logging.info(
                f'Split: {len(x_train)} training, {len(x_val)} validation, '
                f'{len(x_test)} testing')

            # run hyperparameter optimization
            logging.info('Running hyperparameter optimization...')
            study = setup_optuna(
                exp_path, name, cfg.infer.n_startup_trials)
            study.optimize(
                lambda trial: objective_fn(
                    trial, cfg, x_train, theta_train,
                    x_val, theta_val, x_test, theta_test,
                    hodprior, noiseprior,
                    startidx,
                    *cv_args,
                    validation_smoothing_method=validation_smoothing_method,
                    ema_decay=ema_decay),
                n_trials=cfg.infer.n_trials,
                n_jobs=1,
                timeout=60*60*24,  # max 24 hours
                show_progress_bar=False,
                gc_after_trial=True
            )
            # NOTE: n_jobs>1 doesn't seem to speed things up much,
            # It seems processes are fighting for threads.
            # Instead, we parallelize via SLURM


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:

    cfg = parse_nbody_config(cfg)
    model_dir = join(cfg.meta.wdir, cfg.nbody.suite, cfg.sim, 'models')
    if cfg.infer.save_dir is not None:
        model_dir = cfg.infer.save_dir
    if cfg.infer.exp_index is not None:
        cfg.infer.experiments = split_experiments(cfg.infer.experiments)
        cfg.infer.experiments = [cfg.infer.experiments[cfg.infer.exp_index]]

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    tracer = cfg.infer.tracer
    logging.info(f'Running {tracer} inference...')
    for exp in cfg.infer.experiments:
        save_path = join(model_dir, tracer, '+'.join(exp.summary))
        run_experiment(exp, cfg, save_path)


if __name__ == "__main__":
    main()
