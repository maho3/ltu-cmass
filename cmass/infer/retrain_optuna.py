"""
A script to train ML models on existing suites of simulations.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import time

from .tools import split_experiments
from ..utils import timing_decorator, clean_up
from ..nbody.tools import parse_nbody_config

from .train import (load_preprocessed_data,
                    run_training, run_training_with_precompression,
                    evaluate_posterior, plot_training_history)

import optuna


def select_nets_retrain(exp_path, Nnets):
    '''
    Select nets from hyperparameter study with cross-validation splits, and
    retrain on the presaved train/val split NOT USED during cross-validation.
    '''

    # Load the optuna study. Note that model parameters are also stored
    # separately in exp_path/nets/net-xxx/splitn/model_config.yaml
    optunafile_cv = join(exp_path, 'optuna_study.db')
    storage_cv = f"sqlite:///{optunafile_cv}"

    # hack not to pass the summary/summaries argument again, already in exp_path
    study_cv = optuna.load_study(
        storage=storage_cv,
        study_name=exp_path.split("/kmin")[0].split("/")[-1])

    # Load top Nnets architectures by test log prob
    net_dirs = os.listdir(join(exp_path, "nets"))
    log_probs = []
    mcfgs = []
    states_mask = []

    # Safety check
    for n in net_dirs:
        if not os.path.isdir(join(exp_path, "nets", n)):
            net_dirs.remove(n)

    for n in net_dirs:
        # check optuna trial state and retrieve hyperparameters dict
        num = int(n.replace("net-", ""))
        states_mask.append(
            study_cv.trials[num].state == optuna.trial.TrialState.COMPLETE)
        mcfg = study_cv.trials[num].params
        mcfg["batch_size"] = 2**mcfg["log2_batch_size"]
        mcfgs.append(mcfg)

        netdir = join(exp_path, "nets", n)
        log_prob_file = join(netdir, 'log_prob_test.txt')
        if not os.path.exists(log_prob_file):
            log_probs.append(-np.inf)
            continue
        with open(log_prob_file, 'r') as f:
            log_splits = [float(line.strip()) for line in f.readlines()]
        log_probs.append(np.mean(log_splits))

    # The trial must also be COMPLETE
    mask = np.isfinite(log_probs) & np.array(states_mask)
    net_dirs = np.array(net_dirs)[mask]
    log_probs = np.array(log_probs)[mask]
    mcfgs = np.array(mcfgs)[mask]

    logging.info(f'Found {len(log_probs)} converged nets.')
    if len(log_probs) == 0:
        raise ValueError('No converged nets found.')

    top_nets = np.argsort(log_probs)[::-1][:Nnets]
    logging.info(f'Selected nets: {[net_dirs[i] for i in top_nets]}')

    return net_dirs[top_nets], mcfgs[top_nets]


def run_retraining_after_cval(exp, cfg, model_path):
    '''
    Retrain density estimators of interest based on Optuna study with
    cross-validation on the test set
    '''

    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)

    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]

    Nnets = cfg.infer.Nnets
    if cfg.infer.precompress:
        train_fn = run_training_with_precompression
    else:
        train_fn = run_training

    if not cfg.infer.cross_val:
        raise ValueError(
            'There is no reason to retrain network without cross-validation.')

    validation_smoothing_method = cfg.infer.get(
        'validation_smoothing_method', 'none').lower()
    ema_decay = cfg.infer.get('ema_decay', 0.9)

    for kmin in kmin_list:
        for kmax in kmax_list:
            logging.info(
                f'Running training for {name} with {kmin} <= k <= {kmax}')
            exp_path = join(model_path, f'kmin-{kmin}_kmax-{kmax}')

            # Only select a subset of networks within the hyperparameter study
            net_dirs, net_configs = select_nets_retrain(exp_path, Nnets)

            for net, config in zip(net_dirs, net_configs):

                out_dir = join(exp_path, "nets", net, "retrained")
                os.makedirs(out_dir, exist_ok=True)

                # load training/test data: we retrain on the original split
                # from cmass.infer.preprocess
                (x_train, theta_train, ids_train,
                 x_val, theta_val, ids_val,
                 x_test, theta_test, ids_test,
                 hodprior, noiseprior) = load_preprocessed_data(exp_path)

                logging.info(
                    f'Split: {len(x_train)} training, {len(x_val)} validation, '
                    f'{len(x_test)} testing')

                mcfg = OmegaConf.create(config)

                start = time.time()
                posterior, histories = train_fn(
                    x_train, theta_train, x_val, theta_val, out_dir=out_dir,
                    prior_name=cfg.infer.prior, mcfg=mcfg,
                    batch_size=None,
                    learning_rate=None,
                    stop_after_epochs=cfg.infer.stop_after_epochs,
                    val_frac=cfg.infer.val_frac,
                    weight_decay=None,
                    lr_decay_factor=None,
                    lr_patience=None,
                    backend=cfg.infer.backend, engine=cfg.infer.engine,
                    device=cfg.infer.device,
                    hodprior=hodprior, noiseprior=noiseprior, verbose=False,
                    validation_smoothing_method=validation_smoothing_method,
                    ema_decay=ema_decay
                )
                end = time.time()

            # Save the timing and metadata
            with open(join(out_dir, 'timing.txt'), 'w') as f:
                f.write(f'{end - start:.3f}')
            with open(join(out_dir, 'model_config.yaml'), 'w') as f:
                yaml.dump(OmegaConf.to_container(mcfg, resolve=True), f)

            # plot training history
            plot_training_history(histories, out_dir)

            # evaluate the posterior and save to file
            log_prob_test = evaluate_posterior(posterior, x_test, theta_test)
            with open(join(out_dir, 'log_prob_test.txt'), 'w') as f:
                f.write(f'{log_prob_test}\n')


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

    for tracer in ['halo', 'galaxy',
                   'ngc_lightcone', 'sgc_lightcone', 'mtng_lightcone',
                   'simbig_lightcone']:
        if not getattr(cfg.infer, tracer):
            logging.info(f'Skipping {tracer} inference...')
            continue

        logging.info(f'Running {tracer} inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, tracer, '+'.join(exp.summary))
            run_retraining_after_cval(exp, cfg, save_path)


if __name__ == "__main__":
    main()
