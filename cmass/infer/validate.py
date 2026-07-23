"""
A script to validate ML models on existing suites of simulations.

This script loads trained posterior inference models and performs validation 
tasks. These tasks include plotting single posterior distributions and
evaluating posterior coverage metrics. It can also clean up poorly performing
models based on the validation results.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pickle
import scipy
import shutil
import optuna.visualization.matplotlib as vis
from matplotlib import pyplot as plt

from .tools import (select_top_trials, split_experiments, load_posterior,
                    iter_kcuts, kcut_dirname, study_name_from_path)
from ..utils import timing_decorator, clean_up
from ..nbody.tools import parse_nbody_config

from ili.utils.ndes_pt import LampeEnsemble
from ili.validation import (
    PlotSinglePosterior, PlotSinglePosteriorEnsemble, PosteriorCoverage)
import yaml

import optuna


def run_validation(posterior, x, theta, out_dir, names=None):
    logging.info('Running validation...')

    # save posterior
    with open(join(out_dir, 'posterior.pkl'), "wb") as handle:
        pickle.dump(posterior, handle)

    # Plotting single posterior
    logging.info('Plotting single posterior...')
    xobs, thetaobs = x[0], theta[0]
    metric = PlotSinglePosterior(
        num_samples=1000, sample_method='direct',
        labels=names, out_dir=out_dir
    )
    metric(posterior, x_obs=xobs, theta_fid=thetaobs.to('cpu'))

    # Plotting single posterior, overplotting posteriors from each ensemble member
    metric = PlotSinglePosteriorEnsemble(
        num_samples=10_000, sample_method='direct',
        labels=names, out_dir=out_dir
    )
    metric(posterior, x_obs=xobs, theta_fid=thetaobs.to('cpu'))

    # Posterior coverage
    logging.info('Running posterior coverage...')
    metric = PosteriorCoverage(
        num_samples=2000, sample_method='direct',
        labels=names,
        plot_list=["coverage", "histogram", "predictions", "tarp", "logprob"],
        out_dir=out_dir,
        save_samples=True
    )
    metric(posterior, x, theta.to('cpu'))


def plot_optuna_diagnostics(study, exp_path):
    # trials vs loss
    ax = vis.plot_optimization_history(study)
    fig = ax.get_figure()
    fig.savefig(join(exp_path, 'optuna_history.png'), bbox_inches='tight')
    plt.close(fig)

    # hyperparams vs loss
    axs = vis.plot_slice(study)
    fig = axs[0].get_figure()
    fig.savefig(join(exp_path, 'optuna_hyperparam_dependence.png'),
                bbox_inches='tight')
    plt.close(fig)

    # parameter importance
    ax = vis.plot_param_importances(study)
    fig = ax.get_figure()
    fig.savefig(join(exp_path, 'optuna_param_importance.png'),
                bbox_inches='tight')
    plt.close(fig)

    # timeline
    ax = vis.plot_timeline(study)
    fig = ax.get_figure()
    fig.savefig(join(exp_path, 'optuna_timeline.png'), bbox_inches='tight')
    plt.close(fig)

# For cross-validation cases or not, assuming we have been using optuna


def _select_nets_from_dir(exp_path, Nnets):
    """Fallback net selection when no completed Optuna study exists (e.g. the
    experiment was trained via train.py's non-retrain path, which doesn't use
    Optuna at all). Ranks nets/net-* dirs by their saved log_prob_test.txt.
    """
    net_dir = join(exp_path, 'nets')
    net_names = [n for n in os.listdir(net_dir)
                if n.startswith('net-')] if os.path.isdir(net_dir) else []

    numbers, values = [], []
    for n in net_names:
        num = n.split('net-')[-1]
        logprob_file = join(net_dir, n, 'log_prob_test.txt')
        with open(logprob_file) as f:
            value = float(f.read().strip())
        numbers.append(num)
        values.append(value)

    order = sorted(range(len(numbers)), key=lambda i: values[i], reverse=True)
    order = order[:Nnets]
    return [numbers[i] for i in order], [values[i] for i in order]


def load_ensemble(exp_path, Nnets, weighted=True, plot=True, clean=False):
    """
    Load an ensemble of posteriors, preferring the top Nnets architectures
    from an Optuna study (cross-validated hyperparameter search). If no
    Optuna study with completed trials exists, falls back to loading
    whatever nets are present in nets/, ranked by their held-out log prob.
    """
    optunafile_cv = join(exp_path, 'optuna_study.db')
    net_numbers = net_values = None
    if os.path.exists(optunafile_cv):
        study_cv = optuna.load_study(
            storage=f"sqlite:///{optunafile_cv}",
            study_name=study_name_from_path(exp_path))
        try:
            top_trials = select_top_trials(study_cv, Nnets)
        except ValueError:
            logging.info(
                'Optuna study exists but has no completed trials; '
                'falling back to loading nets directly from nets/.')
        else:
            net_numbers = [t.number for t in top_trials]
            net_values = [t.value for t in top_trials]
            if plot:
                plot_optuna_diagnostics(study_cv, exp_path)

    if net_numbers is None:
        logging.info(
            f'No Optuna study found at {optunafile_cv}; loading nets '
            f'directly from {join(exp_path, "nets")}.')
        net_numbers, net_values = _select_nets_from_dir(exp_path, Nnets)

    logging.info(f'Selected {len(net_numbers)} nets.')

    ensemble_list = []
    valid_numbers, valid_values = [], []
    for num, val in zip(net_numbers, net_values):
        model_path = join(exp_path, 'nets', f'net-{num}', 'posterior.pkl')
        if not os.path.exists(model_path):
            logging.warning(f"Model path not found, skipping: {model_path}")
            continue
        pi = load_posterior(model_path, 'cpu')
        ensemble_list.append(pi.posteriors[0])
        valid_numbers.append(num)
        valid_values.append(val)

    if not ensemble_list:
        raise RuntimeError("No valid models found to form an ensemble.")

    if clean:   # Remove net directories that are not in top_nets
        all_net_dirs = os.listdir(join(exp_path, "nets"))
        top_net_numbers = {str(num) for num in valid_numbers}
        for n in all_net_dirs:
            # check if the folder name is net-{number}
            if n.startswith('net-'):
                net_number = n.split('net-')[-1]
                if net_number not in top_net_numbers:
                    shutil.rmtree(join(exp_path, 'nets', n))

    if weighted:
        weights = torch.Tensor(scipy.special.softmax(valid_values))
    else:
        weights = torch.ones(len(ensemble_list)) / len(ensemble_list)

    ensemble = LampeEnsemble(
        posteriors=ensemble_list,
        weights=weights
    )
    return ensemble


def run_experiment(exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)

    for kmin, kmax in iter_kcuts(exp):
        logging.info(
            f'Running validation for {name} with {kmin} <= k <= {kmax}')
        exp_path = join(model_path, kcut_dirname(kmin, kmax))

        # load test data
        try:
            if cfg.infer.testing.suite is None:
                logging.info(f'Loading test data from {exp_path}')
                x_test = np.load(join(exp_path, 'x_test.npy'))
                theta_test = np.load(join(exp_path, 'theta_test.npy'))
                out_path = exp_path
            else:
                test_path = join(
                    cfg.meta.wdir,
                    cfg.infer.testing.suite, cfg.infer.testing.sim,
                    'models', cfg.infer.tracer, name,
                    kcut_dirname(kmin, kmax))
                logging.info(
                    f'Loading external test data from {test_path}')
                x_test = np.load(join(test_path, 'x_test.npy'))
                theta_test = np.load(
                    join(test_path, 'theta_test.npy'))

                out_path = join(exp_path, 'testing',
                                f'{cfg.infer.testing.suite}_{cfg.infer.testing.sim}')
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

            names = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
            filepath = join(exp_path, 'hodprior.csv')
            if cfg.infer.subselect_cosmo is not None:
                names = [names[i] for i in cfg.infer.subselect_cosmo]
            if cfg.infer.include_hod and os.path.exists(filepath):
                hodprior = np.genfromtxt(
                    filepath, delimiter=',', dtype=object)
                names += hodprior[:, 0].astype('str').tolist()
            if cfg.infer.include_noise:
                names += ['noise_radial', 'noise_transverse']
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Could not find test data for {name} with kmax={kmax}.'
                'Make sure to run cmass.infer.preprocess first.'
            )
        logging.info(f'Testing on {len(x_test)} examples')

        # load trained posterior
        posterior_ensemble = load_ensemble(
            exp_path, cfg.infer.Nnets,
            clean=cfg.infer.clean_models)

        # run validation
        x_test = torch.Tensor(x_test).to(cfg.infer.device)
        theta_test = torch.Tensor(theta_test).to(cfg.infer.device)
        run_validation(posterior_ensemble, x_test, theta_test,
                       out_path, names=names)


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
    logging.info(f'Running {tracer} validation...')
    for exp in cfg.infer.experiments:
        save_path = join(model_dir, tracer, '+'.join(exp.summary))
        run_experiment(exp, cfg, save_path)


if __name__ == "__main__":
    main()
