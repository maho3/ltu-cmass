"""
A script to validate ML models on existing suites of simulations.
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
import matplotlib.pyplot as plt

from .tools import split_experiments, load_posterior
from ..utils import timing_decorator
from ..nbody.tools import parse_nbody_config

from ili.utils.ndes_pt import LampeEnsemble
from ili.validation import PlotSinglePosterior, PosteriorCoverage
import yaml


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


def plot_hyperparameter_dependence(log_probs, mcfgs, exp_path):
    hyperparams = ['hidden_features', 'num_transforms',
                   'fcn_width', 'fcn_depth', 'batch_size',
                   'learning_rate', 'weight_decay']
    log_scales = ['hidden_features', 'fcn_width', 'batch_size',
                  'learning_rate', 'weight_decay']

    W = 4
    H = len(hyperparams) // W + (len(hyperparams) % W > 0)
    f, axs = plt.subplots(H, W, figsize=(5*W, 5*H))
    axs = axs.flatten() if H > 1 else [axs]
    for i, hp in enumerate(hyperparams):
        values = [mcfgs[j][hp] for j in range(len(mcfgs))]
        axs[i].plot(values, log_probs, 'x', alpha=0.8)
        axs[i].set_xlabel(hp)
        axs[i].set_ylabel('Log Probability')
        if hp in log_scales:
            axs[i].set_xscale('log')
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    f.savefig(join(exp_path, 'plot_hyperparam_dependence.jpg'),
              bbox_inches='tight', dpi=200)


def load_ensemble(exp_path, Nnets, weighted=True, plot=True):
    # Load top Nnets architectures by test log prob
    net_dirs = os.listdir(join(exp_path, 'nets'))

    log_probs, mcfgs = [], []
    for n in net_dirs:
        log_prob_file = join(exp_path, 'nets', n, 'log_prob_test.txt')
        if os.path.exists(log_prob_file):
            with open(log_prob_file, 'r') as f:
                log_prob = float(f.read().strip())
            log_probs.append(log_prob)
            # Load model config from model_config.yaml
            model_config_path = join(exp_path, 'nets', n, 'model_config.yaml')
            with open(model_config_path, 'r') as f:
                mcfgs.append(yaml.safe_load(f))
        else:
            log_probs.append(-np.inf)
            mcfgs.append(None)

    # Remove nets that did not converge
    mask = np.isfinite(log_probs)
    net_dirs = np.array(net_dirs)[mask]
    log_probs = np.array(log_probs)[mask]
    mcfgs = np.array(mcfgs)[mask]

    if plot:
        plot_hyperparameter_dependence(log_probs, mcfgs, exp_path)

    logging.info(f'Found {len(log_probs)} converged nets.')
    if len(log_probs) == 0:
        raise ValueError('No converged nets found.')

    top_nets = np.argsort(log_probs)[::-1][:Nnets]
    logging.info(f'Selected nets: {[net_dirs[i] for i in top_nets]}')

    ensemble_list = []
    for i in top_nets:
        model_path = join(exp_path, 'nets', net_dirs[i], 'posterior.pkl')
        pi = load_posterior(model_path, 'cpu')
        ensemble_list.append(pi.posteriors[0])

    if weighted:
        ensemble_logprobs = log_probs[top_nets]
        weights = scipy.special.softmax(ensemble_logprobs)
        weights = torch.Tensor(weights)
    else:
        weights = torch.ones(len(top_nets)) / len(top_nets)

    ensemble = LampeEnsemble(
        posteriors=ensemble_list,
        weights=weights  # equally weighted
    )
    return ensemble


def run_experiment(exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)
    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]

    for kmin in kmin_list:
        for kmax in kmax_list:
            logging.info(
                f'Running validation for {name} with {kmin} <= k <= {kmax}')
            exp_path = join(model_path, f'kmin-{kmin}_kmax-{kmax}')

        # load test data
        try:
            logging.info(f'Loading test data from {exp_path}')
            x_test = np.load(join(exp_path, 'x_test.npy'))
            theta_test = np.load(join(exp_path, 'theta_test.npy'))

            names = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
            filepath = join(exp_path, 'hodprior.csv')
            if (not cfg.infer.only_cosmo) and os.path.exists(filepath):
                hodprior = np.genfromtxt(filepath, delimiter=',', dtype=object)
                names += hodprior[:, 0].astype('str').tolist()
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Could not find test data for {name} with kmax={kmax}.'
                'Make sure to run cmass.infer.preprocess first.'
            )
        logging.info(f'Testing on {len(x_test)} examples')

        # load trained posterior
        posterior_ensemble = load_ensemble(exp_path, cfg.infer.Nnets)

        # run validation
        x_test = torch.Tensor(x_test).to(cfg.infer.device)
        theta_test = torch.Tensor(theta_test).to(cfg.infer.device)
        run_validation(posterior_ensemble, x_test, theta_test,
                       exp_path, names=names)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
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
            logging.info(f'Skipping {tracer} validation...')
            continue

        logging.info(f'Running {tracer} validation...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, tracer, '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path)


if __name__ == "__main__":
    main()
