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
import matplotlib.pyplot as plt

from .tools import select_top_trials, split_experiments, load_posterior
from ..utils import timing_decorator, clean_up
from ..nbody.tools import parse_nbody_config

from ili.utils.ndes_pt import LampeEnsemble
from ili.validation import PlotSinglePosterior, PosteriorCoverage
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

# For cross-validation cases or not, asssuming we have been using optuna


def load_ensemble(exp_path, Nnets, weighted=True, plot=True,
                  cval=False, clean=False):
    """
    Load an ensemble of posteriors from an optuna study.
    """
    # Load the optuna study
    optunafile_cv = join(exp_path, 'optuna_study.db')
    storage_cv = f"sqlite:///{optunafile_cv}"

    # hack not to pass the summary/summaries argument again, already in exp_path
    study_cv = optuna.load_study(
        storage=storage_cv,
        study_name=exp_path.split("/kmin")[0].split("/")[-1])

    # Load top Nnets architectures by test log prob
    top_trials = select_top_trials(study_cv, Nnets)

    logging.info(f'Selected {len(top_trials)} nets.')

    if plot:
        log_probs = [t.value for t in top_trials]
        mcfgs = [t.user_attrs['mcfg'] for t in top_trials]
        plot_hyperparameter_dependence(log_probs, mcfgs, exp_path)

    ensemble_list = []
    for t in top_trials:
        model_path = join(exp_path, 'nets',
                          f'net-{t.number}', 'posterior.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "No retrained network found after hyperparameter optimization")
        pi = load_posterior(model_path, 'cpu')
        ensemble_list.append(pi.posteriors[0])

    if clean:   # Remove net directories that are not in top_nets
        all_net_dirs = os.listdir(join(exp_path, "nets"))
        top_net_numbers = {str(t.number) for t in top_trials}
        for n in all_net_dirs:
            # check if the folder name is net-{number}
            if n.startswith('net-'):
                net_number = n.split('net-')[-1]
                if net_number not in top_net_numbers:
                    shutil.rmtree(join(exp_path, 'nets', n))

    if weighted:
        ensemble_logprobs = [t.value for t in top_trials]
        weights = scipy.special.softmax(ensemble_logprobs)
        weights = torch.Tensor(weights)
    else:
        weights = torch.ones(len(top_trials)) / len(top_trials)

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
            if cfg.infer.include_hod and os.path.exists(filepath):
                hodprior = np.genfromtxt(filepath, delimiter=',', dtype=object)
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
            clean=cfg.infer.clean_models, cval=cfg.infer.cross_val)

        # run validation
        x_test = torch.Tensor(x_test).to(cfg.infer.device)
        theta_test = torch.Tensor(theta_test).to(cfg.infer.device)
        run_validation(posterior_ensemble, x_test, theta_test,
                       exp_path, names=names)


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
