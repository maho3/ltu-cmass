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

from .tools import split_experiments, load_posterior
from ..utils import timing_decorator
from ..nbody.tools import parse_nbody_config

from ili.utils.ndes_pt import LampeEnsemble
from ili.validation import PlotSinglePosterior, PosteriorCoverage


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
        num_samples=1000, sample_method='direct',
        labels=names,
        plot_list=["coverage", "histogram", "predictions", "tarp", "logprob"],
        out_dir=out_dir,
        save_samples=True
    )
    metric(posterior, x, theta.to('cpu'))


def load_ensemble(exp_path, Nnets):
    # Load top Nnets architectures by test log prob
    net_dirs = os.listdir(join(exp_path, 'nets'))

    log_probs = []
    for n in net_dirs:
        log_prob_file = join(exp_path, 'nets', n, 'log_prob_test.txt')
        if os.path.exists(log_prob_file):
            with open(log_prob_file, 'r') as f:
                log_prob = float(f.read().strip())
            log_probs.append(log_prob)
        else:
            log_probs.append(-np.inf)

    # Remove nets that did not converge
    mask = np.isfinite(log_probs)
    net_dirs = np.array(net_dirs)[mask]
    log_probs = np.array(log_probs)[mask]

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

    ensemble = LampeEnsemble(
        posteriors=ensemble_list,
        weights=torch.ones(len(top_nets)) / len(top_nets)  # equally weighted
    )
    return ensemble


def run_experiment(exp, cfg, model_path, names):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)
    for kmax in exp.kmax:
        logging.info(f'Running inference for {name} with kmax={kmax}')
        exp_path = join(model_path, f'kmax-{kmax}')

        # load training/test data
        try:
            logging.info(f'Loading test data from {exp_path}')
            x_test = np.load(join(exp_path, 'x_test.npy'))
            theta_test = np.load(join(exp_path, 'theta_test.npy'))
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Could not find test data for {name} with kmax={kmax}'
                '. Make sure to run cmass.infer.preprocess first.'
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

    cosmonames = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']
    hodnames = [r'$\alpha$', r'$\log M_0$', r'$\log M_1$',
                r'$\log M_{\min}$', r'$\sigma_{\log M}$']  # TODO: load these from file?

    if cfg.infer.halo:
        logging.info('Running halo inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'halo', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path, names=cosmonames)
    else:
        logging.info('Skipping halo inference...')

    if cfg.infer.galaxy:
        logging.info('Running galaxies inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'galaxy', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path, names=cosmonames+hodnames)
    else:
        logging.info('Skipping galaxy inference...')

    if cfg.infer.ngc_lightcone:
        logging.info('Running ngc_lightcone inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'ngc_lightcone', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path, names=cosmonames+hodnames)
    else:
        logging.info('Skipping ngc_lightcone inference...')

    if cfg.infer.sgc_lightcone:
        logging.info('Running sgc_lightcone inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'sgc_lightcone', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path, names=cosmonames+hodnames)
    else:
        logging.info('Skipping sgc_lightcone inference...')

    if cfg.infer.mtng_lightcone:
        logging.info('Running mtng_lightcone inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'mtng_lightcone', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path, names=cosmonames+hodnames)
    else:
        logging.info('Skipping mtng_lightcone inference...')


if __name__ == "__main__":
    main()
