"""
A script to train ML models on existing suites of simulations.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from ..utils import timing_decorator
from ..nbody.tools import parse_nbody_config

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.embedding import FCN

import matplotlib.pyplot as plt


def split_experiments(exp_cfg):
    new_exps = []
    for exp in exp_cfg:
        for kmax in exp.kmax:
            new_exp = exp.copy()
            new_exp.kmax = [kmax]
            new_exps.append(new_exp)
    return new_exps


def run_inference(x, theta, cfg, out_dir):
    loader = NumpyLoader(x=x, theta=theta)

    # select the network configuration
    mcfg = cfg.net[cfg.infer.net_index]
    logging.info(f'Using network architecture: {mcfg}')

    # define a prior
    if cfg.infer.prior.lower() == 'uniform':
        prior = ili.utils.Uniform(
            low=theta.min(axis=0),
            high=theta.max(axis=0),
            device=cfg.infer.device)
    else:
        raise NotImplementedError

    if mcfg.fcn_depth == 0:
        embedding = nn.Identity()
    else:
        embedding = FCN(
            n_hidden=[mcfg.fcn_width]*mcfg.fcn_depth,
            act_fn='ReLU'
        )

    # instantiate your neural networks to be used as an ensemble
    if cfg.infer.backend == 'lampe':
        net_loader = ili.utils.load_nde_lampe
        extra_kwargs = {}
    elif cfg.infer.backend == 'sbi':
        net_loader = ili.utils.load_nde_sbi
        extra_kwargs = {'engine': cfg.infer.engine}
    else:
        raise NotImplementedError
    kwargs = {k: v for k, v in mcfg.items() if k in [
        'model', 'hidden_features', 'num_transforms', 'num_components']}
    nets = [net_loader(**kwargs, **extra_kwargs, embedding_net=embedding)]

    # define training arguments
    train_args = {
        'training_batch_size': cfg.infer.batch_size,
        'learning_rate': cfg.infer.learning_rate,
        'validation_fraction': cfg.infer.val_frac,
    }

    # make output directory
    os.makedirs(out_dir, exist_ok=True)

    # initialize the trainer
    runner = InferenceRunner.load(
        backend=cfg.infer.backend,
        engine=cfg.infer.engine,
        prior=prior,
        nets=nets,
        device=cfg.infer.device,
        train_args=train_args,
        out_dir=out_dir
    )

    # train the model
    posterior, histories = runner(loader=loader)

    return posterior, histories


def run_validation(posterior, history, x, theta, out_dir, names=None):
    logging.info('Running validation...')

    # Plot training history
    logging.info('Plotting training history...')
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, h in enumerate(history):
        ax.plot(h['validation_log_probs'], label=f'Net {i}')
    ax.set(xlabel='Epoch', ylabel='Validation log prob')
    ax.legend()
    f.savefig(join(out_dir, 'loss.jpg'), dpi=200, bbox_inches='tight')

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


def run_experiment(exp, cfg, model_path, names=None):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)
    for kmax in exp.kmax:
        logging.info(f'Running inference for {name} with kmax={kmax}')
        exp_path = join(model_path, f'kmax-{kmax}')

        # load training/test data
        try:
            logging.info(f'Loading training/test data from {exp_path}')
            x_train = np.load(join(exp_path, 'x_train.npy'))
            theta_train = np.load(join(exp_path, 'theta_train.npy'))
            x_test = np.load(join(exp_path, 'x_test.npy'))
            theta_test = np.load(join(exp_path, 'theta_test.npy'))
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Could not find training/test data for {name} with kmax={kmax}'
                '. Make sure to run cmass.infer.preprocess first.'
            )

        logging.info(f'Split: {len(x_train)} training, {len(x_test)} testing')

        # run inference
        posterior, history = run_inference(x_train, theta_train, cfg, exp_path)

        # run validation
        x_test = torch.Tensor(x_test).to(cfg.infer.device)
        theta_test = torch.Tensor(theta_test).to(cfg.infer.device)
        run_validation(posterior, history, x_test, theta_test,
                       exp_path, names=names)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    cfg = parse_nbody_config(cfg)
    model_dir = join(cfg.meta.wdir, cfg.nbody.suite, cfg.sim, 'models')
    if cfg.infer.save_dir is not None:
        model_dir = cfg.infer.save_dir
    if cfg.infer.exp_index is not None:
        cfg.infer.experiments = split_experiments(cfg.infer.experiments)
        cfg.infer.experiments = [cfg.infer.experiments[cfg.infer.exp_index]]

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
