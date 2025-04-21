"""
A script to train ML models on existing suites of simulations.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from torch import nn
import yaml
import time

from .tools import split_experiments, prepare_loader
from ..utils import timing_decorator
from ..nbody.tools import parse_nbody_config

import ili
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner
from ili.embedding import FCN


import matplotlib.pyplot as plt


def prepare_prior(cfg, theta=None):
    # define a prior
    if cfg.infer.prior.lower() == 'uniform':
        prior = ili.utils.Uniform(
            low=theta.min(axis=0),
            high=theta.max(axis=0),
            device=cfg.infer.device)
    elif cfg.infer.prior.lower() == 'quijote':
        # cosmology prior
        prior_lims = np.array([
            (0.1, 0.5),  # Omega_m
            (0.03, 0.07),  # Omega_b
            (0.5, 0.9),  # h
            (0.8, 1.2),  # n_s
            (0.6, 1.0),  # sigma8
        ])
        if theta.shape[-1] > 5:  # galaxy or lightcone
            if cfg.bias.hod.model.lower() == 'zheng07':
                # TODO: load these from cmass.bias.tools.hod?
                hod_lims = np.array([
                    [0.0, 1.5],  # alpha
                    [13.0, 15.0],  # logM0
                    [13.0, 15.0],  # logM1
                    [12.0, 14.0],  # logMmin
                    [0.1, 0.6]]  # sigma_logM
                )
                prior_lims = np.vstack([prior_lims, hod_lims])
        prior = ili.utils.Uniform(
            low=prior_lims[:, 0],
            high=prior_lims[:, 1],
            device=cfg.infer.device)
    else:
        raise NotImplementedError

    return prior


def run_inference(x_train, theta_train, x_val, theta_val, cfg, out_dir):
    start = time.time()

    # select the network configuration
    mcfg = cfg.net
    logging.info(f'Using network architecture: {mcfg}')

    # define a prior
    prior = prepare_prior(cfg, theta=theta_train)

    # define an embedding network
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
    bs, lr = cfg.infer.batch_size, cfg.infer.learning_rate
    bs = mcfg.batch_size if bs is None else bs
    lr = mcfg.learning_rate if lr is None else lr
    train_args = {
        'learning_rate': lr,
        'stop_after_epochs': cfg.infer.stop_after_epochs,
        'validation_fraction': cfg.infer.val_frac,
        'lr_decay_factor': cfg.infer.lr_decay_factor,
        'lr_patience': cfg.infer.lr_patience,
    }

    # setup data loaders
    train_loader = prepare_loader(
        x_train, theta_train,
        device=cfg.infer.device,
        batch_size=bs, shuffle=True)
    val_loader = prepare_loader(
        x_val, theta_val,
        device=cfg.infer.device,
        batch_size=bs, shuffle=False)
    loader = TorchLoader(train_loader, val_loader)

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

    # save the model configuration
    with open(join(out_dir, 'config.yaml'), 'w') as f:
        yaml.dump({**kwargs, **train_args}, f)

    # Plot training history
    logging.info('Plotting training history...')
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, h in enumerate(histories):
        ax.plot(h['validation_log_probs'], label=f'Net {i}', lw=1)
    ax.set(xlabel='Epoch', ylabel='Validation log prob')
    ax.legend()
    f.savefig(join(out_dir, 'loss.jpg'), dpi=200, bbox_inches='tight')

    # Save the timing
    end = time.time()
    with open(join(out_dir, 'timing.txt'), 'w') as f:
        f.write(f'{end - start:.3f}')

    return posterior, histories


def evaluate_posterior(posterior, x, theta):
    log_prob = posterior.log_prob(theta=theta, x=x)
    return log_prob.mean()


def run_experiment(exp, cfg, model_path):
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
            x_val = np.load(join(exp_path, 'x_val.npy'))
            theta_val = np.load(join(exp_path, 'theta_val.npy'))
            x_test = np.load(join(exp_path, 'x_test.npy'))
            theta_test = np.load(join(exp_path, 'theta_test.npy'))
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Could not find training/test data for {name} with kmax={kmax}'
                '. Make sure to run cmass.infer.preprocess first.'
            )

        logging.info(
            f'Split: {len(x_train)} training, {len(x_val)} validation, '
            f'{len(x_test)} testing')

        out_dir = join(exp_path, 'nets', f'net-{cfg.infer.net_index}')
        logging.info(f'Saving models to {out_dir}')

        # run inference
        posterior, history = run_inference(
            x_train, theta_train, x_val, theta_val, cfg, out_dir)

        # evaluate the posterior and save to file
        log_prob_test = evaluate_posterior(posterior, x_test, theta_test)
        with open(join(out_dir, 'log_prob_test.txt'), 'w') as f:
            f.write(f'{log_prob_test}\n')


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
    cfg.net = cfg.net[cfg.infer.net_index]

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    if cfg.infer.halo:
        logging.info('Running halo inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'halo', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path)
    else:
        logging.info('Skipping halo inference...')

    if cfg.infer.galaxy:
        logging.info('Running galaxies inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'galaxy', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path)
    else:
        logging.info('Skipping galaxy inference...')

    if cfg.infer.ngc_lightcone:
        logging.info('Running ngc_lightcone inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'ngc_lightcone', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path)
    else:
        logging.info('Skipping ngc_lightcone inference...')

    if cfg.infer.sgc_lightcone:
        logging.info('Running sgc_lightcone inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'sgc_lightcone', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path)
    else:
        logging.info('Skipping sgc_lightcone inference...')

    if cfg.infer.mtng_lightcone:
        logging.info('Running mtng_lightcone inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'mtng_lightcone', '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path)
    else:
        logging.info('Skipping mtng_lightcone inference...')


if __name__ == "__main__":
    main()
