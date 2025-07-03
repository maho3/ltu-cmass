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


def prepare_prior(prior_name, device, theta=None, hodprior=None, noiseprior=None):
    # define a prior
    if prior_name.lower() == 'uniform':
        prior = ili.utils.Uniform(
            low=theta.min(axis=0),
            high=theta.max(axis=0),
            device=device)
    elif prior_name.lower() == 'quijote':
        # cosmology prior
        prior_lims = np.array([
            (0.1, 0.5),  # Omega_m
            (0.03, 0.07),  # Omega_b
            (0.5, 0.9),  # h
            (0.8, 1.2),  # n_s
            (0.6, 1.0),  # sigma8
        ])
        if ((hodprior is not None) or (noiseprior is not None)) and (theta.shape[-1] == 5):
            raise ValueError(
                'HOD or noise priors provided, but theta has only 5 parameters.'
                ' infer_hod or infer_noise might not be set correctly.')
        if hodprior is not None:
            if not np.all(hodprior[:, 1].astype(str) == 'uniform'):
                raise NotImplementedError(
                    "We don't know how to handle non-uniform HOD priors yet.")
            hod_lims = hodprior[:, 2:4].astype(float)
            prior_lims = np.vstack([prior_lims, hod_lims])
        if noiseprior is not None:
            if noiseprior.dist == 'Fixed':
                low, high = -np.inf, np.inf
            elif noiseprior.dist == 'Uniform':
                low, high = noiseprior.params.a, noiseprior.params.b
            elif noiseprior.dist == 'Reciprocal':
                low, high = noiseprior.params.a, noiseprior.params.b
            elif noiseprior.dist == 'Exponential':
                low, high = 0, np.inf
            else:
                raise NotImplementedError(
                    f'Noise prior distribution {noiseprior.dist} not '
                    'implemented.')
            noise_lims = np.array([[low, high]]*2)
            prior_lims = np.vstack([prior_lims, noise_lims])

        prior = ili.utils.Uniform(
            low=prior_lims[:, 0],
            high=prior_lims[:, 1],
            device=device)
    else:
        raise NotImplementedError

    return prior


def run_training(
    x_train, theta_train, x_val, theta_val, out_dir,
    prior_name, mcfg,  # model config
    batch_size, learning_rate, stop_after_epochs, val_frac,
    weight_decay, lr_decay_factor, lr_patience,
    backend, engine, device,
    hodprior=None, noiseprior=None, verbose=True
):
    # select the network configuration
    if verbose:
        logging.info(f'Using network architecture: {mcfg}')

    # define a prior
    prior = prepare_prior(prior_name, device=device,
                          theta=theta_train,
                          hodprior=hodprior, noiseprior=noiseprior)

    # define an embedding network
    if mcfg.fcn_depth == 0:
        embedding = nn.Identity()
    else:
        embedding = FCN(
            n_hidden=[mcfg.fcn_width]*mcfg.fcn_depth,
            act_fn='ReLU'
        )

    # instantiate your neural networks to be used as an ensemble
    if backend == 'lampe':
        net_loader = ili.utils.load_nde_lampe
        extra_kwargs = {}
    elif backend == 'sbi':
        net_loader = ili.utils.load_nde_sbi
        extra_kwargs = {'engine': engine}
    else:
        raise NotImplementedError
    kwargs = {k: v for k, v in mcfg.items() if k in [
        'model', 'hidden_features', 'num_transforms', 'num_components']}
    nets = [net_loader(**kwargs, **extra_kwargs, embedding_net=embedding)]

    # define training arguments
    bs = mcfg.batch_size if 'batch_size' in mcfg else batch_size
    lr = mcfg.learning_rate if 'learning_rate' in mcfg else learning_rate
    wd = mcfg.weight_decay if 'weight_decay' in mcfg else weight_decay
    lrp = mcfg.lr_patience if 'lr_patience' in mcfg else lr_patience
    lrdf = mcfg.lr_decay_factor if 'lr_decay_factor' in mcfg else lr_decay_factor
    train_args = {
        'learning_rate': lr,
        'stop_after_epochs': stop_after_epochs,
        'validation_fraction': val_frac,
        'weight_decay': wd,
        'lr_decay_factor': lrdf,
        'lr_patience': lrp
    }

    # setup data loaders
    train_loader = prepare_loader(
        x_train, theta_train,
        device=device,
        batch_size=bs, shuffle=True)
    val_loader = prepare_loader(
        x_val, theta_val,
        device=device,
        batch_size=bs, shuffle=False)
    loader = TorchLoader(train_loader, val_loader)

    # make output directory
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # initialize the trainer
    runner = InferenceRunner.load(
        backend=backend,
        engine=engine,
        prior=prior,
        nets=nets,
        device=device,
        train_args=train_args,
        out_dir=out_dir
    )

    # train the model
    posterior, histories = runner(loader=loader, verbose=verbose)

    return posterior, histories


def plot_training_history(histories, out_dir):
    # Plot training history
    logging.info('Plotting training history...')
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, h in enumerate(histories):
        ax.plot(h['validation_log_probs'], label=f'Net {i}', lw=1)
    ax.set(xlabel='Epoch', ylabel='Validation log prob')
    ax.legend()
    f.savefig(join(out_dir, 'loss.jpg'), dpi=100, bbox_inches='tight')
    plt.close(f)


def evaluate_posterior(posterior, x, theta):
    log_prob = posterior.log_prob(theta=theta, x=x)
    return log_prob.mean()


def load_preprocessed_data(exp_path):
    """
    Load preprocessed data from the given experiment path.
    """
    try:
        x_train = np.load(join(exp_path, 'x_train.npy'))
        theta_train = np.load(join(exp_path, 'theta_train.npy'))
        x_val = np.load(join(exp_path, 'x_val.npy'))
        theta_val = np.load(join(exp_path, 'theta_val.npy'))
        x_test = np.load(join(exp_path, 'x_test.npy'))
        theta_test = np.load(join(exp_path, 'theta_test.npy'))
        filepath = join(exp_path, 'hodprior.csv')
        hodprior = (np.genfromtxt(filepath, delimiter=',', dtype=object)
                    if os.path.exists(filepath) else None)
        filepath = join(exp_path, 'noiseprior.yaml')
        noiseprior = (OmegaConf.load(filepath)
                      if os.path.exists(filepath) else None)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Could not find training/test data in {exp_path}. '
            'Make sure to run cmass.infer.preprocess first.'
        )

    return (x_train, theta_train, x_val, theta_val, x_test, theta_test,
            hodprior, noiseprior)


def run_experiment(exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)

    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]

    for kmin in kmin_list:
        for kmax in kmax_list:
            logging.info(
                f'Running training for {name} with {kmin} <= k <= {kmax}')
            exp_path = join(model_path, f'kmin-{kmin}_kmax-{kmax}')

            # load training/test data
            (x_train, theta_train,
             x_val, theta_val,
             x_test, theta_test,
             hodprior, noiseprior) = load_preprocessed_data(exp_path)

            logging.info(
                f'Split: {len(x_train)} training, {len(x_val)} validation, '
                f'{len(x_test)} testing')

            out_dir = join(exp_path, 'nets', f'net-{cfg.infer.net_index}')
            logging.info(f'Saving models to {out_dir}')

            # run training
            start = time.time()
            posterior, histories = run_training(
                x_train, theta_train, x_val, theta_val, out_dir=out_dir,
                prior_name=cfg.infer.prior, mcfg=cfg.net,
                batch_size=cfg.infer.batch_size,
                learning_rate=cfg.infer.learning_rate,
                stop_after_epochs=cfg.infer.stop_after_epochs,
                val_frac=cfg.infer.val_frac,
                weight_decay=cfg.infer.weight_decay,
                lr_decay_factor=cfg.infer.lr_decay_factor,
                lr_patience=cfg.infer.lr_patience,
                backend=cfg.infer.backend, engine=cfg.infer.engine,
                device=cfg.infer.device,
                hodprior=hodprior, noiseprior=noiseprior,
            )
            end = time.time()

            # Save the timing and metadata
            with open(join(out_dir, 'timing.txt'), 'w') as f:
                f.write(f'{end - start:.3f}')
            with open(join(out_dir, 'model_config.yaml'), 'w') as f:
                yaml.dump(OmegaConf.to_container(cfg.net, resolve=True), f)

            # plot training history
            plot_training_history(histories, out_dir)

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

    for tracer in ['halo', 'galaxy',
                   'ngc_lightcone', 'sgc_lightcone', 'mtng_lightcone',
                   'simbig_lightcone']:
        if not getattr(cfg.infer, tracer):
            logging.info(f'Skipping {tracer} inference...')
            continue

        logging.info(f'Running {tracer} inference...')
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, tracer, '+'.join(exp.summary))
            run_experiment(exp, cfg, save_path)


if __name__ == "__main__":
    main()
