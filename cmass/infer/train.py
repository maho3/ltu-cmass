"""
Trains posterior inference models using the ltu-ili package.

This script loads preprocessed data and trains a neural network to learn the
posterior distribution of cosmological and HOD parameters, given a set of
summary statistics. The training process is configurable via Hydra.

The script supports:
- Different inference backends (e.g., 'lampe', 'sbi').
- Various neural network architectures (e.g., FCN, CNN).
- Training with and without pre-compression of summary statistics.
- Retraining models based on Optuna hyperparameter optimization studies.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
import yaml
import time
import optuna

from .tools import select_top_trials, split_experiments, prepare_loader
from .hyperparameters import sample_hyperparameters_randomly
from ..utils import timing_decorator, clean_up
from ..nbody.tools import parse_nbody_config

import ili
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner
from ili.embedding import FCN
from .architectures import CNN, MultiHeadEmbedding, FunnelNetwork, MultiHeadFunnel


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
                ' include_hod or include_noise might not be set correctly.')
        if hodprior is not None:
            # TODO: support non-uniform HOD priors
            types = np.char.lower(hodprior[:, 1].astype(str))
            if not np.all((types == 'uniform') | (types == 'truncnorm')):
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


def _train_runner(loader, prior, nets, train_args, out_dir,
                    backend, engine, device, verbose=False):
    """Helper function to run training."""
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


def run_training(
    x_train, theta_train, x_val, theta_val, out_dir,
    cfg, mcfg,
    hodprior=None, noiseprior=None,
    start_idx=None,
    validation_smoothing_method='none', ema_decay=0.9,
):
    """
    Train a neural network to emulate a posterior distribution.
    """
    # select the network configuration
    verbose = cfg.infer.verbose
    if verbose:
        logging.info(f'Using network architecture: {mcfg}')

    # define a prior
    prior = prepare_prior(cfg.infer.prior, device=cfg.infer.device,
                          theta=theta_train,
                          hodprior=hodprior, noiseprior=noiseprior)

    # define an embedding network
    if mcfg.embedding_net == 'fcn':
        if mcfg.fcn_depth == 0:
            embedding = nn.Identity()
        else:
            embedding = FCN(
                n_hidden=[mcfg.fcn_width]*mcfg.fcn_depth,
                act_fn='ReLU'
            )
    elif mcfg.embedding_net == 'cnn':
        out_channels = [mcfg.out_channels] * mcfg.cnn_depth
        embedding = CNN(
            out_channels=out_channels,
            kernel_size=mcfg.kernel_size,
            act_fn='ReLU'
        )
    elif mcfg.embedding_net == 'mhe':
        in_features = np.diff(start_idx).tolist()
        out_features = [mcfg.out_features] * len(in_features)
        hidden_layers = [[mcfg.hidden_width]*mcfg.hidden_depth] * len(in_features)
        embedding = MultiHeadEmbedding(
            start_idx=start_idx,
            in_features=in_features,
            out_features=out_features,
            hidden_layers=hidden_layers,
            act_fn='ReLU'
        )
    elif mcfg.embedding_net == 'fun':
        embedding = FunnelNetwork(
            in_features=x_train.shape[-1],
            out_features=mcfg.out_features,
            hidden_depth=mcfg.hidden_depth,
            act_fn='ReLU'
        )
    elif mcfg.embedding_net == 'mhf':
        in_features = np.diff(start_idx).tolist()
        out_features = [mcfg.out_features] * len(in_features)
        hidden_depth = [mcfg.hidden_depth] * len(in_features)
        embedding = MultiHeadFunnel(
            start_idx=start_idx,
            in_features=in_features,
            out_features=out_features,
            hidden_depth=hidden_depth,
            act_fn='ReLU'
        )
    else:
        raise ValueError(f"Unknown embedding net: {mcfg.embedding_net}")

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
        'learning_rate': mcfg.learning_rate if 'learning_rate' in mcfg else cfg.infer.learning_rate,
        'stop_after_epochs': cfg.infer.stop_after_epochs,
        'validation_fraction': cfg.infer.val_frac,
        'weight_decay': mcfg.weight_decay if 'weight_decay' in mcfg else cfg.infer.weight_decay,
        'lr_decay_factor': mcfg.lr_decay_factor if 'lr_decay_factor' in mcfg else cfg.infer.lr_decay_factor,
        'lr_patience': mcfg.lr_patience if 'lr_patience' in mcfg else cfg.infer.lr_patience,
        'ema_decay': ema_decay,
        'validation_smoothing_method': validation_smoothing_method.lower(),
    }

    # setup data loaders
    batch_size = mcfg.batch_size if 'batch_size' in mcfg else cfg.infer.batch_size
    train_loader = prepare_loader(
        x_train, theta_train,
        device=cfg.infer.device,
        batch_size=batch_size, shuffle=True)
    val_loader = prepare_loader(
        x_val, theta_val,
        device=cfg.infer.device,
        batch_size=batch_size, shuffle=False)
    loader = TorchLoader(train_loader, val_loader)

    # train the model
    posterior, histories = _train_runner(
        loader=loader,
        prior=prior,
        nets=nets,
        train_args=train_args,
        out_dir=out_dir,
        backend=cfg.infer.backend,
        engine=cfg.infer.engine,
        device=cfg.infer.device,
        verbose=verbose,
    )

    return posterior, histories


def run_training_with_precompression(
    x_train, theta_train, x_val, theta_val, out_dir,
    cfg, mcfg,
    hodprior=None, noiseprior=None,
    start_idx=None,
    validation_smoothing_method='none', ema_decay=0.9,
):
    """
    Train a neural network with a pre-compression layer.
    """
    # select the network configuration
    verbose = cfg.infer.verbose
    if verbose:
        logging.info(f'Using network architecture: {mcfg}')
    if cfg.infer.embedding_net != 'fcn':
        # TODO: implement
        raise ValueError(f'Precompression only supported for FCN embedding_net.')

    # define a prior
    prior = prepare_prior(cfg.infer.prior, device=cfg.infer.device,
                          theta=theta_train,
                          hodprior=hodprior, noiseprior=noiseprior)

    # define training arguments
    train_args = {
        'learning_rate': mcfg.learning_rate if 'learning_rate' in mcfg else cfg.infer.learning_rate,
        'stop_after_epochs': cfg.infer.stop_after_epochs,
        'validation_fraction': cfg.infer.val_frac,
        'weight_decay': mcfg.weight_decay if 'weight_decay' in mcfg else cfg.infer.weight_decay,
        'lr_decay_factor': mcfg.lr_decay_factor if 'lr_decay_factor' in mcfg else cfg.infer.lr_decay_factor,
        'lr_patience': mcfg.lr_patience if 'lr_patience' in mcfg else cfg.infer.lr_patience,
        'ema_decay': ema_decay,
        'validation_smoothing_method': validation_smoothing_method.lower(),
    }

    # setup data loaders
    batch_size = mcfg.batch_size if 'batch_size' in mcfg else cfg.infer.batch_size
    train_loader = prepare_loader(
        x_train, theta_train,
        device=cfg.infer.device,
        batch_size=batch_size, shuffle=True)
    val_loader = prepare_loader(
        x_val, theta_val,
        device=cfg.infer.device,
        batch_size=batch_size, shuffle=False)
    loader = TorchLoader(train_loader, val_loader)

    # instantiate your neural networks to be used as an ensemble
    if cfg.infer.backend == 'lampe':
        net_loader = ili.utils.load_nde_lampe
        extra_kwargs = {}
    else:
        raise NotImplementedError

    # ~~~~~ TRAIN PRE-COMPRESSION NETWORK ~~~~~
    logging.info('Training pre-compression network...')
    nets = [net_loader(model='mdn', hidden_features=mcfg.fcn_width,
                       hidden_depth=mcfg.fcn_depth,
                       num_components=4)]

    # train the model
    posterior, _ = _train_runner(
        loader=loader,
        prior=prior,
        nets=nets,
        train_args=train_args,
        out_dir=out_dir,
        backend=cfg.infer.backend,
        engine=cfg.infer.engine,
        device=cfg.infer.device,
        verbose=verbose,
    )

    embedding = posterior.posteriors[0].nde.flow.hyper
    # freeze the embedding network
    for param in embedding.parameters():
        param.requires_grad = False

    # ~~~~~ TRAIN FINAL NETWORK ~~~~~
    logging.info('Training final network with pre-compression...')
    kwargs = {k: v for k, v in mcfg.items() if k in [
        'model', 'hidden_features', 'num_transforms', 'num_components']}
    nets = [net_loader(**kwargs, **extra_kwargs, embedding_net=embedding)]

    # train the model
    posterior, histories = _train_runner(
        loader=loader,
        prior=prior,
        nets=nets,
        train_args=train_args,
        out_dir=out_dir,
        backend=cfg.infer.backend,
        engine=cfg.infer.engine,
        device=cfg.infer.device,
        verbose=verbose,
    )

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
        ids_train = np.load(join(exp_path, 'ids_train.npy'))
        x_val = np.load(join(exp_path, 'x_val.npy'))
        theta_val = np.load(join(exp_path, 'theta_val.npy'))
        ids_val = np.load(join(exp_path, 'ids_val.npy'))
        x_test = np.load(join(exp_path, 'x_test.npy'))
        theta_test = np.load(join(exp_path, 'theta_test.npy'))
        ids_test = np.load(join(exp_path, 'ids_test.npy'))
        filepath = join(exp_path, 'hodprior.csv')
        hodprior = (np.genfromtxt(filepath, delimiter=',', dtype=object)
                    if os.path.exists(filepath) else None)
        filepath = join(exp_path, 'noiseprior.yaml')
        noiseprior = (OmegaConf.load(filepath)
                      if os.path.exists(filepath) else None)
        with open(join(exp_path, 'x_startidx.txt'), 'r') as f:
            _ = f.readline().strip().split(',') # summary labels
            startidx = np.array(f.readline().strip().split(',')).astype(int)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Could not find training/test data in {exp_path}. '
            'Make sure to run cmass.infer.preprocess first.'
        )

    return (x_train, theta_train, ids_train,
            x_val, theta_val, ids_val,
            x_test, theta_test, ids_test,
            hodprior, noiseprior,
            startidx)


def run_experiment(exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)

    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]
    validation_smoothing_method = cfg.infer.get(
        'validation_smoothing_method', 'none')
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

            logging.info(
                f'Split: {len(x_train)} training, {len(x_val)} validation, '
                f'{len(x_test)} testing')

            out_dir = join(exp_path, 'nets', f'net-{cfg.infer.net_index}')
            logging.info(f'Saving models to {out_dir}')

            # run training
            start = time.time()
            posterior, histories = run_training(
                x_train, theta_train, x_val, theta_val, out_dir=out_dir,
                cfg=cfg, mcfg=cfg.net,
                hodprior=hodprior, noiseprior=noiseprior,
                start_idx=startidx,
                validation_smoothing_method=validation_smoothing_method,
                ema_decay=ema_decay,
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


def select_nets_retrain(exp_path, Nnets):
    '''
    Select nets from hyperparameter study with cross-validation splits, and
    retrain on the presaved train/val split NOT USED during cross-validation.
    '''
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

    # return trial numbers and configs
    trial_numbers = [t.number for t in top_trials]
    mcfgs = [t.user_attrs['mcfg'] for t in top_trials]
    return trial_numbers, mcfgs


def run_retraining(exp, cfg, model_path):
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
            trial_numbers, net_configs = select_nets_retrain(exp_path, Nnets)

            for trial_number, config in zip(trial_numbers, net_configs):

                out_dir = join(exp_path, "nets", f"net-{trial_number}")
                os.makedirs(out_dir, exist_ok=True)

                # load training/test data: we retrain on the original split
                # from cmass.infer.preprocess
                (x_train, theta_train, ids_train,
                 x_val, theta_val, ids_val,
                 x_test, theta_test, ids_test,
                 hodprior, noiseprior,
                 startidx) = load_preprocessed_data(exp_path)

                logging.info(
                    f'Split: {len(x_train)} training, {len(x_val)} validation, '
                    f'{len(x_test)} testing')

                mcfg = OmegaConf.create(config)

                start = time.time()
                posterior, histories = train_fn(
                    x_train, theta_train, x_val, theta_val, out_dir=out_dir,
                    cfg=cfg, mcfg=mcfg,
                    hodprior=hodprior, noiseprior=noiseprior, verbose=False,
                    start_idx=startidx,
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
                log_prob_test = evaluate_posterior(
                    posterior, x_test, theta_test)
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

    if not hasattr(cfg.infer, 'retrain') or not cfg.infer.retrain:
        cfg.net = sample_hyperparameters_randomly(
            hyperprior=cfg.net,
            embedding_net=cfg.infer.embedding_net,
            seed=cfg.infer.net_index
        )
        runner = run_experiment
    else:
        runner = run_retraining

    tracer = cfg.infer.tracer
    logging.info(f'Running {tracer} inference...')
    for exp in cfg.infer.experiments:
        save_path = join(model_dir, tracer, '+'.join(exp.summary))
        runner(exp, cfg, save_path)


if __name__ == "__main__":
    main()
