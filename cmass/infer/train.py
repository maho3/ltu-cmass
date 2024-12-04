"""
A script to train ML models on existing suites of simulations.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config
from .loaders import get_cosmo, get_hod, load_Pk, preprocess_Pk

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.embedding import FCN

import matplotlib.pyplot as plt


def load_halo_summaries(suitepath, a, Nmax):
    logging.info(f'Looking for halo summaries at {suitepath}')
    simpaths = os.listdir(suitepath)
    simpaths.sort(key=lambda x: int(x))  # sort by lhid
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]

    summlist, paramlist = [], []
    for lhid in tqdm(simpaths):
        sourcepath = join(suitepath, lhid)
        diagfile = join(sourcepath, 'diag', 'halos.h5')
        summ = load_Pk(diagfile, a)  # TODO: load other summaries
        if len(summ) > 0:
            summlist.append(summ)
            paramlist.append(get_cosmo(sourcepath))

    summaries, parameters = defaultdict(list), defaultdict(list)
    for summ, param in zip(summlist, paramlist):
        for key in summ:
            summaries[key].append(summ[key])
            parameters[key].append(param)

    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} / {len(simpaths)} {key}'
            ' summaries')
    return summaries, parameters


def load_galaxy_summaries(suitepath, a, Nmax):
    logging.info(f'Looking for galaxy summaries at {suitepath}')
    simpaths = os.listdir(suitepath)
    simpaths.sort(key=lambda x: int(x))  # sort by lhid
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]

    summlist, paramlist = [], []
    Ntot = 0
    for lhid in tqdm(simpaths):
        sourcepath = join(suitepath, lhid)
        filelist = os.listdir(join(sourcepath, 'diag', 'galaxies'))
        Ntot += len(filelist)
        for f in filelist:
            diagfile = join(sourcepath, 'diag', 'galaxies', f)
            summ = load_Pk(diagfile, a)
            if len(summ) > 0:
                try:
                    paramlist.append(np.concatenate(
                        [get_cosmo(sourcepath), get_hod(diagfile)], axis=0))
                except (OSError, KeyError):
                    continue
                summlist.append(summ)

    summaries, parameters = defaultdict(list), defaultdict(list)
    for summ, param in zip(summlist, paramlist):
        for key in summ:
            summaries[key].append(summ[key])
            parameters[key].append(param)

    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} / {Ntot} {key}'
            ' summaries')
    return summaries, parameters


def split_train_test(x, theta, test_frac):
    x, theta = np.array(x), np.array(theta)
    cutoff = int(len(x) * (1 - test_frac))
    x_train, x_test = x[:cutoff], x[cutoff:]
    theta_train, theta_test = theta[:cutoff], theta[cutoff:]
    return x_train, x_test, theta_train, theta_test


def run_inference(x, theta, cfg, out_dir):
    loader = NumpyLoader(x=x, theta=theta)

    # define a prior
    if cfg.infer.prior.lower() == 'uniform':
        prior = ili.utils.Uniform(
            low=theta.min(axis=0),
            high=theta.max(axis=0),
            device=cfg.infer.device)
    else:
        raise NotImplementedError

    embedding = FCN(
        n_hidden=cfg.infer.fcn_hidden,
        act_fn='ReLU'
    )

    # instantiate your neural networks to be used as an ensemble
    if cfg.infer.backend == 'lampe':
        net_loader = ili.utils.load_nde_lampe
    elif cfg.infer.backend == 'sbi':
        net_loader = ili.utils.load_nde_sbi
    else:
        raise NotImplementedError
    nets = [
        net_loader(
            model=net.model, hidden_features=net.hidden_features,
            num_transforms=net.num_transforms,
            embedding_net=embedding)
        for net in cfg.infer.nets
    ]

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
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, h in enumerate(history):
        ax.plot(h['validation_log_probs'], label=f'Net {i}')
    ax.set(xlabel='Epoch', ylabel='Validation log prob')
    ax.legend()
    f.savefig(join(out_dir, 'loss.jpg'), dpi=200, bbox_inches='tight')

    # Plotting single posterior
    xobs, thetaobs = x[0], theta[0]
    metric = PlotSinglePosterior(
        num_samples=1000, sample_method='direct',
        labels=names
    )
    metric(posterior, x_obs=xobs, theta_fid=thetaobs)

    # Posterior coverage
    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct',
        labels=names,
        plot_list=["coverage", "histogram", "predictions", "tarp", "logprob"],
        out_dir=out_dir
    )
    metric(posterior, x, theta)


def Pk_pipeline(x, theta, cfg, model_path, names=None):
    for kmax in cfg.infer.Pk.kmax:
        xi, thetai = deepcopy(x), deepcopy(theta)
        logging.info(f'Running inference for Pk with kmax={kmax}')
        out_dir = join(model_path, 'Pk', f'kmax={kmax}')
        xi = preprocess_Pk(xi, kmax, cfg.infer.Pk.poles)
        x_train, x_test, theta_train, theta_test = \
            split_train_test(xi, thetai, cfg.infer.test_frac)

        posterior, history = run_inference(x_train, theta_train, cfg, out_dir)

        run_validation(posterior, history, x_test,
                       theta_test, out_dir, names=names)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    cfg = parse_nbody_config(cfg)
    suite_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, 0, check=False
    )[:-2]  # get to the suite directory
    model_dir = join(cfg.meta.wdir, cfg.nbody.suite, cfg.sim, 'models')

    cosmonames = [r'$\Omega_m$', r'$\Omega_b$', r'$h$', r'$n_s$', r'$\sigma_8$']
    hodnames = [r'$\alpha$', r'$\log M_0$', r'$\log M_1$',
                r'$\log M_{\min}$', r'$\sigma_{\log M}$']  # TODO: load these from file?

    if cfg.infer.halo:
        logging.info('Running halo inference...')
        save_path = join(model_dir, 'halo')
        summaries, parameters = load_halo_summaries(
            suite_path, cfg.nbody.af, cfg.infer.Nmax)
        for key in summaries:
            if 'Pk' in key:
                x, theta = summaries[key], parameters[key]
                Pk_pipeline(x, theta, cfg, save_path, names=cosmonames)
            else:
                raise NotImplementedError  # TODO: implement other summaries+combos
    else:
        logging.info('Skipping halo inference...')

    if cfg.infer.galaxy:
        logging.info('Running galaxies inference...')
        save_path = join(model_dir, 'galaxy')
        summaries, parameters = load_galaxy_summaries(
            suite_path, cfg.nbody.af, cfg.infer.Nmax)
        for key in summaries:
            if 'Pk' in key:
                x, theta = summaries[key], parameters[key]
                Pk_pipeline(x, theta, cfg, save_path, names=cosmonames+hodnames)
            else:
                raise NotImplementedError  # TODO: implement other summaries+combos
    else:
        logging.info('Skipping galaxy inference...')


if __name__ == "__main__":
    main()
