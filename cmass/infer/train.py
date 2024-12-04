"""
A script to train ML models on existing suites of simulations.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py
from collections import defaultdict
from tqdm import tqdm

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config
from ..bias.apply_hod import parse_hod

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.embedding import FCN

import matplotlib.pyplot as plt


def get_cosmo(source_path):
    cfg = OmegaConf.load(join(source_path, 'config.yaml'))
    return np.array(cfg.nbody.cosmo)

def get_halo_Pk(source_path, a):
    diag_file = join(source_path, 'diag', 'halos.h5')
    a = f'{a:.6f}'
    if not os.path.exists(diag_file):
        return {}
    
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            for stat in ['Pk', 'zPk']:
                if stat in f[a]:
                    summ[stat] = f[a][stat][:]
                    if stat+'_k' in f[a]:  # backwards compatibility  (todo: remove)
                        summ[stat+'_k'] = f[a][stat+'_k'][:]
                    elif stat+'_k3D' in f[a]:
                        summ[stat+'_k'] = f[a][stat+'_k3D'][:]
    except OSError:
        return {}
    summ['cosmo'] = get_cosmo(source_path)
    return summ

def load_halo_summaries(suitepath, a, Nmax):
    logging.info(f'Looking for halo summaries at {suitepath}')

    simpaths = os.listdir(suitepath)
    simpaths.sort(key=lambda x: int(x))  # sort by lhid
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]
    summaries, parameters, meta = defaultdict(list), defaultdict(list), defaultdict(list)
    for lhid in tqdm(simpaths):
        summ = get_halo_Pk(join(suitepath, lhid), a)
        for key in ['Pk', 'zPk']:
            if key in summ:
                summaries[key].append(summ[key])
                parameters[key].append(summ['cosmo'])
                meta[key].append(summ[key+'_k'])
    summaries = {key: np.array(val) for key, val in summaries.items()}
    parameters = {key: np.array(val) for key, val in parameters.items()}
    meta = {key: np.array(val) for key, val in meta.items()}

    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} / {len(simpaths)} {key}'
            ' summaries')  
    return summaries, parameters, meta


def cut_kmax(x, k, kmax):
    if kmax is None:
        return x
    if len(k.shape) > 1:
        k = k[0]
    mask = k < kmax
    return x[:, mask, ...]


def preprocess(x, Pk_poles):
    # only use desired poles
    x = x[..., Pk_poles]

    # log transform
    x = np.log(x + 1e-5)

    # impute nans
    x = np.nan_to_num(x, nan=0.0)

    # flatten
    x = x.reshape(len(x), -1)

    return x

def split_train_test(x, theta, test_frac):
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

def run_validation(posterior, history, x, theta, out_dir):
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
        labels=['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma8']
    )
    metric(posterior, x_obs=xobs, theta_fid=thetaobs)

    # Posterior coverage
    metric = PosteriorCoverage(
        num_samples=1000, sample_method='direct', 
        labels=['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma8'],
        plot_list = ["coverage", "histogram", "predictions", "tarp", "logprob"],
        out_dir=out_dir
    )
    metric(posterior, x, theta)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    cfg = parse_nbody_config(cfg)
    suite_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, 0, check=False
    )[:-2]  # get to the suite directory
    model_path = join(cfg.meta.wdir, cfg.nbody.suite, cfg.sim, 'models')

    if cfg.infer.halo:
        logging.info('Running halo inference...')
        summaries, parameters, meta = load_halo_summaries(
            suite_path, cfg.nbody.af, cfg.infer.Nmax
        )
        for key in summaries:
            for kmax in cfg.infer.kmax:
                logging.info(f'Running inference for {key} with kmax={kmax}')
                out_dir = join(model_path, 'halos', f'{key}', f'kmax-{kmax}')
                x, theta = summaries[key], parameters[key]
                x = cut_kmax(x, meta[key], kmax)
                x = preprocess(x, cfg.infer.Pk_poles)
                x_train, x_test, theta_train, theta_test = \
                    split_train_test(x, theta, cfg.infer.test_frac)

                posterior, history = run_inference(x_train, theta_train, cfg, out_dir)

                run_validation(posterior, history, x_test, theta_test, out_dir)
    else:
        logging.info('Skipping halo inference...')


if __name__ == "__main__":
    main()
