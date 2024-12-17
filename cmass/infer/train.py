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
import torch

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config
from .loaders import get_cosmo, get_hod, load_Pk, load_lc_Pk, preprocess_Pk

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.embedding import FCN

import matplotlib.pyplot as plt


def aggregate(summlist, paramlist, idlist):
    summaries = defaultdict(list)
    parameters = defaultdict(list)
    ids = defaultdict(list)
    for summ, param, id in zip(summlist, paramlist, idlist):
        for key in summ:
            summaries[key].append(summ[key])
            parameters[key].append(param)
            ids[key].append(id)
    return summaries, parameters, ids


def load_halo_summaries(suitepath, a, Nmax):
    logging.info(f'Looking for halo summaries at {suitepath}')
    simpaths = os.listdir(suitepath)
    simpaths.sort(key=lambda x: int(x))  # sort by lhid
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]

    summlist, paramlist, idlist = [], [], []
    for lhid in tqdm(simpaths):
        sourcepath = join(suitepath, lhid)
        diagpath = join(sourcepath, 'diag')
        if not os.path.isdir(diagpath):
            continue
        diagfile = join(diagpath, 'halos.h5')
        summ = load_Pk(diagfile, a)  # TODO: load other summaries
        if len(summ) > 0:
            summlist.append(summ)
            paramlist.append(get_cosmo(sourcepath))
            idlist.append(lhid)

    summaries, parameters, ids = aggregate(summlist, paramlist, idlist)

    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} / {len(simpaths)} {key}'
            ' summaries')
    return summaries, parameters, ids


def load_galaxy_summaries(suitepath, a, Nmax):
    logging.info(f'Looking for galaxy summaries at {suitepath}')
    simpaths = os.listdir(suitepath)
    simpaths.sort(key=lambda x: int(x))  # sort by lhid
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]

    summlist, paramlist, idlist = [], [], []
    Ntot = 0
    for lhid in tqdm(simpaths):
        sourcepath = join(suitepath, lhid)
        diagpath = join(sourcepath, 'diag', 'galaxies')
        if not os.path.isdir(diagpath):
            continue
        filelist = os.listdir(diagpath)
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
                idlist.append(lhid)

    summaries, parameters, ids = aggregate(summlist, paramlist, idlist)

    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} / {Ntot} {key}'
            ' summaries')
    return summaries, parameters, ids


def load_lightcone_summaries(suitepath, cap, Nmax):
    logging.info(f'Looking for {cap}_lightcone summaries at {suitepath}')
    simpaths = os.listdir(suitepath)
    simpaths.sort(key=lambda x: int(x))  # sort by lhid
    if Nmax >= 0:
        simpaths = simpaths[:Nmax]

    summlist, paramlist, idlist = [], [], []
    Ntot = 0
    for lhid in tqdm(simpaths):
        sourcepath = join(suitepath, lhid)
        diagpath = join(sourcepath, 'diag', f'{cap}_lightcone')
        if not os.path.isdir(diagpath):
            continue
        filelist = os.listdir(diagpath)
        Ntot += len(filelist)
        for f in filelist:
            diagfile = join(sourcepath, 'diag', f'{cap}_lightcone', f)
            summ = load_lc_Pk(diagfile)
            if len(summ) > 0:
                try:
                    paramlist.append(np.concatenate(
                        [get_cosmo(sourcepath), get_hod(diagfile)], axis=0))
                except (OSError, KeyError):
                    continue
                summlist.append(summ)
                idlist.append(lhid)

    summaries, parameters, ids = aggregate(summlist, paramlist, idlist)

    for key in summaries:
        logging.info(
            f'Successfully loaded {len(summaries[key])} / {Ntot} {key}'
            ' summaries')
    return summaries, parameters, ids


def split_train_test(x, theta, ids, test_frac):
    x, theta, ids = map(np.array, [x, theta, ids])
    unique_ids = np.unique(ids)
    np.random.shuffle(unique_ids)
    cutoff = int(len(unique_ids) * (1 - test_frac))
    ids_train, ids_test = unique_ids[:cutoff], unique_ids[cutoff:]

    train_mask = np.isin(ids, ids_train)
    test_mask = np.isin(ids, ids_test)

    x_train, x_test = x[train_mask], x[test_mask]
    theta_train, theta_test = theta[train_mask], theta[test_mask]

    return x_train, x_test, theta_train, theta_test,


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

    if cfg.infer.fcn_hidden is None:
        embedding = None
    else:
        embedding = FCN(
            n_hidden=cfg.infer.fcn_hidden,
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
    nets = []
    for net in cfg.infer.nets:
        logging.info(f'Adding {net}')
        nets.append(net_loader(
            **net,
            **extra_kwargs,
            embedding_net=embedding)
        )

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

    # save test data
    np.save(join(out_dir, 'x_test.npy'), x.to('cpu'))
    np.save(join(out_dir, 'theta_test.npy'), theta.to('cpu'))


def run_experiment(summaries, parameters, ids, exp, cfg, model_path, names=None):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)
    for kmax in exp.kmax:
        logging.info(f'Running inference for {name} with kmax={kmax}')
        exp_path = join(model_path, f'kmax-{kmax}')
        xs = []
        for summ in exp.summary:
            x, theta, id = summaries[summ], parameters[summ], ids[summ]
            if 'Pk' in summ:
                x = preprocess_Pk(x, kmax)
                # remove first 5 bins (TODO: remove with a smarter fix)
                x = x[:, 5:]
            else:
                raise NotImplementedError  # TODO: implement other summaries
            xs.append(x)
        if not np.all([len(x) == len(xs[0]) for x in xs]):
            raise ValueError(
                f'Inconsistent lengths of summaries for {name}. Check that all '
                'summaries have been computed for the same simulations.')
        x = np.concatenate(xs, axis=-1)

        # split train/test
        x_train, x_test, theta_train, theta_test = \
            split_train_test(x, theta, id, cfg.infer.test_frac)
        logging.info(f'Split: {len(x_train)} training, {len(x_test)} testing')

        # run inference
        posterior, history = run_inference(
            x_train, theta_train, cfg, exp_path)

        # run validation
        x_test = torch.Tensor(x_test).to(cfg.infer.device)
        theta_test = torch.Tensor(theta_test).to(cfg.infer.device)
        run_validation(posterior, history, x_test, theta_test,
                       exp_path, names=names)


def split_experiments(exp_cfg):
    new_exps = []
    for exp in exp_cfg:
        for kmax in exp.kmax:
            new_exp = exp.copy()
            new_exp.kmax = [kmax]
            new_exps.append(new_exp)
    return new_exps


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
        summaries, parameters, ids = load_halo_summaries(
            suite_path, cfg.nbody.af, cfg.infer.Nmax)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'halo', '+'.join(exp.summary))
            run_experiment(summaries, parameters, ids, exp, cfg,
                           save_path, names=cosmonames)
    else:
        logging.info('Skipping halo inference...')

    if cfg.infer.galaxy:
        logging.info('Running galaxies inference...')
        summaries, parameters, ids = load_galaxy_summaries(
            suite_path, cfg.nbody.af, cfg.infer.Nmax)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'galaxy', '+'.join(exp.summary))
            run_experiment(summaries, parameters, ids, exp, cfg,
                           save_path, names=cosmonames+hodnames)
    else:
        logging.info('Skipping galaxy inference...')

    if cfg.infer.ngc_lightcone:
        logging.info('Running ngc_lightcone inference...')
        summaries, parameters, ids = load_lightcone_summaries(
            suite_path, 'ngc', cfg.infer.Nmax)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'ngc_lightcone', '+'.join(exp.summary))
            run_experiment(summaries, parameters, ids, exp, cfg,
                           save_path, names=cosmonames+hodnames)
    else:
        logging.info('Skipping ngc_lightcone inference...')

    if cfg.infer.sgc_lightcone:
        logging.info('Running sgc_lightcone inference...')
        summaries, parameters, ids = load_lightcone_summaries(
            suite_path, 'sgc', cfg.infer.Nmax)
        for exp in cfg.infer.experiments:
            save_path = join(model_dir, 'sgc_lightcone', '+'.join(exp.summary))
            run_experiment(summaries, parameters, ids, exp, cfg,
                           save_path, names=cosmonames+hodnames)
    else:
        logging.info('Skipping sgc_lightcone inference...')


if __name__ == "__main__":
    main()
