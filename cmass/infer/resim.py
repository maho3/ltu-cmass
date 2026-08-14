"""
A script to find "posterior resimulations" for a trained model.

Given a trained posterior ensemble, this picks a single practical test point
(the test-set simulation whose parameters sit closest to the center of the
prior), infers its posterior, and then importance-samples the existing
train/val/test data for simulations whose parameters are representative of that
posterior. No new simulations are generated -- this only reweights and selects
data vectors we already have.

Importance weights are w_i propto q(theta_i | x_obs) / p_prior(theta_i), which
is the correct correction when the dataset's parameters were drawn from the
prior. Note that LampeEnsemble.log_prob is a weighted *sum* of member log
probabilities rather than a true log mixture; we use it anyway for consistency
with cmass.infer.validate.

Selection is the top-n_resim pool points by weight. Downstream statistics
(data-space coverage, ESS) carry w_i through rather than treating selection
as unweighted, so deterministic top-k is more sample-efficient than a
stochastic resample: every selected point is genuinely high-density, and none
are dropped or duplicated by chance.
"""

import os
import logging
from os.path import join
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from matplotlib import pyplot as plt

from .tools import split_experiments, iter_kcuts, kcut_dirname
from .validate import load_ensemble
from ..utils import timing_decorator, clean_up
from ..nbody.tools import parse_nbody_config


SPLITS = ('train', 'val', 'test')

_RESIM_DEFAULTS = dict(
    n_resim=100,        # number of resimulations to select
    n_post=10000,       # number of direct posterior samples (for diagnostics)
    batch_size=2048,    # batch size for log_prob evaluation
    pool=list(SPLITS),  # which splits to draw resimulations from
)


def _resim_cfg(cfg):
    sub = cfg.infer.get('resim', {}) or {}
    out = dict(_RESIM_DEFAULTS)
    out.update({k: v for k, v in sub.items() if v is not None})
    return OmegaConf.create(out)


def load_pool(exp_path, splits):
    """Concatenate the requested splits into a single candidate pool."""
    xs, thetas, ids, tags = [], [], [], []
    for s in splits:
        x = np.load(join(exp_path, f'x_{s}.npy'))
        theta = np.load(join(exp_path, f'theta_{s}.npy'))
        i = np.load(join(exp_path, f'ids_{s}.npy'))
        xs.append(x)
        thetas.append(theta)
        ids.append(i)
        tags.append(np.full(len(x), s))
    return (np.concatenate(xs), np.concatenate(thetas),
            np.concatenate(ids), np.concatenate(tags))


def load_test_split(test_path):
    """Load only the test split of an external suite, from which the single
    observed test point is drawn."""
    x = np.load(join(test_path, 'x_test.npy'))
    theta = np.load(join(test_path, 'theta_test.npy'))
    ids = np.load(join(test_path, 'ids_test.npy'))
    return x, theta, ids


def load_labels(exp_path):
    """Read the summary block labels and their start indices in x."""
    with open(join(exp_path, 'x_startidx.txt')) as f:
        labels = f.readline().strip().split(',')
        startidx = list(map(int, f.readline().strip().split(',')))
    return labels, startidx


def param_names(exp_path):
    """Parameter names for theta's columns as they actually were at training
    time. Read from exp_path's own saved config.yaml rather than the current
    run's cfg -- infer.include_hod/include_noise/subselect_cosmo are commonly
    overridden per-run (e.g. infer.include_hod=False is standard for most
    jobs), which would silently desync from a model trained with different
    flags and misalign every downstream name with the wrong theta column."""
    saved = OmegaConf.load(join(exp_path, 'config.yaml')).infer
    names = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
    if saved.subselect_cosmo is not None:
        names = [names[i] for i in saved.subselect_cosmo]
    filepath = join(exp_path, 'hodprior.csv')
    if saved.include_hod and os.path.exists(filepath):
        hodprior = np.genfromtxt(filepath, delimiter=',', dtype=object)
        names += hodprior[:, 0].astype('str').tolist()
    if saved.include_noise:
        names += ['noise_radial', 'noise_transverse']
    return names


def empirical_quantiles(theta_pool, theta):
    """Map each parameter through the empirical CDF of the training pool, so
    that 0.5 is that parameter's median. Measuring centrality in quantile
    space rather than in scaled parameter units keeps skewed priors (the
    log-uniform noise parameters especially) from being under-penalized far
    out in their tails."""
    ref = np.sort(theta_pool, axis=0)
    n = len(ref)
    q = np.empty_like(theta, dtype=np.float64)
    for j in range(theta.shape[1]):
        lo = np.searchsorted(ref[:, j], theta[:, j], side='left')
        hi = np.searchsorted(ref[:, j], theta[:, j], side='right')
        q[:, j] = 0.5 * (lo + hi) / n  # midrank, so ties sit at their center
    return q


def select_test_point(theta, tags, theta_pool):
    """Index of the test-split point closest to the empirical median of the
    training pool, in per-parameter quantile distance."""
    q = empirical_quantiles(theta_pool, theta)
    d = np.linalg.norm(q - 0.5, axis=-1)
    if tags is not None:
        d = np.where(tags == 'test', d, np.inf)
    if not np.isfinite(d).any():
        raise RuntimeError('No test-split points to choose from.')
    return int(np.argmin(d))


def batched_log_prob(ensemble, theta, x_obs, batch_size, device='cpu'):
    """log q(theta_i | x_obs) for every theta_i, batched."""
    theta = torch.Tensor(theta).to(device)
    x_obs = torch.Tensor(x_obs).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(theta), batch_size):
            t = theta[i:i+batch_size]
            xb = x_obs.unsqueeze(0).expand(len(t), -1)
            out.append(ensemble.log_prob(t, xb).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def normalize_weights(logw):
    w = np.exp(logw - logw.max())
    return w / w.sum()


def top_k_by_weight(logw, k):
    """Indices of the k largest importance weights. Deterministic: every
    selected point is genuinely among the highest-density resimulation
    candidates, and downstream statistics reweight by w_resim rather than
    treating selection as an unweighted draw."""
    return np.argsort(logw)[::-1][:k]


def weighted_quantile(x, w, q):
    """Weighted quantiles along axis 0 of x, with per-row weights w."""
    order = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, order, axis=0)
    ws = w[order]
    cw = np.cumsum(ws, axis=0)
    cw /= cw[-1]
    out = np.empty((len(q), x.shape[1]))
    for j in range(x.shape[1]):
        out[:, j] = np.interp(q, cw[:, j], xs[:, j])
    return out


def plot_dataspace(x_obs, x_resim, w_resim, x_pool, labels, startidx, out_dir):
    """Do the resimulated data vectors cover the observed one?"""
    n = len(labels)
    fig, axs = plt.subplots(2, n, figsize=(4.2*n, 6), squeeze=False,
                            sharex='col')
    q = [0.025, 0.16, 0.5, 0.84, 0.975]
    for j, lab in enumerate(labels):
        sl = slice(startidx[j], startidx[j+1])
        idx = np.arange(startidx[j], startidx[j+1]) - startidx[j]
        obs = x_obs[sl]
        qs = weighted_quantile(x_resim[:, sl], w_resim, q)
        pool_lo, pool_hi = np.percentile(x_pool[:, sl], [2.5, 97.5], axis=0)

        ax = axs[0, j]
        ax.fill_between(idx, pool_lo, pool_hi, color='0.85',
                        label='full pool 95%')
        ax.fill_between(idx, qs[0], qs[4], color='C0', alpha=0.25,
                        label='resim 95%')
        ax.fill_between(idx, qs[1], qs[3], color='C0', alpha=0.45,
                        label='resim 68%')
        ax.plot(idx, qs[2], color='C0', lw=1, label='resim median')
        ax.plot(idx, obs, color='k', lw=1.5, label='observed')
        ax.set_title(lab)
        if j == 0:
            ax.set_ylabel('x')
            ax.legend(fontsize=7, loc='best')

        ax = axs[1, j]
        ax.fill_between(idx, qs[0]-obs, qs[4]-obs, color='C0', alpha=0.25)
        ax.fill_between(idx, qs[1]-obs, qs[3]-obs, color='C0', alpha=0.45)
        ax.plot(idx, qs[2]-obs, color='C0', lw=1)
        ax.axhline(0, color='k', lw=1)
        ax.set_xlabel('feature index')
        if j == 0:
            ax.set_ylabel('resim - observed')

    fig.suptitle('Posterior resimulations in data space')
    fig.tight_layout()
    fig.savefig(join(out_dir, 'plot_resim_dataspace.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def plot_logprob(logq_post, logq_resim, ess, ess_frac, max_w, out_dir):
    """Are the dataset's parameters dense enough to stand in for posterior
    samples? If the dataset histogram sits well left of the direct one, the
    dataset is too sparse around this test point."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.histogram_bin_edges(
        np.concatenate([logq_post, logq_resim]), bins=50)
    ax.hist(logq_post, bins=bins, density=True, histtype='step', lw=1.5,
            color='k', label=f'direct posterior samples (N={len(logq_post)})')
    ax.hist(logq_resim, bins=bins, density=True, histtype='stepfilled',
            alpha=0.5, color='C0',
            label=f'importance-sampled data (N={len(logq_resim)})')
    ax.set_xlabel(r'$\log q(\theta\,|\,x_{\rm obs})$')
    ax.set_title('Posterior density of resimulations vs. direct samples')
    ax.legend(fontsize=8)
    ax.text(0.02, 0.98,
            f'ESS = {ess:.1f} ({100*ess_frac:.2f}% of pool)\n'
            f'max weight = {max_w:.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=8)
    fig.tight_layout()
    fig.savefig(join(out_dir, 'plot_resim_logprob.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def _short(name, maxlen=22):
    """Abbreviate long parameter names so corner axis labels stay readable,
    clipping each underscore-token rather than the tail, so that names
    differing only in the middle (centrals/satellites) stay distinguishable."""
    if len(name) <= maxlen:
        return name
    toks = [t[:4] for t in name.split('_')]
    out = '_'.join(toks)
    while len(out) > maxlen and len(toks) > 1:
        toks.pop()
        out = '_'.join(toks) + '..'
    return out


def plot_corner(theta_post, theta_resim, w_resim, theta_obs, names, out_dir):
    """Check that resims tile the posterior, not just its mode, over every
    inferred parameter (cosmology + HOD + noise). Resim points are a
    deterministic top-weight selection rather than a posterior-representative
    resample, so both the histograms and the marker sizes are weighted by
    w_resim -- otherwise the plot would misleadingly read as an unweighted
    posterior sample."""
    ndim = theta_post.shape[1]
    panel = 2.1 if ndim <= 6 else max(1.0, 12.6 / ndim)
    fs = 8 if ndim <= 6 else 6
    # scale marker area by weight, normalized to a fixed max area
    smax = 40 if ndim <= 6 else 18
    sizes = smax * w_resim / w_resim.max()
    fig, axs = plt.subplots(ndim, ndim, figsize=(panel*ndim, panel*ndim),
                            squeeze=False)
    for i in range(ndim):
        for j in range(ndim):
            ax = axs[i, j]
            ax.tick_params(labelsize=fs-1)
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                bins = np.histogram_bin_edges(theta_post[:, i], bins=30)
                ax.hist(theta_post[:, i], bins=bins, density=True,
                        histtype='step', color='k')
                ax.hist(theta_resim[:, i], bins=bins, weights=w_resim,
                        density=True, histtype='stepfilled', alpha=0.5,
                        color='C0')
                ax.axvline(theta_obs[i], color='C3', lw=1.5)
                ax.set_yticks([])
            else:
                ax.scatter(theta_post[:, j], theta_post[:, i], s=1,
                           alpha=0.1, color='0.5', rasterized=True)
                ax.scatter(theta_resim[:, j], theta_resim[:, i],
                           s=sizes, color='C0', alpha=0.7, edgecolor='none')
                ax.scatter(theta_obs[j], theta_obs[i], marker='*',
                           s=120 if ndim <= 6 else 50,
                           color='C3', zorder=5)
            if i == ndim - 1:
                ax.set_xlabel(_short(names[j]), fontsize=fs, rotation=30,
                              ha='right')
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(_short(names[i]), fontsize=fs)
            elif j > 0:
                ax.set_yticklabels([])
            if ndim > 6:
                ax.locator_params(nbins=3)
    fig.suptitle('Resimulations (blue, sized by weight) vs. direct '
                 'posterior (grey); truth in red')
    fig.tight_layout()
    fig.savefig(join(out_dir, 'plot_resim_corner.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def run_resim(exp_path, cfg, rcfg, test_path=None, out_dir=None):
    """Importance-sample the training suite at exp_path for resimulations of a
    single test point. The test point comes from exp_path's own test split
    unless test_path is given, in which case it is drawn from that external
    suite's test split (infer.testing) while the pool stays the training
    suite."""
    device = cfg.infer.device
    out_dir = out_dir or join(exp_path, 'resim')
    os.makedirs(out_dir, exist_ok=True)

    # load the candidate pool -- always the training suite
    x, theta, ids, tags = load_pool(exp_path, list(rcfg.pool))
    labels, startidx = load_labels(exp_path)
    names = param_names(exp_path)
    logging.info(f'Pool: {len(x)} data vectors, {theta.shape[1]} parameters')

    # load the trained ensemble
    ensemble = load_ensemble(exp_path, cfg.infer.Nnets, plot=False,
                             clean=False).to(device)

    # choose the single test point closest to the training pool's median
    keep = np.ones(len(x), dtype=bool)
    if test_path is None:
        obs_source = 'pool'
        iobs = select_test_point(theta, tags, theta)
        x_obs, theta_obs, id_obs, split_obs = (
            x[iobs], theta[iobs], ids[iobs], tags[iobs])
        keep[iobs] = False  # never resimulate the test point with itself
    else:
        obs_source = test_path
        logging.info(f'Drawing the test point from {test_path}')
        x_t, theta_t, ids_t = load_test_split(test_path)
        if x_t.shape[1] != x.shape[1] or theta_t.shape[1] != theta.shape[1]:
            raise ValueError(
                f'Testing suite at {test_path} has incompatible shapes '
                f'(x: {x_t.shape[1]} vs {x.shape[1]}, theta: '
                f'{theta_t.shape[1]} vs {theta.shape[1]}). It must be '
                'preprocessed with the same summaries and k-cut.')
        # quantiles are still referenced to the training pool
        iobs = select_test_point(theta_t, None, theta)
        x_obs, theta_obs, id_obs, split_obs = (
            x_t[iobs], theta_t[iobs], ids_t[iobs], 'test')

    logging.info(f'Test point: index {iobs}, lhid {id_obs}, '
                 f'split {split_obs}, source {obs_source}')
    q_obs = empirical_quantiles(theta, theta_obs[None])[0]
    logging.info('theta_obs = ' + ', '.join(
        f'{n}={v:.4g} (q={q:.2f})'
        for n, v, q in zip(names, theta_obs, q_obs)))

    # importance weights over the pool
    logq = batched_log_prob(ensemble, theta[keep], x_obs,
                            int(rcfg.batch_size), device)
    with torch.no_grad():
        logp = ensemble.prior.log_prob(
            torch.Tensor(theta[keep]).to(device)).cpu().numpy()
    logw = logq - logp.astype(np.float64)
    logw[~np.isfinite(logw)] = -np.inf

    w = normalize_weights(logw)
    ess = 1. / np.sum(w**2)
    ess_frac = ess / len(w)
    logging.info(f'ESS = {ess:.1f} / {len(w)} ({100*ess_frac:.2f}%), '
                 f'max weight = {w.max():.4f}')
    if ess < 10:
        logging.warning(
            f'Effective sample size is very low (ESS={ess:.1f}). The dataset '
            'is too sparse around this test point for the resimulations to be '
            'representative of the posterior.')

    # select resimulations without replacement
    n_resim = min(int(rcfg.n_resim), int((w > 0).sum()))
    if n_resim < int(rcfg.n_resim):
        logging.warning(f'Only {n_resim} candidates with nonzero weight; '
                        f'requested {rcfg.n_resim}.')
    sel = top_k_by_weight(logw, n_resim)

    x_pool, theta_pool = x[keep], theta[keep]
    ids_pool, tags_pool = ids[keep], tags[keep]
    x_resim, theta_resim = x_pool[sel], theta_pool[sel]
    logq_resim = logq[sel]
    w_resim = w[sel] / w[sel].sum()

    # direct posterior samples, for the sparsity diagnostic
    with torch.no_grad():
        theta_post = ensemble.sample(
            (int(rcfg.n_post),), torch.Tensor(x_obs).to(device),
            show_progress_bars=False).cpu().numpy()
    logq_post = batched_log_prob(ensemble, theta_post, x_obs,
                                 int(rcfg.batch_size), device)

    # save
    np.savez(
        join(out_dir, 'resim.npz'),
        x_obs=x_obs, theta_obs=theta_obs, id_obs=id_obs,
        split_obs=split_obs, index_obs=iobs, obs_source=obs_source,
        x_resim=x_resim, theta_resim=theta_resim, ids_resim=ids_pool[sel],
        split_resim=tags_pool[sel], logw_resim=logw[sel], w_resim=w_resim,
        logq_resim=logq_resim,
        logw_pool=logw, ids_pool=ids_pool, split_pool=tags_pool,
        theta_post=theta_post, logq_post=logq_post,
        ess=ess, ess_frac=ess_frac, max_weight=w.max(),
        param_names=np.array(names), labels=np.array(labels),
        startidx=np.array(startidx),
    )
    logging.info(f'Saved {n_resim} resimulations to {out_dir}')

    # plots
    plot_dataspace(x_obs, x_resim, w_resim, x_pool, labels, startidx, out_dir)
    plot_logprob(logq_post, logq_resim, ess, ess_frac, w.max(), out_dir)
    plot_corner(theta_post, theta_resim, w_resim, theta_obs, names, out_dir)


def run_experiment(exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)
    rcfg = _resim_cfg(cfg)

    testing = cfg.infer.testing
    for kmin, kmax in iter_kcuts(exp):
        exp_path = join(model_path, kcut_dirname(kmin, kmax))
        if not os.path.isdir(exp_path):
            logging.warning(f'No experiment at {exp_path}. Skipping...')
            continue

        # the pool always comes from exp_path; infer.testing only changes
        # where the single observed test point is drawn from
        test_path = out_dir = None
        if testing.suite is not None:
            test_path = join(
                cfg.meta.wdir, testing.suite, testing.sim, 'models',
                cfg.infer.tracer, name, kcut_dirname(kmin, kmax))
            if not os.path.isdir(test_path):
                logging.warning(
                    f'No testing data at {test_path}. Skipping...')
                continue
            out_dir = join(exp_path, 'testing',
                           f'{testing.suite}_{testing.sim}', 'resim')

        logging.info(f'Running resim for {name} with {kmin} <= k <= {kmax}')
        run_resim(exp_path, cfg, rcfg, test_path=test_path, out_dir=out_dir)


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
    logging.info(f'Running {tracer} resimulation sampling...')
    for exp in cfg.infer.experiments:
        save_path = join(model_dir, tracer, '+'.join(exp.summary))
        run_experiment(exp, cfg, save_path)


if __name__ == "__main__":
    main()
