"""
Validates a trained NLE by plotting emulations on test data.

Supports out-of-distribution (OOD) testing via cfg.infer.testing, mirroring
the same pattern as cmass.infer.validate. When cfg.infer.testing.suite is set,
test data is loaded from that suite/sim and outputs are written to a
testing/ subdirectory.

Usage:
    python -m cmass.infer.validate_nle suite=... sim=...
    python -m cmass.infer.validate_nle suite=... sim=... \
        infer.testing.suite=quijotelike infer.testing.sim=fastpm
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib.pyplot as plt

from .train_nle import load_nle
from .tools import split_experiments
from ..utils import timing_decorator, clean_up
from ..nbody.tools import parse_nbody_config


log = logging.getLogger(__name__)


def _load_segments(exp_path):
    """Parse x_startidx.txt into a list of (name, start, end) tuples, or None."""
    p = join(exp_path, 'x_startidx.txt')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        names = f.readline().strip().split(',')
        indices = list(map(int, f.readline().strip().split(',')))
    return list(zip(names, indices[:-1], indices[1:]))


def plot_emulations(model, x_test, theta_test, out_path, device='cpu',
                    n_panels=9, segments=None):
    """Draw predicted mean/std from p(x|theta) and pull residuals for random test points.

    If segments is a list of (name, start, end) tuples, each summary is
    coloured independently; otherwise a single colour is used.
    """
    rng = np.random.default_rng(0)
    n_panels = min(n_panels, len(x_test))
    idxs = rng.choice(len(x_test), size=n_panels, replace=False)

    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows * 2, ncols, figsize=(5 * ncols, 3 * nrows * 2),
        gridspec_kw={'height_ratios': [2, 1] * nrows})
    axes = np.atleast_2d(axes)

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            row_i = i // ncols
            col_i = i % ncols
            ax_data = axes[2 * row_i, col_i]
            ax_res = axes[2 * row_i + 1, col_i]

            theta_i = torch.tensor(
                theta_test[idx], dtype=torch.float32
            ).unsqueeze(0).to(device)

            mean, std = model.predict(theta_i)
            mean = mean.squeeze(0).cpu().numpy()
            std = std.squeeze(0).cpu().numpy()
            bins = np.arange(len(mean))
            residual = (x_test[idx] - mean) / std

            if segments is not None:
                for k, (seg_name, seg_s, seg_e) in enumerate(segments):
                    color = prop_cycle[k % len(prop_cycle)]
                    sl = slice(seg_s, seg_e)
                    label = seg_name if i == 0 else None
                    ax_data.plot(bins[sl], mean[sl], lw=1, color=color,
                                 label=label)
                    ax_data.fill_between(
                        bins[sl], mean[sl] - std[sl], mean[sl] + std[sl],
                        alpha=0.3, color=color)
                    ax_data.plot(bins[sl], x_test[idx][sl], '--', lw=1,
                                 color=color)
                    ax_res.plot(bins[sl], residual[sl], lw=1, color=color)
            else:
                ax_data.plot(bins, mean, lw=1, color='C0',
                             label='predicted mean' if i == 0 else None)
                ax_data.fill_between(
                    bins, mean - std, mean + std, alpha=0.3, color='C0',
                    label=r'$\pm 1\sigma$' if i == 0 else None)
                ax_data.plot(bins, x_test[idx], 'k--', lw=1,
                             label='truth' if i == 0 else None)
                ax_res.plot(bins, residual, lw=1, color='C0')

            ax_data.set_title(
                r'$\theta$=[' +
                ', '.join(f'{v:.2f}' for v in theta_test[idx]) + ']',
                fontsize=8)
            if i == 0:
                ax_data.legend(fontsize=7)

            ax_res.axhline(0, color='k', lw=0.8, ls='--')
            ax_res.axhline(1, color='gray', lw=0.5, ls=':')
            ax_res.axhline(-1, color='gray', lw=0.5, ls=':')
            ax_res.set_xlabel('data-vector index')
            ax_res.set_ylabel(r'$(x_{\rm true} - \mu)/\sigma$', fontsize=7)

    for j in range(n_panels, nrows * ncols):
        row_j = j // ncols
        col_j = j % ncols
        axes[2 * row_j, col_j].set_visible(False)
        axes[2 * row_j + 1, col_j].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Saved emulation plot to {out_path}')


def run_experiment(exp, cfg, model_path):
    assert len(exp.summary) > 0, 'No summaries provided for inference'
    name = '+'.join(exp.summary)
    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]

    for kmin in kmin_list:
        for kmax in kmax_list:
            log.info(
                f'Running NLE validation for {name} '
                f'with {kmin} <= k <= {kmax}')
            exp_path = join(model_path, f'kmin-{kmin}_kmax-{kmax}')

            # ── resolve test data path (OOD or in-distribution) ──────
            if cfg.infer.testing.suite is None:
                log.info(f'Loading test data from {exp_path}')
                test_path = exp_path
                out_path = exp_path
            else:
                test_path = join(
                    cfg.meta.wdir,
                    cfg.infer.testing.suite, cfg.infer.testing.sim,
                    'models', cfg.infer.tracer, name,
                    f'kmin-{kmin}_kmax-{kmax}')
                log.info(f'Loading OOD test data from {test_path}')
                out_path = join(
                    exp_path, 'testing',
                    f'{cfg.infer.testing.suite}_{cfg.infer.testing.sim}')
                os.makedirs(out_path, exist_ok=True)

            try:
                x_test = np.load(join(test_path, 'x_test.npy'))
                theta_test = np.load(join(test_path, 'theta_test.npy'))
            except FileNotFoundError:
                log.error(
                    f'Missing test data in {test_path}. '
                    'Run cmass.infer.preprocess first.')
                continue

            log.info(f'Testing on {len(x_test)} examples')

            # ── load trained NLE ─────────────────────────────────────
            nle_path = join(exp_path, 'nle.pt')
            if not os.path.exists(nle_path):
                log.error(f'No NLE checkpoint at {nle_path}. '
                          'Run cmass.infer.train_nle first.')
                continue

            model = load_nle(nle_path, device=cfg.infer.device)
            log.info(f'Loaded NLE from {nle_path}')

            segments = _load_segments(exp_path)
            if segments is not None:
                log.info(f'Colouring by segments: {[s[0] for s in segments]}')

            # ── plot emulations ──────────────────────────────────────
            plot_emulations(
                model, x_test, theta_test,
                join(out_path, 'nle_emulations.png'),
                device=cfg.infer.device,
                segments=segments)


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

    log.info('Running NLE validation with config:\n' + OmegaConf.to_yaml(cfg))

    tracer = cfg.infer.tracer
    log.info(f'Running {tracer} NLE validation...')
    for exp in cfg.infer.experiments:
        save_path = join(model_dir, tracer, '+'.join(exp.summary))
        run_experiment(exp, cfg, save_path)


if __name__ == "__main__":
    main()
