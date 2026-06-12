"""
Out-of-distribution inference evaluation across noise configurations.

Replicates the 'Inference on different noise configurations' section of
matts_tests/test_noise_calibration.ipynb. Edit the CONFIG block below to
set paths, summaries, and kmin/kmax combinations, then run:

    python scripts/ood_noise_inference.py

Figures are saved to scripts/figures/{summary}_{kstr}/.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from os.path import join

# ── Configuration ─────────────────────────────────────────────────────────────

BASEDIR = '/work/hdd/bdne/maho3/cmass-ili/quijotelike/fastpm_charm6/models/galaxy'
TESTDIR = '/work/hdd/bdne/maho3/cmass-ili/quijote/nbody_hodz_gridnoise/models/galaxy'
NOISES_PATH = '/work/hdd/bdne/maho3/cmass-ili/noise_priors/noisegrid.csv'

SUMMARIES = [
    'zPk0+zPk2+zPk4',
    'zPk0+zPk2+zPk4+zBk0',
]

KMINMAX_PAIRS = [
    (0.0, 0.2),
    (0.0, 0.4),
]

# ──────────────────────────────────────────────────────────────────────────────

matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 13,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

PARAM_NAMES = [r'\Omega_m', r'\Omega_b', r'h', r'n_s', r'\sigma_8']
PARAM_INDICES = [0, 4]  # Omega_m and sigma_8


def marginal_coverage(samples: np.ndarray, trues: np.ndarray, nbins: int = 20):
    valid = np.isfinite(samples)
    num_samples = valid.sum(axis=0)
    num_data = trues.shape[0]
    ranks = ((samples < trues[None, :]) & valid).sum(axis=0) / num_samples
    bin_edges = np.linspace(0, 1, nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    coverage = (ranks[None, :] <= bin_centers[:, None]).mean(axis=1)
    error = np.sqrt(coverage * (1 - coverage) / num_data)
    return bin_centers, coverage, error


def median_coverage(samples: np.ndarray, trues: np.ndarray):
    medians = np.median(samples, axis=0)
    return (trues < medians).mean()


def load_data(basedir, testdir, s, kstr):
    """Load theta, noiseidx, OOD samples, and self-consistent samples."""
    sim = '_'.join(testdir.rstrip('/').split('/')[-4:-2])

    theta = np.load(join(testdir, s, kstr, 'theta_test.npy'))
    ids = np.load(join(testdir, s, kstr, 'ids_test.npy'))

    # Handle two naming conventions seen in the notebook
    noiseid_path = join(testdir, s, kstr, 'noiseid_test.npy')
    if not os.path.exists(noiseid_path):
        noiseid_path = join(testdir, s, kstr, 'noiseids_test.npy')
    noiseidx = np.load(noiseid_path)[:, 0]

    samples_path = join(basedir, s, kstr, 'testing',
                        sim, 'posterior_samples.npy')
    samples = np.load(samples_path)

    percs = np.percentile(samples, q=[50, 16, 84], axis=0)
    percs[1] = percs[0] - percs[1]
    percs[2] = percs[2] - percs[0]

    theta_self = np.load(join(basedir, s, kstr, 'theta_test.npy'))
    samples_self = np.load(join(basedir, s, kstr, 'posterior_samples.npy'))
    percs_self = np.percentile(samples_self, q=[50, 16, 84], axis=0)
    percs_self[1] = percs_self[0] - percs_self[1]
    percs_self[2] = percs_self[2] - percs_self[0]

    return theta, noiseidx, samples, percs, theta_self, samples_self, percs_self


def plot_single_noise_scatter(theta, percs, theta_self, percs_self,
                              noiseidx, noises, n, s, figdir, label):
    """True vs predicted scatter for two params at one noise index."""
    Nplot = min(50, len(np.argwhere(noiseidx == n).flatten()))
    idx = np.argwhere(noiseidx == n).flatten()
    np.random.seed(42)
    idx = np.random.choice(idx, Nplot)
    idx_self = np.random.choice(len(theta_self), Nplot)

    minmax = np.array([theta.min(axis=0), theta.max(axis=0)])

    f, axs = plt.subplots(1, 2, figsize=(10, 5))
    for i, j in enumerate(PARAM_INDICES):
        ax = axs[i]
        ax.plot(minmax[:, j], minmax[:, j], 'k--')
        ax.errorbar(theta[idx, j], percs[0][idx, j],
                    yerr=[percs[1][idx, j], percs[2][idx, j]],
                    fmt='o', label='OOD')
        ax.errorbar(theta_self[idx_self, j], percs_self[0][idx_self, j],
                    yerr=[percs_self[1][idx_self, j],
                          percs_self[2][idx_self, j]],
                    fmt='o', label='Self')
        ax.set(xlabel=f'True ${PARAM_NAMES[j]}$',
               ylabel=f'Predicted ${PARAM_NAMES[j]}$')
    axs[-1].legend()
    f.suptitle(
        f'{s}\n'
        f'$\\sigma_{{\\rm rad}}={noises[n,0]:.2f}$  '
        f'$\\sigma_{{\\rm tran}}={noises[n,1]:.2f}$'
    )
    plt.tight_layout()
    fname = join(figdir, f'scatter_n{n}.png')
    f.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(f)
    print(f'  Saved {fname}')


def plot_residuals(theta, percs, theta_self, percs_self,
                   noiseidx, noises, n, s, figdir, label):
    """Residual plots + histogram for two params at one noise index."""
    idx_all = np.argwhere(noiseidx == n).flatten()
    np.random.seed(42)
    idx_plot = np.random.choice(idx_all, min(100, len(idx_all)))
    idx_self_all = np.arange(len(theta_self))
    idx_self_plot = np.random.choice(
        idx_self_all, min(len(idx_all), len(idx_self_all)))

    fig, axs = plt.subplots(2, 2, figsize=(11, 8),
                            gridspec_kw={'width_ratios': [3, 1]},
                            sharey='row')
    for i, j in enumerate(PARAM_INDICES):
        residuals_all = percs[0][idx_all, j] - theta[idx_all, j]
        residuals_plot = percs[0][idx_plot, j] - theta[idx_plot, j]
        residuals_self_all = percs_self[0][idx_self_all,
                                           j] - theta_self[idx_self_all, j]
        residuals_self_plot = percs_self[0][idx_self_plot,
                                            j] - theta_self[idx_self_plot, j]

        ax = axs[i, 0]
        ax.axhline(0, c='k', ls='--')
        ax.errorbar(theta[idx_plot, j], residuals_plot,
                    yerr=[percs[1][idx_plot, j], percs[2][idx_plot, j]],
                    fmt='o', label='OOD', alpha=0.7)
        ax.errorbar(theta_self[idx_self_plot, j], residuals_self_plot,
                    yerr=[percs_self[1][idx_self_plot, j],
                          percs_self[2][idx_self_plot, j]],
                    fmt='s', label='Self', alpha=0.7)
        ax.set(xlabel=f'True ${PARAM_NAMES[j]}$',
               ylabel=f'Predicted - True ${PARAM_NAMES[j]}$')
        ax.legend()

        ax_hist = axs[i, 1]
        ax_hist.hist(residuals_all, bins=15, orientation='horizontal',
                     alpha=0.5, label='OOD', density=True)
        ax_hist.hist(residuals_self_all, bins=15, orientation='horizontal',
                     alpha=0.5, label='Self', density=True)
        ax_hist.axhline(0, c='k', ls='--')
        ax_hist.set_xlabel('Count')
        ax_hist.tick_params(labelleft=False)
        medcov = np.sum(residuals_all < 0) / len(residuals_all)
        medcov_self = np.sum(residuals_self_all < 0) / len(residuals_self_all)
        ax_hist.set_title(f'{medcov*100:.1f}% / {medcov_self*100:.1f}%')
        ax_hist.legend()

        ylim = 0.15 if i == 0 else 0.25
        ax.set_ylim(-ylim, ylim)

    fig.suptitle(
        f'{s}\n'
        f'$\\sigma_{{\\rm rad}}={noises[n,0]:.2f}$  '
        f'$\\sigma_{{\\rm tran}}={noises[n,1]:.2f}$'
    )
    plt.tight_layout()
    fname = join(figdir, f'residuals_n{n}.png')
    fig.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def plot_all_noise_true_vs_pred(theta, percs, noiseidx, noises, p, s, figdir, label):
    """7x7 grid of true vs predicted across all noise configurations."""
    n_noise = len(noises)
    ncols = int(np.round(np.sqrt(n_noise)))
    nrows = int(np.ceil(n_noise / ncols))

    minmax = np.array([theta.min(axis=0), theta.max(axis=0)])

    f, axs = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows),
                          sharex=True, sharey=True)
    axs = axs.flatten()
    for i in range(n_noise):
        idx = np.argwhere(noiseidx == i).flatten()
        if len(idx) == 0:
            axs[i].set_visible(False)
            continue
        idx = np.random.choice(idx, min(50, len(idx)))
        ax = axs[i]
        ax.plot(minmax[:, p], minmax[:, p], 'k--')
        ax.errorbar(theta[idx, p], percs[0][idx, p],
                    yerr=[percs[1][idx, p], percs[2][idx, p]],
                    fmt='o', ms=3)
        ax.set_title(
            f'$\\sigma_r={noises[i,0]:.2f}$ $\\sigma_t={noises[i,1]:.2f}$')
        if i % ncols == 0:
            ax.set_ylabel(f'Pred ${PARAM_NAMES[p]}$')
        if i // ncols == nrows - 1:
            ax.set_xlabel(f'True ${PARAM_NAMES[p]}$')

    for i in range(n_noise, len(axs)):
        axs[i].set_visible(False)

    f.suptitle(f'{s}  ${PARAM_NAMES[p]}$', fontsize=16)
    plt.tight_layout()
    pname = 'Om' if p == 0 else 's8'
    fname = join(figdir, f'pred_{pname}.png')
    f.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(f)
    print(f'  Saved {fname}')


def plot_coverage_grid(samples, theta, noiseidx, noises, samples_self, theta_self,
                       p, s, figdir, label):
    """7x7 grid of marginal coverage (PIT) curves across all noise configurations."""
    n_noise = len(noises)
    ncols = int(np.round(np.sqrt(n_noise)))
    nrows = int(np.ceil(n_noise / ncols))

    f, axs = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows),
                          sharex=True, sharey=True)
    axs = axs.flatten()
    for i in range(n_noise):
        ax = axs[i]
        idx = np.argwhere(noiseidx == i).flatten()
        if len(idx) == 0:
            ax.set_visible(False)
            continue
        idx_self = np.random.choice(len(theta_self), len(idx))

        _x, _y, _err = marginal_coverage(samples[:, idx, p], theta[idx, p])
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(_x, _y)
        ax.fill_between(_x, _y - _err, _y + _err, alpha=0.5, label='OOD')

        _x, _y, _err = marginal_coverage(
            samples_self[:, idx_self, p], theta_self[idx_self, p])
        ax.plot(_x, _y, alpha=0.7)
        ax.fill_between(_x, _y - _err, _y + _err, alpha=0.3, label='Self')

        ax.set_title(
            f'$\\sigma_r={noises[i,0]:.2f}$ $\\sigma_t={noises[i,1]:.2f}$')
        if i % ncols == 0:
            ax.set_ylabel(f'${PARAM_NAMES[p]}$ Percentiles')
        if i // ncols == nrows - 1:
            ax.set_xlabel(f'True ${PARAM_NAMES[p]}$ Percentiles')
        if i == 0:
            ax.legend()

    for i in range(n_noise, len(axs)):
        axs[i].set_visible(False)

    f.suptitle(f'{s}  ${PARAM_NAMES[p]}$', fontsize=16)
    plt.tight_layout()
    pname = 'Om' if p == 0 else 's8'
    fname = join(figdir, f'coverage_{pname}.png')
    f.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(f)
    print(f'  Saved {fname}')


def plot_median_coverage_heatmap(samples, theta, noiseidx, noises, s, figdir, label):
    """Heatmap of median coverage for Omega_m and sigma_8 over the noise grid."""
    sig_rad_unique = np.sort(np.unique(noises[:, 0]))
    sig_trans_unique = np.sort(np.unique(noises[:, 1]))
    n_rad = len(sig_rad_unique)
    n_trans = len(sig_trans_unique)
    n_noise = len(noises)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, p in zip(axes, PARAM_INDICES):
        heatmap = np.full((n_trans, n_rad), np.nan)
        for i in range(n_noise):
            ir = np.searchsorted(sig_rad_unique, noises[i, 0])
            it = np.searchsorted(sig_trans_unique, noises[i, 1])
            idx = np.argwhere(noiseidx == i).flatten()
            if len(idx) == 0:
                continue
            heatmap[it, ir] = median_coverage(samples[:, idx, p], theta[idx, p])

        im = ax.imshow(heatmap, vmin=0, vmax=1, cmap='RdBu', origin='upper')
        for i in range(n_trans):
            for j in range(n_rad):
                if not np.isnan(heatmap[i, j]):
                    ax.text(j, i, f'{heatmap[i,j]:.2f}',
                            ha='center', va='center')
        ax.set_xticks(range(n_rad))
        ax.set_xticklabels([f'{v:.2f}' for v in sig_rad_unique])
        ax.set_yticks(range(n_trans))
        ax.set_yticklabels([f'{v:.2f}' for v in sig_trans_unique])
        ax.set_xlabel(r'$\sigma_{\rm rad}$')
        ax.set_ylabel(r'$\sigma_{\rm tran}$')
        ax.set_title(f'${PARAM_NAMES[p]}$')
        plt.colorbar(im, ax=ax, label='Coverage at 0.5')

    fig.suptitle(f'{s}', fontsize=16)
    plt.tight_layout()
    fname = join(figdir, 'heatmap.png')
    fig.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')


def run(basedir, testdir, noises_path, summaries, kminmax_pairs):
    np.random.seed(42)
    figroot = join(os.path.dirname(__file__), 'figures')
    os.makedirs(figroot, exist_ok=True)

    noises = np.loadtxt(noises_path, delimiter=',')
    n_noise = len(noises)

    for s in summaries:
        for kmin, kmax in kminmax_pairs:
            kstr = f'kmin-{kmin:.1f}_kmax-{kmax:.1f}'
            label = f"{s.replace('+', '_')}_{kstr}"
            figdir = join(figroot, label)
            os.makedirs(figdir, exist_ok=True)

            print(f'\n=== {s}  {kstr} ===')

            try:
                (theta, noiseidx, samples, percs,
                 theta_self, samples_self, percs_self) = load_data(
                    basedir, testdir, s, kstr)
            except FileNotFoundError as e:
                print(f'  SKIP (missing file): {e}')
                continue

            # Pick a representative noise index near the middle of the grid
            n_rep = n_noise // 2

            plot_single_noise_scatter(
                theta, percs, theta_self, percs_self,
                noiseidx, noises, n_rep, s, figdir, label)

            plot_residuals(
                theta, percs, theta_self, percs_self,
                noiseidx, noises, n_rep, s, figdir, label)

            for p in PARAM_INDICES:
                plot_all_noise_true_vs_pred(
                    theta, percs, noiseidx, noises, p, s, figdir, label)

                plot_coverage_grid(
                    samples, theta, noiseidx, noises,
                    samples_self, theta_self,
                    p, s, figdir, label)

            plot_median_coverage_heatmap(
                samples, theta, noiseidx, noises, s, figdir, label)


if __name__ == '__main__':
    run(BASEDIR, TESTDIR, NOISES_PATH, SUMMARIES, KMINMAX_PAIRS)
