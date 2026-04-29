"""
Median-coverage heatmap grids for SBI posterior estimators across summaries
and kmax values.

Produces one figure per parameter:
  - Rows = summaries, cols = kmaxes
  - Each cell is a 7x7 heatmap of median coverage over the (sigma_rad,
    sigma_tran) noise grid.

Figures are saved to FIG_DIR.
"""
import os
from os.path import join, exists

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

mpl.style.use('../matts_tests/style.mcstyle')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WDIR = '/work/hdd/bdne/maho3/cmass-ili'
BASEDIR = f'{WDIR}/quijote/meshed_hodz/models/galaxy'
TESTDIR_BASE = f'{WDIR}/quijote/nbody_hodz_gridnoise/models/galaxy'
SIM_TEST = 'quijote_nbody_hodz_gridnoise'
NOISE_GRID_PATH = f'{WDIR}/noise_priors/noisegrid.csv'
FIG_DIR = './figures'
os.makedirs(FIG_DIR, exist_ok=True)

z = 'z'
SUMMARY_NAMES = [
    f'{z}Pk0',
    f'{z}Pk0+{z}Pk2+{z}Pk4',
    f'{z}Pk0+{z}Pk2+{z}Pk4+{z}Bk0',
    f'{z}Pk0+{z}Pk2+{z}Pk4+{z}EqBk0',
]
KMAX_VALS = [0.1, 0.2, 0.3, 0.4]
PARAM_NAMES = [r'\Omega_m', r'\Omega_b', r'h', r'n_s', r'\sigma_8']
PARAM_IDXS = [0, 4]
N_NOISE = 49

noises = np.loadtxt(NOISE_GRID_PATH, delimiter=',')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def model_paths(s, kmax):
    kstr = f'kmin-0.0_kmax-{kmax:.1f}'
    train = join(BASEDIR, s, kstr)
    test = join(TESTDIR_BASE, s, kstr)
    return {
        'ood_samples': join(train, 'testing', SIM_TEST, 'posterior_samples.npy'),
        'theta_ood': join(test, 'theta_test.npy'),
        'noiseid_ood': join(test, 'noiseid_test.npy'),
    }


def simple(label):
    if isinstance(label, list):
        return [simple(l) for l in label]
    label = label.replace('nbar', r'$\bar{n}$')
    label = label.replace('zPk0+zPk2+zPk4', r'$zP_{0,2,4}$')
    label = label.replace('zPk0', r'$zP_{0}$')
    label = label.replace('zBk0', r'$zB_{0}$')
    label = label.replace('zEqBk0', r'$zEqB_{0}$')
    label = label.replace('zQk0', r'$zQ_{0}$')
    label = label.replace('+', ', ')
    return label


def median_coverage(samples, trues):
    """Fraction of trues below the posterior median."""
    medians = np.median(samples, axis=0)
    return (trues < medians).mean()


# ---------------------------------------------------------------------------
# Loader: per-(summary, kmax), compute heatmap for ALL params in one pass
# ---------------------------------------------------------------------------
def load_heatmaps(s, kmax, p_list):
    """Return dict[p] -> (7,7) median-coverage heatmap, or None."""
    paths = model_paths(s, kmax)
    if not all(exists(paths[k]) for k in paths):
        return None
    try:
        samples = np.load(paths['ood_samples'], mmap_mode='r')
        theta = np.load(paths['theta_ood'])
        noiseidx = np.load(paths['noiseid_ood'])[:, 0]
    except (OSError, ValueError, IndexError):
        return None

    out = {p: np.full((7, 7), np.nan) for p in p_list}

    for n in range(N_NOISE):
        idx = np.flatnonzero(noiseidx == n)
        if len(idx) == 0:
            continue
        # Pull the slice once per noise level, then index params from RAM
        s_ood = np.asarray(samples[:, idx, :])
        for p in p_list:
            out[p][divmod(n, 7)] = median_coverage(
                s_ood[:, :, p], theta[idx, p])

    return out


# ---------------------------------------------------------------------------
# Build cache once: (summary, kmax) -> {p: heatmap} or None
# ---------------------------------------------------------------------------
print('Loading all (summary, kmax) combinations...')
cache = {}
for s in tqdm(SUMMARY_NAMES):
    for kval in KMAX_VALS:
        cache[(s, kval)] = load_heatmaps(s, kval, PARAM_IDXS)


# ---------------------------------------------------------------------------
# Plot: Median-coverage heatmap grids
# ---------------------------------------------------------------------------
for p_idx in PARAM_IDXS:
    nrows, ncols = len(SUMMARY_NAMES), len(KMAX_VALS)
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(2.5 * ncols, 2.5 * nrows),
                            squeeze=False)
    im = None
    for r, s in enumerate(SUMMARY_NAMES):
        for c, kval in enumerate(KMAX_VALS):
            ax = axs[r, c]
            res = cache.get((s, kval))
            hm = res[p_idx] if (res is not None and p_idx in res) else None

            if hm is None or np.all(np.isnan(hm)):
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                im = ax.imshow(hm, vmin=0, vmax=1, cmap='RdBu', origin='upper')
                ax.set_xticks([0, 3, 6])
                ax.set_yticks([0, 3, 6])
                ax.set_xticklabels(
                    [f'{noises[i*7, 1]:.2f}' for i in [0, 3, 6]], fontsize=7)
                ax.set_yticklabels(
                    [f'{noises[j, 0]:.2f}' for j in [0, 3, 6]], fontsize=7)

            if r == 0:
                ax.set_title(f'k<{kval}', fontsize=10)
            if c == 0:
                ax.set_ylabel(simple(s) + '\n' + r'$\sigma_{\rm rad}$',
                              fontsize=9)
            if r == nrows - 1:
                ax.set_xlabel(r'$\sigma_{\rm tran}$', fontsize=9)

    if im is not None:
        fig.colorbar(
            im, ax=axs,
            label=f'Median coverage (${PARAM_NAMES[p_idx]}$)',
            fraction=0.02, pad=0.02)
    fig.suptitle(f'Median coverage — ${PARAM_NAMES[p_idx]}$', fontsize=14)
    fname = join(FIG_DIR, f'median_coverage_p{p_idx}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'Saved {fname}')

plt.show()
