"""
Model scaling diagnostics: kmax and feature-length sweeps.

Replicates the 'Compare models' section of matts_tests/test_toy_noised.ipynb.
Generates two sets of plots:
  1. kmax scaling   — fixed summary, increasing kmax
  2. feature scaling — fixed kmax, increasing feature length (more summary types)

Edit the CONFIG block below, then run:

    python scripts/model_scaling_diagnostics.py

Figures are saved to scripts/figures/model_scaling/.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import optuna
from os.path import join, exists

# ── Configuration ─────────────────────────────────────────────────────────────

WDIR = '/work/hdd/bdne/maho3/cmass-ili'
NBODY = 'quijotelike'
SIM = 'fastpm_charm5'
TRACER = 'galaxy'
Z = 'z'

# kmax sweep: fix one summary, vary kmax
KMAX_SUMMARY = f'{Z}Pk0+{Z}Pk2+{Z}Pk4'
KMAX_VALUES = [0.1, 0.2, 0.3, 0.4]

# feature sweep: fix kmax, vary summary complexity
FEAT_KMAX = 0.4
FEAT_SUMMARIES = [
    f'{Z}Pk0',
    f'{Z}Pk0+{Z}Pk2+{Z}Pk4',
    f'{Z}Pk0+{Z}Pk2+{Z}Pk4+{Z}Bk0',
    f'{Z}Pk0+{Z}Pk2+{Z}Pk4+{Z}EqBk0',
]

# Fiducial cosmology for filtering test points
THETAFID = np.array([0.3, 0.5, 0.7, 1.0, 0.8])
NBAR_LO = np.log10(1.0e-4)
NBAR_HI = np.log10(5.0e-4)
NBAR_RTOL = 0.1  # relative tolerance on theta for fiducial match

PARAM_NAMES = [r'\Omega_m', r'\Omega_b', r'h', r'n_s', r'\sigma_8']
PARAM_IDXS = [0, 4]

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


# ── Label helpers ──────────────────────────────────────────────────────────────

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


# ── Data helpers ───────────────────────────────────────────────────────────────

def modeldir(nbody, sim, tracer, summary, kmax, wdir=WDIR):
    return join(wdir, nbody, sim, 'models', tracer,
                summary, f'kmin-0.0_kmax-{kmax:.1f}')


def load_samples(mdir):
    """Return (samples, theta, nbar) or raise FileNotFoundError."""
    samples = np.load(join(mdir, 'posterior_samples.npy'))
    theta = np.load(join(mdir, 'theta_test.npy'))
    nbar_path = join(mdir, 'nbar_test.npy')
    if exists(nbar_path):
        nbar = np.load(nbar_path)
    else:
        x = np.load(join(mdir, 'x_test.npy'))
        nbar = x[:, -1]
    if nbar.ndim > 1:
        nbar = nbar.mean(axis=1)
    return samples, theta, nbar


def feature_length(mdir):
    x = np.load(join(mdir, 'x_test.npy'))
    return x.shape[-1]


def fiducial_stdev(mdir):
    """Median posterior stdev near fiducial cosmology; returns (stdev array, count) or None."""
    if not exists(join(mdir, 'posterior_samples.npy')):
        return None
    try:
        samples, theta, nbar = load_samples(mdir)
    except (OSError, ValueError):
        return None
    mask = np.all(np.isclose(theta[:, PARAM_IDXS],
                              THETAFID[PARAM_IDXS], rtol=NBAR_RTOL), axis=1)
    mask &= (nbar > NBAR_LO) & (nbar < NBAR_HI)
    if not mask.any():
        return None
    return np.std(samples[:, mask], axis=0)


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_optuna_history(modeldirs, labels, title, figdir, fname='optuna_history.png'):
    """Best validation log-prob vs Optuna trial number for each model."""
    f, ax = plt.subplots(figsize=(8, 4))
    for i, (mdir, lab) in enumerate(zip(modeldirs, labels)):
        db = join(mdir, 'optuna_study.db')
        if not exists(db):
            print(f'  SKIP optuna (no db): {mdir}')
            continue
        study_name = mdir.rstrip('/').split('/')[-2]
        try:
            study = optuna.load_study(study_name=study_name,
                                      storage=f'sqlite:///{db}')
        except Exception as e:
            print(f'  SKIP optuna ({e}): {mdir}')
            continue
        trials = [t for t in study.trials if t.value is not None]
        if not trials:
            continue
        nums = [t.number for t in trials]
        vals = [t.value for t in trials]
        best = np.maximum.accumulate(vals)
        ax.plot(nums, best, label=lab, color=f'C{i}')

    ax.set(xlabel='Optuna trial number', ylabel='Best validation_log_prob')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True)
    ax.set_title(title)
    plt.tight_layout()
    fpath = join(figdir, fname)
    f.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close(f)
    print(f'  Saved {fpath}')


def plot_stdev_vs_theta(modeldirs, labels, title, figdir, fname='stdev_vs_theta.png'):
    """Median posterior stdev in bins of true parameter value."""
    Nbins = 4
    f, axs = plt.subplots(1, 2, figsize=(10, 5))
    for i, (mdir, lab) in enumerate(zip(modeldirs, labels)):
        if not exists(join(mdir, 'posterior_samples.npy')):
            continue
        try:
            samples, theta, _ = load_samples(mdir)
        except (OSError, ValueError):
            continue
        stdev = samples.std(axis=0)
        off = (i - (len(modeldirs) - 1) / 2) * 0.02
        fmt = ['o', 's', '*', 'D', '^'][i % 5]
        for j, p in enumerate(PARAM_IDXS):
            ax = axs[j]
            lo, hi = theta[:, p].min(), theta[:, p].max()
            edges = np.linspace(lo, hi, Nbins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ys = []
            for k in range(Nbins):
                mask = (theta[:, p] >= edges[k]) & (theta[:, p] < edges[k + 1])
                ys.append(np.percentile(stdev[mask, p], [50, 16, 84]))
            ys = np.array(ys)
            ax.errorbar(centers + off * (hi - lo), ys[:, 0],
                        yerr=[ys[:, 0] - ys[:, 1], ys[:, 2] - ys[:, 0]],
                        fmt=fmt, color=f'C{i}', label=lab, capsize=3)
            ax.set(xlabel=f'True ${PARAM_NAMES[p]}$',
                   ylabel=fr'$\Delta {PARAM_NAMES[p]}$')
            ax.set_ylim(0)
            ax.grid(True)

    axs[1].legend(fontsize=8, ncol=2, loc='upper right')
    f.suptitle(title)
    plt.tight_layout()
    fpath = join(figdir, fname)
    f.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close(f)
    print(f'  Saved {fpath}')


def plot_aggregate_calibration(modeldirs, labels, title, figdir,
                                fname='calibration.png'):
    """Fraction of test points below median and within 16–84th percentile."""
    import matplotlib.lines as mlines
    color_med = 'tab:blue'
    color_68 = 'tab:orange'

    f, axs = plt.subplots(1, 2, figsize=(8, 4))
    for i, (mdir, _) in enumerate(zip(modeldirs, labels)):
        if not exists(join(mdir, 'posterior_samples.npy')):
            continue
        try:
            samples, theta, _ = load_samples(mdir)
        except (OSError, ValueError):
            continue
        p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)
        N = len(theta)
        Nhods = 5
        err_med = np.sqrt(0.25 * Nhods / N)
        err_68 = np.sqrt(0.68 * 0.32 * Nhods / N)
        for j, p in enumerate(PARAM_IDXS):
            ax = axs[j]
            y_med = np.mean(theta[:, p] < p50[:, p])
            y_68 = np.mean((theta[:, p] >= p16[:, p]) & (theta[:, p] <= p84[:, p]))
            ax.errorbar(i, y_med, yerr=err_med,
                        marker='o', color=color_med, linestyle='none')
            ax.errorbar(i, y_68, yerr=err_68,
                        marker='^', color=color_68, linestyle='none')
            ax.set_title(f'${PARAM_NAMES[p]}$')
            ax.set_ylabel('True fraction in range')
            if i == 0:
                ax.axhline(0.50, color=color_med, ls=':', alpha=0.6)
                ax.axhline(0.68, color=color_68, ls=':', alpha=0.6)

    handles = [
        mlines.Line2D([], [], color=color_med, marker='o',
                      linestyle='none', label='Median (50%)'),
        mlines.Line2D([], [], color=color_68, marker='^',
                      linestyle='none', label='16–84th (68%)'),
    ]
    axs[0].legend(handles=handles, fontsize=9)
    for ax in axs:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(simple(labels), rotation=45, ha='right')
        ax.grid(True, axis='y')

    f.suptitle(title)
    plt.tight_layout()
    fpath = join(figdir, fname)
    f.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close(f)
    print(f'  Saved {fpath}')


def plot_fiducial_stdev_bar(modeldirs, labels, title, figdir,
                             fname='fiducial_stdev.png'):
    """Median ± 16/84th stdev at fiducial cosmology, categorical x-axis."""
    f, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
    valid_labels = []
    x_idx = 0
    for i, (mdir, lab) in enumerate(zip(modeldirs, labels)):
        stdev = fiducial_stdev(mdir)
        if stdev is None:
            print(f'  SKIP fiducial_stdev (no data): {mdir}')
            continue
        fmt = ['o', 's', '*', 'D', '^'][i % 5]
        for j, p in enumerate(PARAM_IDXS):
            ax = axs[j]
            perc = np.percentile(stdev[:, p], [50, 16, 84])
            ax.errorbar(x_idx, perc[0],
                        yerr=[[perc[0] - perc[1]], [perc[2] - perc[0]]],
                        fmt=fmt, color=f'C{i}', capsize=4)
            ax.set_ylabel(fr'$\Delta {PARAM_NAMES[p]}$')
            ax.set_ylim(0)
            ax.grid(True, axis='y')
        valid_labels.append(lab)
        x_idx += 1

    for ax in axs:
        ax.set_xticks(range(len(valid_labels)))
        ax.set_xticklabels(simple(valid_labels), rotation=45, ha='right')

    f.suptitle(title)
    plt.tight_layout()
    fpath = join(figdir, fname)
    f.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close(f)
    print(f'  Saved {fpath}')


def plot_kmax_scaling(summaries, kmax_values, nbody, sim, tracer,
                      title, figdir, fname='kmax_scaling.png'):
    """Fiducial stdev vs kmax for each summary type (line plot)."""
    markers = ['o', 's', '*', 'D', '^', 'v']
    kmax_arr = np.array(kmax_values)
    kspan = kmax_arr.max() - kmax_arr.min()

    f, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    for g, s in enumerate(summaries):
        off = (g - (len(summaries) - 1) / 2) * 0.01 * kspan
        xs, percs_list = [], [[], []]
        for k in kmax_values:
            mdir = modeldir(nbody, sim, tracer, s, k)
            stdev = fiducial_stdev(mdir)
            if stdev is None:
                print(f'  SKIP kmax_scaling (no data): {mdir}')
                continue
            xs.append(k + off)
            for j, p in enumerate(PARAM_IDXS):
                percs_list[j].append(
                    np.percentile(stdev[:, p], [50, 16, 84]))
        if not xs:
            continue
        for j, p in enumerate(PARAM_IDXS):
            perc = np.array(percs_list[j])
            axs[j].errorbar(
                xs, perc[:, 0],
                yerr=[perc[:, 0] - perc[:, 1], perc[:, 2] - perc[:, 0]],
                label=simple(s), color=f'C{g}',
                marker=markers[g % len(markers)], linestyle='-', capsize=3)
            axs[j].set(xlabel=r'$k_{\max}\ [h/\mathrm{Mpc}]$',
                       ylabel=fr'$\Delta {PARAM_NAMES[p]}$',
                       ylim=(0, None))
            axs[j].grid(True)

    axs[1].legend(fontsize=9, loc='upper right', ncol=1)
    f.suptitle(title)
    plt.tight_layout()
    fpath = join(figdir, fname)
    f.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close(f)
    print(f'  Saved {fpath}')


def plot_feature_length_scaling(summaries, kmax, nbody, sim, tracer,
                                 title, figdir, fname='feature_length_scaling.png'):
    """Fiducial stdev vs feature vector length (x_len) at fixed kmax."""
    f, axs = plt.subplots(1, 2, figsize=(10, 5))
    xlens, stdevs = [], []
    labels_valid = []
    for i, s in enumerate(summaries):
        mdir = modeldir(nbody, sim, tracer, s, kmax)
        if not exists(join(mdir, 'x_test.npy')):
            print(f'  SKIP feature_scaling (no x_test): {mdir}')
            continue
        stdev = fiducial_stdev(mdir)
        if stdev is None:
            print(f'  SKIP feature_scaling (no stdev): {mdir}')
            continue
        xlens.append(feature_length(mdir))
        stdevs.append(stdev)
        labels_valid.append(s)

    for j, p in enumerate(PARAM_IDXS):
        ax = axs[j]
        for i, (xl, stdev, lab) in enumerate(zip(xlens, stdevs, labels_valid)):
            perc = np.percentile(stdev[:, p], [50, 16, 84])
            ax.errorbar(xl, perc[0],
                        yerr=[[perc[0] - perc[1]], [perc[2] - perc[0]]],
                        fmt='o', color=f'C{i}', label=simple(lab), capsize=4)
        ax.set(xlabel='Feature vector length',
               ylabel=fr'$\Delta {PARAM_NAMES[p]}$',
               ylim=(0, None))
        ax.grid(True)

    axs[1].legend(fontsize=9, loc='upper right')
    f.suptitle(title)
    plt.tight_layout()
    fpath = join(figdir, fname)
    f.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close(f)
    print(f'  Saved {fpath}')


# ── Main ───────────────────────────────────────────────────────────────────────

def run(wdir, nbody, sim, tracer,
        kmax_summary, kmax_values,
        feat_kmax, feat_summaries):
    np.random.seed(42)
    figroot = join(os.path.dirname(os.path.abspath(__file__)),
                   'figures', 'model_scaling')
    os.makedirs(figroot, exist_ok=True)

    base_title = f'{nbody} / {sim} / {tracer}'

    # ── 1. kmax sweep ──────────────────────────────────────────────────────────
    print('\n=== kmax sweep ===')
    kmax_dir = join(figroot, 'kmax_sweep')
    os.makedirs(kmax_dir, exist_ok=True)

    kmax_mdirs = [modeldir(nbody, sim, tracer, kmax_summary, k, wdir)
                  for k in kmax_values]
    kmax_labels = [f'k<{k}' for k in kmax_values]
    kmax_title = f'{base_title}\n{simple(kmax_summary)}: varying kmax'

    plot_optuna_history(kmax_mdirs, kmax_labels, kmax_title, kmax_dir)
    plot_stdev_vs_theta(kmax_mdirs, kmax_labels, kmax_title, kmax_dir)
    plot_aggregate_calibration(kmax_mdirs, kmax_labels, kmax_title, kmax_dir)
    plot_fiducial_stdev_bar(kmax_mdirs, kmax_labels, kmax_title, kmax_dir)

    # Also make the line-plot version of kmax scaling for all feat_summaries
    plot_kmax_scaling(feat_summaries, kmax_values, nbody, sim, tracer,
                      f'{base_title}\nkmax scaling', kmax_dir)

    # ── 2. Feature-length sweep ────────────────────────────────────────────────
    print('\n=== feature-length sweep ===')
    feat_dir = join(figroot, 'feature_sweep')
    os.makedirs(feat_dir, exist_ok=True)

    feat_mdirs = [modeldir(nbody, sim, tracer, s, feat_kmax, wdir)
                  for s in feat_summaries]
    feat_labels = [s for s in feat_summaries]
    feat_title = f'{base_title}\nkmax={feat_kmax}: varying summary'

    plot_optuna_history(feat_mdirs, feat_labels, feat_title, feat_dir)
    plot_stdev_vs_theta(feat_mdirs, feat_labels, feat_title, feat_dir)
    plot_aggregate_calibration(feat_mdirs, feat_labels, feat_title, feat_dir)
    plot_fiducial_stdev_bar(feat_mdirs, feat_labels, feat_title, feat_dir)
    plot_feature_length_scaling(feat_summaries, feat_kmax, nbody, sim, tracer,
                                 feat_title, feat_dir)


if __name__ == '__main__':
    run(
        wdir=WDIR,
        nbody=NBODY,
        sim=SIM,
        tracer=TRACER,
        kmax_summary=KMAX_SUMMARY,
        kmax_values=KMAX_VALUES,
        feat_kmax=FEAT_KMAX,
        feat_summaries=FEAT_SUMMARIES,
    )