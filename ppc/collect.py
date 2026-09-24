"""
Step 4 of the posterior predictive check (PPC) campaign.

Collects the per-draw diagnostics into the deliverable arrays and plots the
posterior predictive bands against the observed data vector.

    <out>/theta_ppc.npy   (Ndraw, 17)   theta actually simulated
    <out>/x_ppc.npy       (Ndraw, 117)  the INFERENCE blocks only, in training
                                        x ordering -- the deliverable
    <out>/x_ppc_all.npz   every plotted summary block, inference or not
    <out>/plots/ppc_bands.png, ppc_corner.png, ppc_logprob.png

Rows align: x_ppc[i] is the data vector simulated at theta_ppc[i]. Draws whose
diagnostics are missing or whose recorded parameters disagree with the draw are
dropped from BOTH arrays together and marked in manifest.tsv.

The plot covers ALL available z-space summaries, not just the ones the posterior
was conditioned on. Held-out summaries (zBk0, zEqBk0, zSqBk0, ...) carry no
weight in the inference, so disagreement there is informative: it says the
forward model reproduces the constrained statistics but not the unconstrained
ones. Panel titles mark which is which.

For summaries outside the inference set there is no x_obs in the trained
experiment, so the observed vector is recomputed from the observed lhid's own
diagnostics file -- located by matching its recorded HOD parameters to
theta_obs, not by hardcoding a filename. For an out-of-distribution campaign
(ppc/draw.py --testing_suite) that sim lives in the testing suite, whose box
need not match the training suite's, so its own config.yaml sets the L-N dir.

k-grid caveat: the training summaries predate ltu-cmass ba2334f (2026-07-14,
pylians -> pypower for periodic-box P(k)), so training P(k) sits on a uniform
0.01 grid and these PPC runs sit on pypower's effective-k grid. Both give 39
bins in k <= 0.4. Bk is unaffected. Per instruction this is left as-is and
everything is plotted against the PPC (pypower) k, with residuals taken
element-wise -- matching how x_ppc and x_obs are used downstream.

OOD p-value (plots/ppc_pvalue.png, ppc_pvalues.tsv): tests H0 "x_obs is a
draw from the posterior predictive the PPC ensemble samples". Per block, and
for the concatenated inference / held-out vectors, features are standardized
by the PPC draws, projected on their top-k PCs (N ~ 100 draws cannot support a
full covariance), and scored by Mahalanobis d^2. p is the leave-one-out rank
of d^2_obs among the draws (floored at 1/(N+1)); a Hotelling-F p extrapolates
past that floor under a Gaussian assumption. A separate LOO p scores the
deviation orthogonal to the top-k PCs. --pvalue_only rebuilds this figure from
an existing x_ppc_all.npz without touching the per-draw sims.
"""

import argparse
import os
import shutil
from os.path import join, exists
import numpy as np
import h5py
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt   # noqa: E402

from cmass.infer.loaders import (      # noqa: E402
    preprocess_Pk, preprocess_Bk, load_Pk, load_Bk, load_lc_Pk, load_lc_Bk,
    _is_in_kminmax, _get_Bk_mask)
from cmass.infer.resim import (        # noqa: E402
    load_pool, plot_logprob, plot_corner, batched_log_prob)
from cmass.infer.tools import resolve_kmax   # noqa: E402
from ppc.layout import (                   # noqa: E402
    ExpPath, WDIR, N_COSMO, N_NOISE, fmt_kmax, sim_subdir)

PPC = join(WDIR, 'ppc/abacuslike_fastpm_charm7_cosmoHOD_reparam',
           'zPk0+zPk2+zPk4_kmin-0.0_kmax-0.4/obs01880')
AF = 0.666660000066666           # analysis snapshot (a), key '0.666660'
TAGS = ('Eq', 'Sq', 'Ss', 'Is', '')

# Entity -> colour, fixed. Observed is ink, the PPC ensemble is one hue, the
# training pool is recessive fill. Separated by lightness as well as hue, so it
# survives CVD and greyscale print.
C_OBS, C_PPC, C_POOL = 'k', 'C0', '0.85'


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ppc_dir', default=PPC)
    p.add_argument('--sim_sub', default=None,
                   help='per-draw sim subdir (default: fastpm/L<L>-N<N> from '
                        'the experiment config)')
    p.add_argument('--hod_seed', type=int, default=1,
                   help='which hodNNNNN[_augNNNNN].h5 to read per draw')
    p.add_argument('--aug_seed', type=int, default=1,
                   help='lightcone augmentation seed of the file to read')
    p.add_argument('--atol', type=float, default=1e-5,
                   help='tolerance when checking recorded vs drawn params')
    p.add_argument('--summaries', default=None,
                   help='comma-separated blocks to plot. Default: the '
                        'inference blocks plus every other available z-space '
                        'summary')
    p.add_argument('--bk_kmax', type=float, default=None,
                   help='kmax for bispectrum blocks (default: same as the '
                        'experiment k-cut)')
    p.add_argument('--obs_dir', default=None,
                   help='per-lhid dir of the observed sim. Default: derived '
                        'from exp_path and id_obs')
    p.add_argument('--no_plot', action='store_true')
    p.add_argument('--n_post', type=int, default=10000,
                   help='direct posterior samples for the theta-space plots')
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--device', default='cpu')
    p.add_argument('--no_theta_plots', action='store_true')
    p.add_argument('--n_pca', type=int, default=10,
                   help='PCs kept for the OOD Mahalanobis p-value')
    p.add_argument('--no_pvalue', action='store_true')
    p.add_argument('--pvalue_only', action='store_true',
                   help='only remake the p-value figure from x_ppc_all.npz')
    return p


def actual_nnets(exp_path, nnets):
    """How many nets load_ensemble will really use.

    cfg.infer.Nnets is what was *requested*; load_ensemble silently skips top
    trials whose posterior.pkl is missing. Reporting the requested count in a
    figure title would overstate the ensemble, so resolve it the same way
    load_ensemble does -- without paying to unpickle anything.
    """
    import optuna
    from cmass.infer.tools import select_top_trials, study_name_from_path
    db = join(exp_path, 'optuna_study.db')
    if not exists(db):
        return None
    try:
        study = optuna.load_study(storage=f'sqlite:///{db}',
                                  study_name=study_name_from_path(exp_path))
        top = select_top_trials(study, nnets)
    except Exception:
        return None
    return sum(exists(join(exp_path, 'nets', f'net-{t.number}',
                           'posterior.pkl')) for t in top)


def split_tag(summ):
    """'zEqBk0' -> ('zBk0', 'Eq'), replicating run_preprocessing's loop."""
    for tag in TAGS:
        if tag in summ:
            return summ.replace(tag, ''), tag
    return summ, ''


def load_summ(diagfile, lightcone=False):
    """A lightcone h5 is flat at the root and already in redshift space, so it
    needs its own loaders and its keys carry no 'z' prefix."""
    s = {}
    if lightcone:
        s.update(load_lc_Pk(diagfile))
        s.update(load_lc_Bk(diagfile))
    else:
        s.update(load_Pk(diagfile, AF))
        s.update(load_Bk(diagfile, AF))
    return s


def preprocess_block(summ, data, cfg, kmin, kmax, bk_kmax):
    """One summary block, preprocessed exactly as run_preprocessing does."""
    base, tag = split_tag(summ)
    norm_key = base[:-1] + '0'
    is_bk = ('Bk' in base) or ('Qk' in base)
    skmax = (bk_kmax if (is_bk and bk_kmax is not None)
             else resolve_kmax(kmax, summ))
    norm = None if '0' in base else data[norm_key]
    if is_bk:
        x = preprocess_Bk(data[base], kmin=kmin, kmax=skmax, norm=norm,
                          mode=tag, correct_shot=cfg.infer.correct_shot)
    else:
        x = preprocess_Pk(data[base], kmin=kmin, kmax=skmax, norm=norm,
                          correct_shot=cfg.infer.correct_shot,
                          loglinear_start_idx=cfg.infer.loglinear_start_idx)
    return x


def block_axis(summ, kdata, kmin, kmax, bk_kmax):
    """x values and axis label for a block.

    P(k) blocks and equilateral bispectra have a single k per feature, so they
    get a real k axis. General triangle configurations do not -- three k's per
    point -- so those fall back to a triangle index.
    """
    base, tag = split_tag(summ)
    is_bk = ('Bk' in base) or ('Qk' in base)
    skmax = (bk_kmax if (is_bk and bk_kmax is not None)
             else resolve_kmax(kmax, summ))
    if not is_bk:
        k = np.asarray(kdata)
        return k[_is_in_kminmax(k, kmin, skmax)], r'$k$ [$h$/Mpc]', skmax
    k123 = np.asarray(kdata)
    mask = _get_Bk_mask(k123, kmin, skmax, equilateral=(tag == 'Eq'),
                        squeezed=(tag == 'Sq'), subsampled=(tag == 'Ss'),
                        isoceles=(tag == 'Is'))
    if tag == 'Eq':      # k1 == k2 == k3, so a k axis is meaningful
        return k123[0][mask], r'$k$ [$h$/Mpc]', skmax
    return np.arange(int(mask.sum())), 'triangle index', skmax


def ylabel_for(summ):
    base, _ = split_tag(summ)
    if '0' in base:
        stat = 'B' if 'Bk' in base else ('Q' if 'Qk' in base else 'P')
        return rf'signed $\log_{{10}} {stat}_0$'
    ell = base[-1]
    stat = 'B' if 'Bk' in base else ('Q' if 'Qk' in base else 'P')
    return rf'${stat}_{ell}/{stat}_0$'


def find_obs_diag(diagdir, theta_obs, names, atol):
    """The observed lhid has several HOD realizations; pick the one whose
    recorded HOD parameters are theta_obs. Identifying it by content rather
    than filename keeps this correct if the pool ordering ever changes."""
    want = theta_obs[N_COSMO:-N_NOISE]
    hodnames = list(names[N_COSMO:-N_NOISE])
    for fn in sorted(os.listdir(diagdir)):
        path = join(diagdir, fn)
        try:
            with h5py.File(path, 'r') as f:
                if [str(s) for s in f.attrs['HOD_names']] != hodnames:
                    continue
                if not np.allclose(np.asarray(f.attrs['HOD_params'], float),
                                   want, atol=atol):
                    continue
                noise = np.array([f.attrs['noise_radial'],
                                  f.attrs['noise_transverse']], float)
                if not np.allclose(noise, theta_obs[-N_NOISE:], atol=atol):
                    continue
        except (OSError, KeyError):
            continue
        return path
    raise SystemExit(
        f'No diagnostics file in {diagdir} matches theta_obs. Cannot '
        'reconstruct the observed vector for held-out summaries.')


def check_draw(diagfile, theta, names, atol):
    """Confirm the sim actually used the parameters we drew (TODO.md §7.6)."""
    problems = []
    with h5py.File(diagfile, 'r') as f:
        cosmo = np.asarray(f.attrs['cosmo_params'], dtype=float)
        hod = np.asarray(f.attrs['HOD_params'], dtype=float)
        hodnames = [str(s) for s in f.attrs['HOD_names']]
        noise = np.array([f.attrs['noise_radial'],
                          f.attrs['noise_transverse']], dtype=float)
    if not np.allclose(cosmo, theta[:N_COSMO], atol=atol):
        problems.append(f'cosmo {cosmo} != {theta[:N_COSMO]}')
    if hodnames != list(names[N_COSMO:-N_NOISE]):
        problems.append(f'HOD name order {hodnames}')
    elif not np.allclose(hod, theta[N_COSMO:-N_NOISE], atol=atol):
        problems.append(f'HOD {hod} != {theta[N_COSMO:-N_NOISE]}')
    if not np.allclose(noise, theta[-N_NOISE:], atol=atol):
        problems.append(f'noise {noise} != {theta[-N_NOISE:]}')
    return problems


def plot_bands(blocks, title, out_path, ncols=4):
    """Value + residual panels per summary block.

    Two rows per group of columns: the vector itself with 68/95 bands, and the
    residual against the observed. Held-out blocks are flagged in the panel
    title (text, never colour alone).
    """
    n = len(blocks)
    ncols = min(ncols, n)
    ngroup = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(2 * ngroup, ncols, squeeze=False,
                            figsize=(4.3 * ncols, 3.1 * 2 * ngroup))
    q = [2.5, 16, 50, 84, 97.5]
    for idx, b in enumerate(blocks):
        g, c = divmod(idx, ncols)
        ax, axr = axs[2 * g, c], axs[2 * g + 1, c]
        x, obs, X = b['x'], b['obs'], b['x_ppc']
        qs = np.percentile(X, q, axis=0)

        if b['pool'] is not None:
            lo, hi = np.percentile(b['pool'], [2.5, 97.5], axis=0)
            ax.fill_between(x, lo, hi, color=C_POOL, label='training pool 95%')
        ax.fill_between(x, qs[0], qs[4], color=C_PPC, alpha=0.25,
                        label='PPC 95%')
        ax.fill_between(x, qs[1], qs[3], color=C_PPC, alpha=0.45,
                        label='PPC 68%')
        ax.plot(x, qs[2], color=C_PPC, lw=2, label='PPC median')
        ax.plot(x, obs, color=C_OBS, lw=2, label='observed')
        used = 'inference' if b['used'] else 'held out'
        ax.set_title(f"{b['name']}  ({used}, {b['n']} bins)", fontsize=10)
        ax.set_ylabel(b['ylabel'], fontsize=9)

        axr.fill_between(x, qs[0] - obs, qs[4] - obs, color=C_PPC, alpha=0.25)
        axr.fill_between(x, qs[1] - obs, qs[3] - obs, color=C_PPC, alpha=0.45)
        axr.plot(x, qs[2] - obs, color=C_PPC, lw=2)
        axr.axhline(0, color=C_OBS, lw=2)
        axr.set_xlabel(b['xlabel'], fontsize=9)
        axr.set_ylabel('PPC $-$ observed', fontsize=9)

        for a in (ax, axr):
            a.tick_params(labelsize=8)
            a.grid(alpha=0.25, lw=0.5)
            a.set_axisbelow(True)
            for side in ('top', 'right'):
                a.spines[side].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7.5, loc='best', framealpha=0.9)

    for idx in range(n, ngroup * ncols):       # blank unused panels
        g, c = divmod(idx, ncols)
        axs[2 * g, c].axis('off')
        axs[2 * g + 1, c].axis('off')

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _pca_d2(Xfit, Y, kmax):
    """Mahalanobis d^2 of rows Y in the top-1..kmax PCs of standardized Xfit.

    Returns d2 and out-of-subspace residual |b|^2 - |proj|^2, each (nY, kmax)
    with column j for k = j+1, plus the PC projections and variances. One SVD
    serves every k.
    """
    mu, sd = Xfit.mean(0), Xfit.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.)
    A, B = (Xfit - mu) / sd, (Y - mu) / sd
    _, S, Vt = np.linalg.svd(A, full_matrices=False)
    kmax = min(kmax, len(S))
    P = B @ Vt[:kmax].T
    var = S[:kmax]**2 / (len(A) - 1)
    d2 = np.cumsum(P**2 / var, axis=-1)
    resid = (B**2).sum(-1)[:, None] - np.cumsum(P**2, axis=-1)
    return d2, resid, P, var


def ppc_pvalues(X, obs, kmax):
    """p-values of obs against the PPC draws X (N, D), for k = 1..kmax.

    Every draw is scored against a PCA/covariance fit to the other N-1, so it
    is exchangeable with obs under H0 and the rank p is exact up to the 1/(N+1)
    floor. p_F is Hotelling's T^2 = d^2 N/(N+1) ~ k(N-1)/(N-k) F(k, N-k), the
    finite-N replacement for chi^2_k.
    """
    from scipy import stats
    N, D = X.shape
    kmax = min(kmax, D, N - 2)
    d2_obs, r_obs, _, _ = _pca_d2(X, obs[None], kmax)
    loo = [_pca_d2(np.delete(X, i, 0), X[i:i + 1], kmax)[:2]
           for i in range(N)]
    d2_loo = np.concatenate([d for d, _ in loo])
    r_loo = np.concatenate([r for _, r in loo])
    ks = np.arange(1, kmax + 1)
    p_emp = (1 + (d2_loo >= d2_obs).sum(0)) / (N + 1)
    p_res = (1 + (r_loo >= r_obs).sum(0)) / (N + 1)
    p_res = np.where(ks < D, p_res, np.nan)      # no residual when k = D
    fstat = d2_obs[0] * N / (N + 1) * (N - ks) / (ks * (N - 1))
    p_F = stats.f.sf(fstat, ks, N - ks)
    return dict(ks=ks, N=N, D=D, d2_obs=d2_obs[0], d2_loo=d2_loo,
                p_emp=p_emp, p_F=p_F, p_res=p_res)


def pvalue_groups(blocks):
    """Each block, plus the concatenated inference and held-out vectors, in
    display order: inference first, each section closed by its aggregate."""
    groups = []
    for used, agg in ((True, 'inference (all)'), (False, 'held out (all)')):
        sel = [b for b in blocks if b['used'] == used]
        groups += [(b['name'], used, b['x_ppc'], b['obs']) for b in sel]
        if len(sel) > 1:
            groups.append((agg, used,
                           np.concatenate([b['x_ppc'] for b in sel], 1),
                           np.concatenate([b['obs'] for b in sel])))
    return groups


def _hist_panel(ax, r, k, name):
    """LOO d^2 of the draws vs d^2_obs, with the Hotelling reference."""
    from scipy import stats
    j = min(k, len(r['ks'])) - 1
    k, N = r['ks'][j], r['N']
    d2l, d2o = r['d2_loo'][:, j], r['d2_obs'][j]
    hi = max(1.5 * np.percentile(d2l, 97.5), 1.1 * d2o)
    bins = np.linspace(0, hi, 30)
    nover = int((d2l > hi).sum())
    ax.hist(np.minimum(d2l, hi * 0.999), bins=bins, density=True,
            color=C_PPC, alpha=0.45,
            label=f'PPC draws, leave-one-out (N={N})'
                  + (f', {nover} piled in last bin' if nover else ''))
    c = (N + 1) / N * k * (N - 1) / (N - k)      # d^2 = c * F
    t = np.linspace(1e-3, hi, 400)
    ax.plot(t, stats.f.pdf(t / c, k, N - k) / c, color=C_PPC, lw=2,
            label=f'Hotelling $T^2$ reference, k={k}')
    ax.axvline(d2o, color=C_OBS, lw=2, label=r'$x_{\rm obs}$')
    ax.set_title(f'{name}: $d^2$ in top-{k} PCs  (D={r["D"]})\n'
                 f'$p_{{\\rm LOO}}$={r["p_emp"][j]:.3f},  '
                 f'$p_F$={r["p_F"][j]:.1e}', fontsize=10)
    ax.set_xlabel(r'Mahalanobis $d^2$', fontsize=9)
    ax.set_ylabel('density', fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.9)


def _dot_panel(ax, res, names, key_style, floor, title):
    y = np.arange(len(names))
    for key, lab, kw in key_style:
        ax.plot(-np.log10(res[key]), y, ls='none', ms=8, label=lab, **kw)
    ax.axvline(-np.log10(0.05), color='0.3', ls='--', lw=1,
               label='p = 0.05')
    ax.axvline(-np.log10(floor), color='0.3', ls=':', lw=1,
               label='LOO floor 1/(N+1)')
    ax.set_xlabel(r'$-\log_{10}\,p$   (right = more OOD)', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9, loc='upper center',
              bbox_to_anchor=(0.5, -0.1), ncol=2)


def plot_pvalue(blocks, k, title, out_path, tsv_path, kscan=30):
    """One figure: what is tested (a), the test on the two aggregate vectors
    (b, c), and every block's p at k (d), across k (e), and off-subspace (f).
    """
    from matplotlib.patches import Ellipse
    from scipy import stats
    groups = pvalue_groups(blocks)
    N = len(groups[0][2])
    kmax = max(k, min(kscan, N // 3))
    res = [ppc_pvalues(X, o, kmax) for _, _, X, o in groups]
    names = [g[0] for g in groups]
    used = [g[1] for g in groups]
    at_k = lambda key: np.array(  # noqa: E731
        [r[key][min(k, len(r['ks'])) - 1] for r in res])
    tab = {key: at_k(key) for key in ('d2_obs', 'p_emp', 'p_F', 'p_res')}
    kk = np.array([min(k, len(r['ks'])) for r in res])

    with open(tsv_path, 'w') as f:
        f.write('group\tused\tD\tk\td2_obs\tp_loo\tp_F\tp_resid\n')
        for i, n in enumerate(names):
            f.write(f'{n}\t{int(used[i])}\t{res[i]["D"]}\t{kk[i]}\t'
                    f'{tab["d2_obs"][i]:.4g}\t{tab["p_emp"][i]:.4g}\t'
                    f'{tab["p_F"][i]:.4g}\t{tab["p_res"][i]:.4g}\n')

    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.3,
                          wspace=0.32)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    axd = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[1, 1], sharey=axd)
    axf = fig.add_subplot(gs[1, 2], sharey=axd)

    # (a) the inference vector in its leading PC plane
    ia = names.index('inference (all)') if 'inference (all)' in names else 0
    Xa, oa = groups[ia][2], groups[ia][3]
    _, _, Pd, var = _pca_d2(Xa, Xa, 2)
    _, _, Po, _ = _pca_d2(Xa, oa[None], 2)
    ax = axa
    ax.scatter(Pd[:, 0], Pd[:, 1], s=18, color=C_PPC, alpha=0.6,
               label='PPC draws')
    for q, ls in ((0.68, '-'), (0.95, '--')):
        r2 = stats.chi2.ppf(q, 2)
        ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(r2 * var[0]),
                             2 * np.sqrt(r2 * var[1]), fill=False,
                             color=C_PPC, ls=ls, lw=1.2,
                             label=f'{int(q * 100)}% Gaussian contour'))
    ax.plot(Po[0, 0], Po[0, 1], '*', color=C_OBS, ms=16,
            label=r'$x_{\rm obs}$')
    ax.set_xlabel('PC1 (standardized)', fontsize=9)
    ax.set_ylabel('PC2 (standardized)', fontsize=9)
    ax.set_title(f'(a) {names[ia]}: leading 2 of k={kk[ia]} PCs of the '
                 'PPC draws\n$d^2$ sums (PC$_i$/$\\sigma_i$)$^2$ over all k',
                 fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9)

    # (b), (c) the test itself on the aggregate vectors
    for ax, agg, lab in ((axb, 'inference (all)', '(b)'),
                         (axc, 'held out (all)', '(c)')):
        if agg in names:
            _hist_panel(ax, res[names.index(agg)], k, f'{lab} {agg}')
        else:
            ax.axis('off')

    # (d) every group at k
    _dot_panel(axd, tab, names,
               [('p_emp', 'leave-one-out rank', dict(marker='o', color=C_PPC)),
                ('p_F', 'Hotelling F (Gaussian)',
                 dict(marker='s', mfc='none', color=C_OBS))],
               1 / (N + 1), f'(d) p-value per block, k={k} PCs')
    y = np.arange(len(names))
    axd.set_yticks(y)
    axd.set_yticklabels([f'{n}  [{"inf" if u else "held"}]'
                         for n, u in zip(names, used)], fontsize=8.5)
    axd.invert_yaxis()

    # (e) robustness to k
    M = np.full((len(res), kmax), np.nan)
    for i, r in enumerate(res):
        M[i, :len(r['ks'])] = -np.log10(r['p_emp'])
    im = axe.imshow(M, aspect='auto', cmap='Blues', vmin=0,
                    vmax=np.log10(N + 1), interpolation='nearest',
                    extent=[0.5, kmax + 0.5, len(res) - 0.5, -0.5])
    axe.axvline(k, color=C_OBS, lw=1.5, ls='--')
    axe.set_xlabel('k (PCs kept)', fontsize=9)
    axe.set_title(f'(e) leave-one-out p vs k  (dashed: k={k} used; '
                  'blank: k > D)',
                  fontsize=10)
    cb = fig.colorbar(im, ax=axe, pad=0.02)
    cb.set_label(r'$-\log_{10}\,p_{\rm LOO}$', fontsize=9)
    plt.setp(axe.get_yticklabels(), visible=False)

    # (f) deviation the top-k PCs cannot see
    _dot_panel(axf, tab, names,
               [('p_res', 'leave-one-out rank of |residual|$^2$',
                 dict(marker='D', color=C_PPC))],
               1 / (N + 1),
               f'(f) deviation orthogonal to the top-{k} PCs\n'
               '(blank: k = D, nothing left over)')
    plt.setp(axf.get_yticklabels(), visible=False)

    nsep = sum(used) - 0.5
    for ax in (axd, axe, axf):
        ax.axhline(nsep, color='0.3', lw=1)
        ax.tick_params(labelsize=8)
    for ax in (axa, axb, axc, axd, axf):
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    fig.suptitle(
        title + '\n'
        r'H$_0$: $x_{\rm obs}$ is a draw from the posterior predictive '
        r'$p(x\,|\,x_{\rm obs})$ sampled by the PPC draws.  Statistic: '
        r'Mahalanobis $d^2$ of standardized $x_{\rm obs}$ in the top-k PCs '
        'of the draws.  Small p = out of distribution.\n'
        'Inference blocks reuse $x_{\\rm obs}$ (fit and test), so their p is '
        'conservative; held-out blocks are a clean test.',
        fontsize=10.5)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def pvalue_title(exp, obs_exp, inf_labels, id_obs, n_ok):
    obs_from = ('' if obs_exp is exp else
                f' from {obs_exp.suite}/{obs_exp.sim} (out-of-distribution)')
    return (f'PPC OOD test  |  {exp.suite}/{exp.sim}, tracer={exp.tracer}, '
            f'conditioned on {"+".join(inf_labels)} at {exp.kmin} '
            f'$\\leq k \\leq$ {fmt_kmax(exp.kmax)}\n'
            f'$x_{{\\rm obs}}$ = lhid {id_obs}{obs_from}, '
            f'{n_ok} PPC draws')


def pvalue_from_saved(out, n_pca):
    """Rebuild the blocks from x_ppc_all.npz, which already holds every
    plotted block's draws and observed vector."""
    draws = np.load(join(out, 'posterior_draws.npz'), allow_pickle=True)
    inf_labels = [str(s) for s in draws['labels']]
    exp = ExpPath(str(draws['exp_path']))
    obs_exp = (ExpPath(str(draws['test_path']))
               if str(draws['test_path'] if 'test_path' in draws else '')
               else exp)
    z = np.load(join(out, 'x_ppc_all.npz'))
    names = [n for n in z.files if not n.endswith(('_obs', '_k'))]
    blocks = [dict(name=n, x_ppc=z[n], obs=z[n + '_obs'],
                   used=n in inf_labels) for n in names]
    n_ok = len(blocks[0]['x_ppc'])
    plotdir = join(out, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    plot_pvalue(blocks, n_pca,
                pvalue_title(exp, obs_exp, inf_labels, str(draws['id_obs']),
                             n_ok),
                join(plotdir, 'ppc_pvalue.png'), join(out, 'ppc_pvalues.tsv'))
    print(f'Wrote {join(plotdir, "ppc_pvalue.png")}, ppc_pvalues.tsv')


def main():
    args = build_argparser().parse_args()
    out = args.ppc_dir
    if args.pvalue_only:
        return pvalue_from_saved(out, args.n_pca)

    draws =np.load(join(out, 'posterior_draws.npz'), allow_pickle=True)
    theta_draws = draws['theta_draws']
    names = [str(s) for s in draws['param_names']]
    # sims and their recorded attrs use physical params; for a reparam model
    # theta_draws/names are (degen_r, degen_phi) and *_phys hold the inverse
    theta_phys = (draws['theta_phys'] if 'theta_phys' in draws
                  else theta_draws)
    names_phys = ([str(s) for s in draws['names_phys']]
                  if 'names_phys' in draws else names)
    inf_labels = [str(s) for s in draws['labels']]
    startidx_ref = list(draws['startidx'])
    exp = ExpPath(str(draws['exp_path']))
    obs_exp = (ExpPath(str(draws['test_path']))
               if str(draws['test_path'] if 'test_path' in draws else '')
               else exp)
    x_obs = draws['x_obs']
    theta_obs = np.asarray(draws['theta_obs'])
    theta_obs_phys = (np.asarray(draws['theta_obs_phys'])
                      if 'theta_obs_phys' in draws else theta_obs)
    id_obs = str(draws['id_obs'])
    nnets_req = int(draws['nnets']) if 'nnets' in draws else None

    cfg = OmegaConf.load(join(exp, 'config.yaml'))
    if cfg.infer.pca_features or exists(join(exp, 'pca.pkl')):
        raise SystemExit('Experiment uses PCA; must apply it, never refit.')
    kmin, kmax = exp.kmin, exp.kmax
    sim_sub = args.sim_sub or sim_subdir(cfg)
    print(f'exp_path   = {exp}')
    print(f'tracer     = {exp.tracer}'
          + ('  (lightcone)' if exp.is_lightcone else ''))
    print(f'k-cut      = {kmin} <= k <= {fmt_kmax(kmax)}')
    print(f'inference  = {inf_labels}, startidx {startidx_ref}')
    print(f'correct_shot={cfg.infer.correct_shot}, '
          f'loglinear_start_idx={cfg.infer.loglinear_start_idx}')

    # --- the observed sim's own diagnostics (for held-out summaries) --------
    obs_dir = args.obs_dir
    if obs_dir is None:
        # An OOD observation lives in the testing suite, whose box need not
        # match the training suite's, so use that experiment's own config.
        obs_cfg = (cfg if obs_exp is exp else
                   OmegaConf.load(join(obs_exp, 'config.yaml')))
        obs_dir = join(obs_exp.suite_root,
                       f'L{obs_cfg.nbody.L}-N{obs_cfg.nbody.N}', id_obs)
    if obs_exp is not exp:
        print(f'x_obs is   = out-of-distribution, from {obs_exp}')
    obs_diag = find_obs_diag(join(obs_dir, obs_exp.diag_dir),
                             theta_obs_phys, names_phys, args.atol)
    print(f'x_obs from = {obs_diag}')
    obs_data = load_summ(obs_diag, lightcone=exp.is_lightcone)

    # --- which blocks to plot ----------------------------------------------
    if args.summaries:
        plot_labels = [s.strip() for s in args.summaries.split(',')]
    else:
        # every redshift-space summary present, plus the equilateral/squeezed
        # slices of the bispectrum monopole. A lightcone is already in redshift
        # space, so its keys carry no 'z' and every key qualifies.
        pre = '' if exp.is_lightcone else 'z'
        avail = sorted(k for k in obs_data if k.startswith(pre))
        plot_labels = (
            inf_labels
            + [s for s in avail if s not in inf_labels]
            + [f'{pre}{t}Bk0' for t in ('Eq', 'Sq')
               if f'{pre}Bk0' in avail])
    print(f'plotting   = {plot_labels}')

    # --- gather per-draw summaries -----------------------------------------
    draw_summs = []
    needed_bases = {split_tag(lab)[0] for lab in plot_labels}
    needed_bases |= {split_tag(lab)[0][:-1] + '0' for lab in plot_labels}
    kept, status = [], {}
    for i, theta in enumerate(theta_phys):
        simdir = join(out, sim_sub, str(i))
        diagfile = join(simdir,
                        exp.diag_file(args.hod_seed, args.aug_seed))
        if not exists(diagfile):
            status[i] = 'missing_diag'
            continue
        s = load_summ(diagfile, lightcone=exp.is_lightcone)
        if any(b not in s for b in needed_bases):
            status[i] = 'incomplete_summ'
            continue
        problems = check_draw(diagfile, theta, names_phys, args.atol)
        if problems:
            status[i] = 'param_mismatch'
            print(f'  draw {i}: PARAM MISMATCH -- ' + '; '.join(problems))
            continue
        draw_summs.append(s)
        kept.append(i)
        status[i] = 'ok'

    n_ok = len(kept)
    print(f'\n{n_ok}/{len(theta_draws)} draws usable')
    for st in sorted(set(status.values()) - {'ok'}):
        bad = [i for i, v in status.items() if v == st]
        print(f'  {st}: {len(bad)} -> {bad}')
    if n_ok == 0:
        raise SystemExit('No usable draws.')

    # per-draw dicts, reindexed by base summary for preprocess_*
    draw_data = {b: [s[b] for s in draw_summs] for b in needed_bases}

    # the training pool, for the grey context band on inference blocks
    POOL, _, _, _ = load_pool(exp, ('train', 'val', 'test'))

    # --- preprocess every plotted block ------------------------------------
    blocks, xs_inf = [], []
    for lab in plot_labels:
        base, _ = split_tag(lab)
        X = preprocess_block(lab, draw_data, cfg, kmin, kmax, args.bk_kmax)
        o = preprocess_block(lab, {b: [obs_data[b]] for b in needed_bases},
                             cfg, kmin, kmax, args.bk_kmax)[0]
        xax, xlabel, skmax = block_axis(
            lab, obs_data[base]['k'], kmin, kmax, args.bk_kmax)
        if len(xax) != X.shape[1]:
            raise SystemExit(
                f'{lab}: axis has {len(xax)} points but block has '
                f'{X.shape[1]} features.')
        used = lab in inf_labels
        pool = None
        if used:                      # pool band only where a trained x exists
            j = inf_labels.index(lab)
            pool = POOL[:, startidx_ref[j]:startidx_ref[j + 1]]
            o = x_obs[startidx_ref[j]:startidx_ref[j + 1]]
            xs_inf.append(X)
        blocks.append(dict(name=lab, x=xax, xlabel=xlabel,
                           ylabel=ylabel_for(lab), obs=o, x_ppc=X,
                           pool=pool, used=used, n=X.shape[1], kmax=skmax))
        print(f'  {lab:9s} {X.shape[1]:3d} bins  '
              f'{"inference" if used else "held out"}  kmax={skmax}')

    # --- deliverables: inference blocks only, unchanged ---------------------
    x_ppc = np.concatenate(xs_inf, axis=-1)
    startidx = list(np.cumsum([0] + [b.shape[1] for b in xs_inf]))
    if startidx != list(startidx_ref):
        raise SystemExit(
            f'Block layout {startidx} != training {list(startidx_ref)}.')
    theta_ppc = theta_draws[kept]
    np.save(join(out, 'x_ppc.npy'), x_ppc)
    np.save(join(out, 'theta_ppc.npy'), theta_ppc)
    np.savez(join(out, 'x_ppc_all.npz'),
             **{b['name']: b['x_ppc'] for b in blocks},
             **{b['name'] + '_obs': b['obs'] for b in blocks},
             **{b['name'] + '_k': b['x'] for b in blocks})
    print(f'\nWrote x_ppc.npy {x_ppc.shape}, theta_ppc.npy {theta_ppc.shape}, '
          f'x_ppc_all.npz ({len(blocks)} blocks)')

    # --- manifest -----------------------------------------------------------
    mpath = join(out, 'manifest.tsv')
    rows = open(mpath).read().splitlines()
    header, body = rows[0], rows[1:]
    row_of = {int(r.split('\t')[0]): r.split('\t') for r in body}
    with open(mpath, 'w') as f:
        f.write(header + '\n')
        for i in range(len(theta_draws)):
            r = row_of[i]
            r[1] = status.get(i, 'pending')
            f.write('\t'.join(r) + '\n')

    if args.no_plot:
        return

    # --- bands --------------------------------------------------------------
    n_nets = actual_nnets(exp, nnets_req or cfg.infer.Nnets)
    nets = f'{n_nets}-net ' if n_nets else ''
    if n_nets and nnets_req and n_nets != nnets_req:
        print(f'NOTE: ensemble is {n_nets} nets, not the {nnets_req} '
              f'requested (missing posterior.pkl for some top trials)')
    obs_from = ('' if obs_exp is exp else
                f' from {obs_exp.suite}/{obs_exp.sim} (out-of-distribution)')
    title = (
        f'Posterior predictive check  |  {exp.suite}/{exp.sim}, '
        f'tracer={exp.tracer}\n'
        f'conditioned on {"+".join(inf_labels)} at {kmin} $\\leq k \\leq$ '
        f'{fmt_kmax(kmax)}  |  {nets}{cfg.infer.backend}/{cfg.infer.engine} '
        f'ensemble, '
        f'correct_shot={cfg.infer.correct_shot}\n'
        f'$x_{{\\rm obs}}$ = lhid {id_obs}{obs_from} '
        f'({os.path.basename(obs_diag)}), '
        f'{n_ok} joint draws from $q(\\theta|x_{{\\rm obs}})$')
    plotdir = join(out, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    plot_bands(blocks, title, join(plotdir, 'ppc_bands.png'))
    print(f'Wrote {join(plotdir, "ppc_bands.png")}')

    if not args.no_pvalue:
        plot_pvalue(blocks, args.n_pca,
                    pvalue_title(exp, obs_exp, inf_labels, id_obs, n_ok),
                    join(plotdir, 'ppc_pvalue.png'),
                    join(out, 'ppc_pvalues.tsv'))
        print(f'Wrote {join(plotdir, "ppc_pvalue.png")}, ppc_pvalues.tsv')

    if args.no_theta_plots:
        return

    # --- theta-space plots --------------------------------------------------
    # resim.py's importance-sampling diagnostics, reused. For a PPC these are
    # bookkeeping checks, not tests of the model: theta_ppc are direct
    # posterior draws, so both SHOULD agree with the direct sample. What they
    # catch is drift between what we drew and what got simulated. The ensemble
    # is unweighted, so ESS = N and max weight = 1/N trivially.
    import torch
    from cmass.infer.validate import load_ensemble
    w = np.full(n_ok, 1. / n_ok)
    ensemble = load_ensemble(str(exp), cfg.infer.Nnets, plot=False,
                             clean=False).to(args.device)
    with torch.no_grad():
        theta_post = ensemble.sample(
            (args.n_post,), torch.Tensor(x_obs).to(args.device),
            show_progress_bars=False).cpu().numpy()
    logq_post = batched_log_prob(ensemble, theta_post, x_obs,
                                 args.batch_size, args.device)
    logq_ppc = batched_log_prob(ensemble, theta_ppc, x_obs,
                                args.batch_size, args.device)
    print(f'log q(theta|x_obs): direct median {np.median(logq_post):.2f}, '
          f'simulated median {np.median(logq_ppc):.2f}')

    plot_logprob(logq_post, logq_ppc, float(n_ok), 1.0, 1. / n_ok, plotdir)
    shutil.move(join(plotdir, 'plot_resim_logprob.png'),
                join(plotdir, 'ppc_logprob.png'))
    plot_corner(theta_post, theta_ppc, w, theta_obs, names, plotdir)
    shutil.move(join(plotdir, 'plot_resim_corner.png'),
                join(plotdir, 'ppc_corner.png'))
    np.save(join(out, 'logq_ppc.npy'), logq_ppc)
    print('Wrote ppc_logprob.png, ppc_corner.png, logq_ppc.npy')


if __name__ == '__main__':
    main()
