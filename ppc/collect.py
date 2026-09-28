"""
Step 4 of the posterior predictive check (PPC) campaign.

Collects the per-draw diagnostics into the deliverable arrays and plots the
posterior predictive bands against the observed data vector.

    <out>/theta_ppc.npy   (Ndraw, 17)   theta actually simulated
    <out>/x_ppc.npy       (Ndraw, 117)  the INFERENCE blocks only, in training
                                        x ordering -- the deliverable
    <out>/x_ppc_all.npz   every plotted summary block, inference or not;
                          <name>_k123 holds each bispectrum triangle's sides
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

OOD p-value (plots/ppc_pcapvalue.png, ppc_pcapvalues.tsv): tests H0 "x_obs
is a draw from the posterior predictive the PPC ensemble samples", per block
and for the concatenated inference / held-out vectors. The statistic is the
Mahalanobis distance in the top-k PCs of the draws (N ~ 100 draws cannot
support a full covariance), calibrated by the matched leave-one-out rank of
predictive_checks.ppc_mahalanobis(method='pca'), floored at 1/(N+1). A
Hotelling-F p extrapolates past that floor under a Gaussian assumption, a
separate LOO p scores the deviation orthogonal to the top-k PCs, and a full-D
Ledoit-Wolf covariance keeps every direction.

k-bin p-value (plots/ppc_kbinpvalue.png, ppc_kbinpvalues.tsv): the same
matched test with no PCA, on the raw features inside a k-range -- sliding
windows across the BAO scales, and cumulative k <= kmax cuts with P(k)
averaged into coarser bins. Subsets with more than N/2 features are not
tested, so each fold's covariance stays well estimated. Bispectrum blocks need
<name>_k123, stored from ltu-cmass 2026-09-24 on.

--pvalue_only rebuilds both figures from an existing x_ppc_all.npz without
touching the per-draw sims.
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
    p.add_argument('--win_width', type=float, default=0.06,
                   help='k-window width for the data-space p-value')
    p.add_argument('--win_step', type=float, default=0.02,
                   help='k-window step for the data-space p-value')
    p.add_argument('--kcoarse', type=float, default=0.04,
                   help='P(k) averaging width for the cumulative-kmax p-value')
    p.add_argument('--no_pvalue', action='store_true')
    p.add_argument('--pvalue_only', action='store_true',
                   help='only remake the p-value figures from x_ppc_all.npz')
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
    """x values, axis label, k-cut and triangle sides (3, n) for a block.

    P(k) blocks and equilateral bispectra have a single k per feature, so they
    get a real k axis. General triangle configurations do not -- three k's per
    point -- so those fall back to a triangle index, and the sides are
    returned separately (None for P(k)).
    """
    base, tag = split_tag(summ)
    is_bk = ('Bk' in base) or ('Qk' in base)
    skmax = (bk_kmax if (is_bk and bk_kmax is not None)
             else resolve_kmax(kmax, summ))
    if not is_bk:
        k = np.asarray(kdata)
        return (k[_is_in_kminmax(k, kmin, skmax)], r'$k$ [$h$/Mpc]', skmax,
                None)
    k123 = np.asarray(kdata)
    mask = _get_Bk_mask(k123, kmin, skmax, equilateral=(tag == 'Eq'),
                        squeezed=(tag == 'Sq'), subsampled=(tag == 'Ss'),
                        isoceles=(tag == 'Is'))
    if tag == 'Eq':      # k1 == k2 == k3, so a k axis is meaningful
        return k123[0][mask], r'$k$ [$h$/Mpc]', skmax, k123[:, mask]
    return (np.arange(int(mask.sum())), 'triangle index', skmax,
            k123[:, mask])


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


# --- OOD p-value: matched leave-one-out Mahalanobis in PCA space ------------
# Same construction as predictive_checks.ppc_mahalanobis(method='pca'): in fold
# j the mean and covariance come from the N-1 draws other than j, and both the
# held-out draw x_j and x_obs are scored against that SAME fit. Under H0 the two
# are exchangeable within every fold, so
#     p = (1 + #{j : T_j >= T_obs,j}) / (N + 1)
# is calibrated without any distributional assumption.


def pca_basis(X, kmax):
    """Mean, top-kmax eigenvectors (D, k) and eigenvalues of cov(X) on the raw
    features. Modes below eps * max(D, N-1) * lambda_max are dropped, as in
    predictive_checks."""
    N, D = X.shape
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    lam = S**2 / (N - 1)
    n_ok = int((lam > np.finfo(float).eps * max(D, N - 1) * lam[0]).sum())
    k = min(kmax, n_ok)
    return mu, Vt[:k].T, lam[:k]


def pca_scores(x, mu, V, lam):
    """For every k = 1..len(lam): Mahalanobis T in the top-k PCs, and the
    squared norm of the part of x - mu those k PCs leave unexplained."""
    dx = x - mu
    proj2 = (dx @ V)**2
    T = np.cumsum(proj2 / lam, axis=-1)
    resid = (dx**2).sum(-1)[..., None] - np.cumsum(proj2, axis=-1)
    return T, resid


def loo_rank_p(t_rep, t_obs):
    """Matched leave-one-out p along axis 0, and its Monte Carlo noise
    std(b) / sqrt(N), as predictive_checks reports it."""
    b = t_rep >= t_obs
    N = len(b)
    return (1 + b.sum(0)) / (N + 1), b.std(0) / np.sqrt(N)


def pca_pvalues(X, obs, kmax):
    """Matched leave-one-out p-values of obs against the PPC draws X (N, D),
    for k = 1..kmax PCs."""
    from scipy import stats
    N, D = X.shape
    kmax = min(kmax, D, N - 2)
    folds = []
    for j in range(N):
        basis = pca_basis(np.delete(X, j, 0), kmax)
        folds.append(pca_scores(np.stack([X[j], obs]), *basis))
    kmax = min(T.shape[-1] for T, _ in folds)
    T = np.array([t[:, :kmax] for t, _ in folds])     # (N, [draw, obs], k)
    R = np.array([r[:, :kmax] for _, r in folds])
    ks = np.arange(1, kmax + 1)
    p, p_std = loo_rank_p(T[:, 0], T[:, 1])
    p_res, _ = loo_rank_p(R[:, 0], R[:, 1])
    p_res[ks >= D] = np.nan                   # k = D leaves no residual

    # Gaussian extrapolation below the 1/(N+1) floor, from obs against all N
    # draws: Hotelling T^2 = d^2 N/(N+1) ~ k(N-1)/(N-k) F(k, N-k)
    d2_obs = pca_scores(obs, *pca_basis(X, kmax))[0][:kmax]
    fstat = d2_obs * N / (N + 1) * (N - ks) / (ks * (N - 1))
    p_F = stats.f.sf(fstat, ks, N - ks)
    return dict(ks=ks, N=N, D=D, t_rep=T[:, 0], t_obs=T[:, 1], p=p,
                p_std=p_std, p_res=p_res, p_F=p_F)


def lw_pvalue(X, obs):
    """The same matched leave-one-out test over all D features, with a
    Ledoit-Wolf shrinkage covariance instead of a PC truncation. Each fold's
    draws standardize the features first: the shrinkage target is a multiple
    of the identity, which only makes sense on a common scale."""
    from sklearn.covariance import LedoitWolf
    N = len(X)
    T = np.empty((N, 2))
    for j in range(N):
        rest = np.delete(X, j, 0)
        mu, sd = rest.mean(0), rest.std(0, ddof=1)
        sd = np.where(sd > 0, sd, 1.)
        lw = LedoitWolf().fit((rest - mu) / sd)
        T[j] = lw.mahalanobis((np.stack([X[j], obs]) - mu) / sd)
    return loo_rank_p(T[:, 0], T[:, 1])[0]


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


def _matched_panel(ax, r, k, name, what=None):
    """One point per fold: x_obs and the held-out draw, scored against the
    same N-1 draws. p is the fraction of points on or above the diagonal."""
    j = min(k, len(r['ks'])) - 1
    k, N = r['ks'][j], r['N']
    t_obs, t_rep = r['t_obs'][:, j], r['t_rep'][:, j]
    above = t_rep >= t_obs
    ax.scatter(t_obs[above], t_rep[above], s=18, color=C_PPC,
               label=f'draw at least as extreme ({above.sum()}/{N})')
    ax.scatter(t_obs[~above], t_rep[~above], s=18, facecolor='none',
               edgecolor=C_PPC, label=f'$x_{{\\rm obs}}$ more extreme '
                                      f'({(~above).sum()}/{N})')
    lo = min(t_obs.min(), t_rep.min()) * 0.8
    hi = max(t_obs.max(), t_rep.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], color=C_OBS, lw=1.5,
            label='$T_j = T_{\\rm obs,j}$')
    ax.set(xscale='log', yscale='log', xlim=(lo, hi), ylim=(lo, hi))
    what = what or f'top-{k} PCs'
    ax.set_title(f'{name}: matched folds, {what}  (d={r["D"]})\n'
                 f'$p_{{\\rm LOO}}$={r["p"][j]:.3f} $\\pm$ '
                 f'{r["p_std"][j]:.3f},  $p_F$={r["p_F"][j]:.1e}',
                 fontsize=10)
    ax.set_xlabel(r'$T_{\rm obs,j}$: $x_{\rm obs}$ vs fold-$j$ fit',
                  fontsize=9)
    ax.set_ylabel(r'$T_j$: held-out draw $j$ vs the same fit', fontsize=9)
    ax.legend(fontsize=7.5, framealpha=0.9, loc='upper left')


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


def plot_pcapvalue(blocks, k, title, out_path, tsv_path, kscan=30):
    """One figure: what is tested (a), the matched test on the two aggregate
    vectors (b, c), and every block's p at k (d), across k (e), and
    off-subspace (f)."""
    from matplotlib.patches import Ellipse
    from scipy import stats
    groups = pvalue_groups(blocks)
    N = len(groups[0][2])
    kmax = max(k, min(kscan, N // 3))
    res = [pca_pvalues(X, o, kmax) for _, _, X, o in groups]
    names = [g[0] for g in groups]
    used = [g[1] for g in groups]
    kk = np.array([min(k, len(r['ks'])) for r in res])
    tab = {key: np.array([r[key][ki - 1] for r, ki in zip(res, kk)])
           for key in ('p', 'p_std', 'p_F', 'p_res')}
    tab['p_lw'] = np.array([lw_pvalue(X, o) for _, _, X, o in groups])

    with open(tsv_path, 'w') as f:
        f.write('group\tused\tD\tk\tp_loo\tp_loo_std\tp_F\tp_resid\tp_lw\n')
        for i, n in enumerate(names):
            f.write(f'{n}\t{int(used[i])}\t{res[i]["D"]}\t{kk[i]}\t'
                    + '\t'.join(f'{tab[c][i]:.4g}' for c in
                                ('p', 'p_std', 'p_F', 'p_res', 'p_lw'))
                    + '\n')

    fig = plt.figure(figsize=(19, 13.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.3,
                          wspace=0.32, top=0.88)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    axd = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[1, 1], sharey=axd)
    axf = fig.add_subplot(gs[1, 2], sharey=axd)

    # (a) the inference vector in its leading PC plane
    ia = names.index('inference (all)') if 'inference (all)' in names else 0
    Xa, oa = groups[ia][2], groups[ia][3]
    mu, V, lam = pca_basis(Xa, 2)
    Pd, Po = (Xa - mu) @ V, (oa - mu) @ V
    ax = axa
    ax.scatter(Pd[:, 0], Pd[:, 1], s=18, color=C_PPC, alpha=0.6,
               label='PPC draws')
    for q, ls in ((0.68, '-'), (0.95, '--')):
        r2 = stats.chi2.ppf(q, 2)
        ax.add_patch(Ellipse((0, 0), 2 * np.sqrt(r2 * lam[0]),
                             2 * np.sqrt(r2 * lam[1]), fill=False,
                             color=C_PPC, ls=ls, lw=1.2,
                             label=f'{int(q * 100)}% Gaussian contour'))
    ax.plot(Po[0], Po[1], '*', color=C_OBS, ms=16, label=r'$x_{\rm obs}$')
    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.set_title(f'(a) {names[ia]}: leading 2 of k={kk[ia]} PCs of the '
                 'PPC draws\n$T$ sums (PC$_i$/$\\sigma_i$)$^2$ over all k',
                 fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9)

    # (b), (c) the test itself on the aggregate vectors
    for ax, agg, lab in ((axb, 'inference (all)', '(b)'),
                         (axc, 'held out (all)', '(c)')):
        if agg in names:
            _matched_panel(ax, res[names.index(agg)], k, f'{lab} {agg}')
        else:
            ax.axis('off')

    # (d) every group at k
    _dot_panel(axd, tab, names,
               [('p', f'top-{k} PCs, leave-one-out',
                 dict(marker='o', color=C_PPC)),
                ('p_F', f'top-{k} PCs, Hotelling F (Gaussian)',
                 dict(marker='s', mfc='none', color=C_OBS)),
                ('p_lw', 'all D features, Ledoit-Wolf, leave-one-out',
                 dict(marker='^', color=C_OBS))],
               1 / (N + 1), f'(d) p-value per block: top-{k} PCs vs all '
               'D features (Ledoit-Wolf)')
    y = np.arange(len(names))
    axd.set_yticks(y)
    axd.set_yticklabels([f'{n}  [{"inf" if u else "held"}]'
                         for n, u in zip(names, used)], fontsize=8.5)
    axd.invert_yaxis()

    # (e) robustness to k
    M = np.full((len(res), kmax), np.nan)
    for i, r in enumerate(res):
        M[i, :len(r['ks'])] = -np.log10(r['p'])
    im = axe.imshow(M, aspect='auto', cmap='Blues', vmin=0,
                    vmax=np.log10(N + 1), interpolation='nearest',
                    extent=[0.5, kmax + 0.5, len(res) - 0.5, -0.5])
    axe.axvline(k, color=C_OBS, lw=1.5, ls='--')
    axe.set_xlabel('k (PCs kept)', fontsize=9)
    axe.set_title(f'(e) leave-one-out p vs k  (dashed: k={k} used; '
                  'blank: k > D)', fontsize=10)
    cb = fig.colorbar(im, ax=axe, pad=0.02)
    cb.set_label(r'$-\log_{10}\,p_{\rm LOO}$', fontsize=9)
    plt.setp(axe.get_yticklabels(), visible=False)

    # (f) deviation the top-k PCs cannot see
    _dot_panel(axf, tab, names,
               [('p_res', 'leave-one-out, |residual|$^2$',
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
        r'Mahalanobis $T$ in the top-k PCs of the draws (and, in (d), in all '
        'D features under a Ledoit-Wolf covariance).\n'
        'Matched leave-one-out: in fold $j$, draw $j$ and $x_{\\rm obs}$ are '
        'scored against the same fit to the other $N-1$ draws.  '
        'Small p = out of distribution.\n'
        'Inference blocks reuse $x_{\\rm obs}$ (fit and test), so their p is '
        'conservative; held-out blocks are a clean test.',
        fontsize=10.5)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --- OOD p-value in data space: k-bin subsets --------------------------------
# The same matched leave-one-out test on the raw features inside a k-range,
# instead of the top PCs. With every mode kept, T is the full Mahalanobis
# distance. A subset is tested only if it has d <= N/2 features, so the
# covariance of the N-1 draws in each fold stays well estimated.

BAO_K = (0.05, 0.3)      # h/Mpc, where the BAO wiggles sit in P(k)


def feature_k(block):
    """One k per feature: the bin k for P(k), the largest side for a
    bispectrum triangle. None when a bispectrum block has no saved sides
    (x_ppc_all.npz collected before k123 was stored)."""
    if block.get('k123') is not None:
        return np.asarray(block['k123']).max(0)
    base, tag = split_tag(block['name'])
    if ('Bk' in base or 'Qk' in base) and tag != 'Eq':
        return None
    return np.asarray(block['x'])


def coarsen(X, k, dk):
    """Average adjacent features into k-bins of width dk. Returns the
    averaged features and each coarse bin's upper edge."""
    idx = np.floor(k / dk + 1e-9).astype(int)
    bins = np.unique(idx)
    Xc = np.stack([X[..., idx == i].mean(-1) for i in bins], -1)
    return Xc, (bins + 1) * dk


def kbin_groups(blocks):
    """Blocks with a k per feature, plus all P(k) multipoles together."""
    groups = []
    for b in blocks:
        k = feature_k(b)
        if k is None:
            print(f'  {b["name"]}: no triangle sides saved, skipped in the '
                  'k-bin test (re-collect to store k123)')
            continue
        is_pk = 'Pk' in split_tag(b['name'])[0]
        groups.append(dict(name=b['name'], X=b['x_ppc'], obs=b['obs'], k=k,
                           is_pk=is_pk))
    pk = [g for g in groups if g['is_pk']]
    if len(pk) > 1:
        groups.append(dict(
            name='+'.join(g['name'] for g in pk),
            X=np.concatenate([g['X'] for g in pk], 1),
            obs=np.concatenate([g['obs'] for g in pk]),
            k=np.concatenate([g['k'] for g in pk]), is_pk=True, combo=pk))
    return groups


def subset_pvalue(X, obs, sel, cap):
    """Matched leave-one-out p on the features in sel, or NaN when there are
    none or more than cap of them."""
    d = int(sel.sum())
    out = dict(d=d, p=np.nan, p_std=np.nan, p_F=np.nan, r=None)
    if 0 < d <= cap:
        r = pca_pvalues(X[:, sel], obs[sel], d)
        out.update(p=r['p'][-1], p_std=r['p_std'][-1], p_F=r['p_F'][-1], r=r)
    return out


def kbin_scans(groups, N, width, step, dk):
    """p in sliding k-windows and below cumulative kmax cuts, per group.

    Windows use the raw bins. The kmax scan averages P(k) into dk-wide bins
    so that all multipoles fit under the d <= N/2 cap out to the largest k;
    bispectrum triangles stay raw and drop out once they exceed it.
    """
    cap = N // 2
    kp = np.unique(np.concatenate([g['k'] for g in groups if g['is_pk']]
                                  or [g['k'] for g in groups]))
    half = 0.5 * (np.diff(kp).min() if len(kp) > 1 else 0.)
    grid = step / 2                 # round edges, whatever the k-binning
    k_lo = np.floor((kp.min() - half) / grid + 1e-6) * grid
    k_hi = kp.max() + half
    starts = np.arange(k_lo, k_hi - width + 1e-9, step)
    windows = np.stack([starts, starts + width], 1)
    cuts = dk * np.arange(1, int(np.ceil(k_hi / dk - 1e-9)) + 1)

    win, cum = [], []
    for g in groups:
        win.append([subset_pvalue(g['X'], g['obs'],
                                  (g['k'] >= lo) & (g['k'] < hi), cap)
                    for lo, hi in windows])
        if g['is_pk']:
            # obs rides along as the last row, so it is averaged identically
            parts = [coarsen(np.vstack([h['X'], h['obs']]), h['k'], dk)
                     for h in g.get('combo', [g])]
            XO = np.concatenate([x for x, _ in parts], 1)
            kc = np.concatenate([edge for _, edge in parts])
            X, obs = XO[:-1], XO[-1]
        else:
            X, obs, kc = g['X'], g['obs'], g['k']
        cum.append([subset_pvalue(X, obs, kc <= c + 1e-9, cap)
                    for c in cuts])
    return windows, cuts, win, cum, cap


def _kbin_heatmap(ax, fig, x, res, names, N, cap, xlabel, title, widths):
    """-log10 p per group (rows) and k-range (columns). Each cell is labelled
    with its d. Hatched cells were not tested: no bins, or d > N/2."""
    from matplotlib.patches import Rectangle
    M = np.array([[c['p'] for c in row] for row in res])
    D = np.array([[c['d'] for c in row] for row in res])
    x0, x1 = x[0] - widths / 2, x[-1] + widths / 2
    ax.add_patch(Rectangle((x0, -0.5), x1 - x0, len(res), fill=False,
                           hatch='///', color='0.8', lw=0, zorder=0))
    im = ax.imshow(-np.log10(M), aspect='auto', cmap='Blues', vmin=0,
                   vmax=np.log10(N + 1), interpolation='nearest',
                   extent=[x0, x1, len(res) - 0.5, -0.5], zorder=1)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if D[i, j]:
                dark = np.isfinite(M[i, j]) and M[i, j] < 10**-1.2
                ax.text(x[j], i, D[i, j], ha='center', va='center',
                        fontsize=6, color='w' if dark else '0.3', zorder=2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'$-\log_{10}\,p_{\rm LOO}$', fontsize=9)


def _kbin_lines(ax, x, res, groups, N, xlabel, title):
    """P(k) groups only: the multipoles and their combination."""
    for i, g in enumerate(groups):
        if not g['is_pk']:
            continue
        p = np.array([c['p'] for c in res[i]])
        combo = 'combo' in g
        ax.plot(x, -np.log10(p), marker='o', ms=4,
                lw=2.5 if combo else 1.5,
                color=C_OBS if combo else None, label=g['name'])
    ax.axvspan(*BAO_K, color=C_POOL, zorder=0, label='BAO range')
    ax.axhline(-np.log10(0.05), color='0.3', ls='--', lw=1, label='p = 0.05')
    ax.axhline(-np.log10(1 / (N + 1)), color='0.3', ls=':', lw=1,
               label='LOO floor 1/(N+1)')
    ax.set_ylim(-0.05, np.log10(N + 1) + 0.15)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(r'$-\log_{10}\,p_{\rm LOO}$   (up = more OOD)', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.9)


def plot_kbinpvalue(blocks, title, out_path, tsv_path, width=0.06,
                    step=0.02, dk=0.04, kmax_train=None):
    """One figure: the per-bin deviation being tested (c), p in sliding
    k-windows (a, d) and below cumulative kmax cuts (b, e), and the matched
    folds at the most OOD P(k) window (f)."""
    groups = kbin_groups(blocks)
    if not any(g['is_pk'] for g in groups):
        print('No P(k) blocks with k; skipping the k-bin p-value figure.')
        return False
    N = len(groups[0]['X'])
    windows, cuts, win, cum, cap = kbin_scans(groups, N, width, step, dk)
    centres = windows.mean(1)
    names = [g['name'] for g in groups]

    with open(tsv_path, 'w') as f:
        f.write('scan\tgroup\tk_lo\tk_hi\td\tp_loo\tp_loo_std\tp_F\n')
        for scan, res, ranges in (
                ('window', win, windows),
                ('kmax', cum, [(0., c) for c in cuts])):
            for g, row in zip(names, res):
                for (lo, hi), c in zip(ranges, row):
                    f.write(f'{scan}\t{g}\t{lo:.4g}\t{hi:.4g}\t{c["d"]}\t'
                            f'{c["p"]:.4g}\t{c["p_std"]:.4g}\t'
                            f'{c["p_F"]:.4g}\n')

    fig = plt.figure(figsize=(19, 13.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.32,
                          wspace=0.3, top=0.89)
    axa, axb, axc, axd, axe, axf = (fig.add_subplot(gs[i, j])
                                    for i in range(2) for j in range(3))
    single = [i for i, g in enumerate(groups)
              if g['is_pk'] and 'combo' not in g]
    dwin = max(c['d'] for i in single for c in win[i])

    # (a), (b) the P(k) curves
    _kbin_lines(axa, centres, win, groups, N, r'window centre $k$ [$h$/Mpc]',
                f'(a) sliding k-window, width {width} (d $\\leq$ {dwin} per '
                f'multipole)')
    _kbin_lines(axb, cuts, cum, groups, N, r'$k_{\rm max}$ [$h$/Mpc]',
                f'(b) cumulative $k \\leq k_{{\\rm max}}$, P(k) averaged into '
                f'$\\Delta k$ = {dk} bins')
    if kmax_train is not None:
        axb.axvline(kmax_train, color=C_OBS, lw=1.5, ls='-.',
                    label='training $k_{\\rm max}$')
        axb.legend(fontsize=7.5, framealpha=0.9)

    # (c) what is being tested, bin by bin
    for g in (groups[i] for i in single):
        mu, sd = g['X'].mean(0), g['X'].std(0, ddof=1)
        axc.plot(g['k'], (g['obs'] - mu) / sd, marker='o', ms=3, lw=1.2,
                 label=g['name'])
    axc.axvspan(*BAO_K, color=C_POOL, zorder=0, label='BAO range')
    for s in (-2, 2):
        axc.axhline(s, color='0.3', ls='--', lw=1)
    axc.axhline(0, color=C_OBS, lw=1)
    axc.set_xlabel(r'$k$ [$h$/Mpc]', fontsize=9)
    axc.set_ylabel(r'$(x_{\rm obs} - \bar x_{\rm PPC}) / \sigma_{\rm PPC}$',
                   fontsize=9)
    axc.set_title('(c) per-bin deviation of $x_{\\rm obs}$ from the PPC '
                  'draws\n(marginal only; the tests use the full covariance)',
                  fontsize=10)
    axc.legend(fontsize=7.5, framealpha=0.9)

    # (d), (e) every group, cells labelled with d
    _kbin_heatmap(axd, fig, centres, win, names, N, cap,
                  r'window centre $k$ [$h$/Mpc]',
                  '(d) sliding window, all blocks\n(cell text: d; hatched: '
                  f'untested, d = 0 or d > N/2 = {cap})', step)
    _kbin_heatmap(axe, fig, cuts, cum, names, N, cap,
                  r'$k_{\rm max}$ [$h$/Mpc]',
                  '(e) cumulative $k_{\\rm max}$, all blocks\n(bispectrum '
                  'triangles by largest side, not averaged)', dk)

    # (f) the matched folds behind the most OOD P(k) window
    ic = next((i for i, g in enumerate(groups) if 'combo' in g), single[0])
    pw = np.array([c['p'] for c in win[ic]])
    if np.isfinite(pw).any():
        jw = int(np.nanargmin(pw))
        lo, hi = windows[jw]
        c = win[ic][jw]
        _matched_panel(axf, c['r'], c['d'], f'(f) {names[ic]}',
                       what=f'{lo:.3f} $\\leq k <$ {hi:.3f}, most OOD window')
    else:
        axf.axis('off')

    for ax in (axa, axb, axc, axf):
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in (axa, axb, axc, axd, axe, axf):
        ax.tick_params(labelsize=8)

    fig.suptitle(
        title + '\n'
        r'H$_0$: $x_{\rm obs}$ is a draw from the posterior predictive '
        r'sampled by the PPC draws.  Statistic: full Mahalanobis $T$ on the '
        r'raw features inside a k-range (no PCA), tested only when '
        f'd $\\leq$ N/2 = {cap}.\n'
        'Matched leave-one-out: in fold $j$, draw $j$ and $x_{\\rm obs}$ are '
        'scored against the same fit to the other $N-1$ draws.  '
        'Small p = out of distribution.\n'
        'Inference blocks reuse $x_{\\rm obs}$ (fit and test), so their p is '
        'conservative; held-out blocks are a clean test.',
        fontsize=10.5)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


def write_kbinpvalue(blocks, out, title, kmax_train, width, step, dk):
    plotdir = join(out, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    if plot_kbinpvalue(blocks, title, join(plotdir, 'ppc_kbinpvalue.png'),
                       join(out, 'ppc_kbinpvalues.tsv'), width, step, dk,
                       kmax_train):
        print(f'Wrote {join(plotdir, "ppc_kbinpvalue.png")}, '
              'ppc_kbinpvalues.tsv')


def pvalue_title(exp, obs_exp, inf_labels, id_obs, n_ok):
    obs_from = ('' if obs_exp is exp else
                f' from {obs_exp.suite}/{obs_exp.sim} (out-of-distribution)')
    return (f'PPC OOD test  |  {exp.suite}/{exp.sim}, tracer={exp.tracer}, '
            f'conditioned on {"+".join(inf_labels)} at {exp.kmin} '
            f'$\\leq k \\leq$ {fmt_kmax(exp.kmax)}\n'
            f'$x_{{\\rm obs}}$ = lhid {id_obs}{obs_from}, '
            f'{n_ok} PPC draws')


def scalar_kmax(exp):
    """The training k-cut, if it is one number (not a per-summary mix)."""
    return None if isinstance(exp.kmax, dict) else exp.kmax


def write_pcapvalue(blocks, out, n_pca, title):
    plotdir = join(out, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    plot_pcapvalue(blocks, n_pca, title, join(plotdir, 'ppc_pcapvalue.png'),
                   join(out, 'ppc_pcapvalues.tsv'))
    print(f'Wrote {join(plotdir, "ppc_pcapvalue.png")}, ppc_pcapvalues.tsv')


def pvalue_from_saved(out, args):
    """Rebuild the blocks from x_ppc_all.npz, which already holds every
    plotted block's draws and observed vector."""
    draws = np.load(join(out, 'posterior_draws.npz'), allow_pickle=True)
    inf_labels = [str(s) for s in draws['labels']]
    exp = ExpPath(str(draws['exp_path']))
    obs_exp = (ExpPath(str(draws['test_path']))
               if str(draws['test_path'] if 'test_path' in draws else '')
               else exp)
    z = np.load(join(out, 'x_ppc_all.npz'))
    names = [n for n in z.files if not n.endswith(('_obs', '_k', '_k123'))]
    blocks = [dict(name=n, x_ppc=z[n], obs=z[n + '_obs'], x=z[n + '_k'],
                   k123=z[n + '_k123'] if n + '_k123' in z.files else None,
                   used=n in inf_labels) for n in names]
    title = pvalue_title(exp, obs_exp, inf_labels, str(draws['id_obs']),
                         len(blocks[0]['x_ppc']))
    write_pcapvalue(blocks, out, args.n_pca, title)
    write_kbinpvalue(blocks, out, title, scalar_kmax(exp), args.win_width,
                     args.win_step, args.kcoarse)


def main():
    args = build_argparser().parse_args()
    out = args.ppc_dir
    if args.pvalue_only:
        return pvalue_from_saved(out, args)

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
        xax, xlabel, skmax, k123 = block_axis(
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
                           pool=pool, used=used, n=X.shape[1], kmax=skmax,
                           k123=k123))
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
             **{b['name'] + '_k': b['x'] for b in blocks},
             **{b['name'] + '_k123': b['k123'] for b in blocks
                if b['k123'] is not None})
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
        ptitle = pvalue_title(exp, obs_exp, inf_labels, id_obs, n_ok)
        write_pcapvalue(blocks, out, args.n_pca, ptitle)
        write_kbinpvalue(blocks, out, ptitle, scalar_kmax(exp),
                         args.win_width, args.win_step, args.kcoarse)

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
