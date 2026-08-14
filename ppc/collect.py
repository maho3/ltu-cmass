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
theta_obs, not by hardcoding a filename.

k-grid caveat: the training summaries predate ltu-cmass ba2334f (2026-07-14,
pylians -> pypower for periodic-box P(k)), so training P(k) sits on a uniform
0.01 grid and these PPC runs sit on pypower's effective-k grid. Both give 39
bins in k <= 0.4. Bk is unaffected. Per instruction this is left as-is and
everything is plotted against the PPC (pypower) k, with residuals taken
element-wise -- matching how x_ppc and x_obs are used downstream.

No Mahalanobis distances, no p-values -- bands only.
"""

import argparse
import os
import re
import shutil
from os.path import join, exists
import numpy as np
import h5py
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt   # noqa: E402

from cmass.infer.loaders import (      # noqa: E402
    preprocess_Pk, preprocess_Bk, load_Pk, load_Bk,
    _is_in_kminmax, _get_Bk_mask)
from cmass.infer.resim import (        # noqa: E402
    load_pool, plot_logprob, plot_corner, batched_log_prob)
from cmass.infer.tools import resolve_kmax   # noqa: E402

WDIR = '/work/hdd/bdne/maho3/cmass-ili'
PPC = join(WDIR, 'ppc/abacuslike_fastpm_charm6_comphod',
           'zPk0+zPk2+zPk4_kmin-0.0_kmax-0.4/obs01880')
AF = 0.666660000066666           # analysis snapshot (a), key '0.666660'
N_COSMO, N_NOISE = 5, 2
TAGS = ('Eq', 'Sq', 'Ss', 'Is', '')

# Entity -> colour, fixed. Observed is ink, the PPC ensemble is one hue, the
# training pool is recessive fill. Separated by lightness as well as hue, so it
# survives CVD and greyscale print.
C_OBS, C_PPC, C_POOL = 'k', 'C0', '0.85'


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ppc_dir', default=PPC)
    p.add_argument('--sim_sub', default='fastpm/L2000-N256')
    p.add_argument('--hod_seed', type=int, default=1,
                   help='which diag/galaxies/hodNNNNN.h5 to read per draw')
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


def kcut_from_path(exp_path):
    m = re.match(r'kmin-([\d.]+)_kmax-([\d.]+)', os.path.basename(exp_path))
    if not m:
        raise ValueError(f'Cannot parse k-cut from {exp_path}')
    return float(m.group(1)), float(m.group(2))


def split_tag(summ):
    """'zEqBk0' -> ('zBk0', 'Eq'), replicating run_preprocessing's loop."""
    for tag in TAGS:
        if tag in summ:
            return summ.replace(tag, ''), tag
    return summ, ''


def load_summ(diagfile):
    s = {}
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


def find_obs_diag(obs_dir, theta_obs, names, atol):
    """The observed lhid has several HOD realizations; pick the one whose
    recorded HOD parameters are theta_obs. Identifying it by content rather
    than filename keeps this correct if the pool ordering ever changes."""
    diagdir = join(obs_dir, 'diag', 'galaxies')
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


def main():
    args = build_argparser().parse_args()
    out = args.ppc_dir

    draws = np.load(join(out, 'posterior_draws.npz'), allow_pickle=True)
    theta_draws = draws['theta_draws']
    names = [str(s) for s in draws['param_names']]
    inf_labels = [str(s) for s in draws['labels']]
    startidx_ref = list(draws['startidx'])
    exp_path = str(draws['exp_path'])
    x_obs = draws['x_obs']
    theta_obs = np.asarray(draws['theta_obs'])
    id_obs = str(draws['id_obs'])
    nnets_req = int(draws['nnets']) if 'nnets' in draws else None

    cfg = OmegaConf.load(join(exp_path, 'config.yaml'))
    if cfg.infer.pca_features or exists(join(exp_path, 'pca.pkl')):
        raise SystemExit('Experiment uses PCA; must apply it, never refit.')
    kmin, kmax = kcut_from_path(exp_path)
    print(f'exp_path   = {exp_path}')
    print(f'k-cut      = {kmin} <= k <= {kmax}')
    print(f'inference  = {inf_labels}, startidx {startidx_ref}')
    print(f'correct_shot={cfg.infer.correct_shot}, '
          f'loglinear_start_idx={cfg.infer.loglinear_start_idx}')

    # --- the observed sim's own diagnostics (for held-out summaries) --------
    obs_dir = args.obs_dir
    if obs_dir is None:
        suite_root = exp_path.split(os.sep + 'models' + os.sep)[0]
        obs_dir = join(suite_root, f'L{cfg.nbody.L}-N{cfg.nbody.N}', id_obs)
    obs_diag = find_obs_diag(obs_dir, theta_obs, names, args.atol)
    print(f'x_obs from = {obs_diag}')
    obs_data = load_summ(obs_diag)

    # --- which blocks to plot ----------------------------------------------
    if args.summaries:
        plot_labels = [s.strip() for s in args.summaries.split(',')]
    else:
        # every z-space summary present, plus the equilateral/squeezed slices
        # of the bispectrum monopole
        avail = sorted(k for k in obs_data if k.startswith('z'))
        plot_labels = (
            inf_labels
            + [s for s in avail if s not in inf_labels]
            + [f'z{t}Bk0' for t in ('Eq', 'Sq') if 'zBk0' in avail])
    print(f'plotting   = {plot_labels}')

    # --- gather per-draw summaries -----------------------------------------
    draw_summs = []
    needed_bases = {split_tag(lab)[0] for lab in plot_labels}
    needed_bases |= {split_tag(lab)[0][:-1] + '0' for lab in plot_labels}
    kept, status = [], {}
    for i, theta in enumerate(theta_draws):
        simdir = join(out, args.sim_sub, str(i))
        diagfile = join(simdir, 'diag', 'galaxies',
                        f'hod{args.hod_seed:05d}.h5')
        if not exists(diagfile):
            status[i] = 'missing_diag'
            continue
        s = load_summ(diagfile)
        if any(b not in s for b in needed_bases):
            status[i] = 'incomplete_summ'
            continue
        problems = check_draw(diagfile, theta, names, args.atol)
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
    POOL, _, _, _ = load_pool(exp_path, ('train', 'val', 'test'))

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
    parts = exp_path.rstrip('/').split(os.sep)
    suite, sim, tracer = parts[-6], parts[-5], parts[-3]
    n_nets = actual_nnets(exp_path, nnets_req or cfg.infer.Nnets)
    nets = f'{n_nets}-net ' if n_nets else ''
    if n_nets and nnets_req and n_nets != nnets_req:
        print(f'NOTE: ensemble is {n_nets} nets, not the {nnets_req} '
              f'requested (missing posterior.pkl for some top trials)')
    title = (
        f'Posterior predictive check  |  {suite}/{sim}, tracer={tracer}\n'
        f'conditioned on {"+".join(inf_labels)} at {kmin} $\\leq k \\leq$ '
        f'{kmax}  |  {nets}{cfg.infer.backend}/{cfg.infer.engine} ensemble, '
        f'correct_shot={cfg.infer.correct_shot}\n'
        f'$x_{{\\rm obs}}$ = lhid {id_obs} ({os.path.basename(obs_diag)}), '
        f'{n_ok} joint draws from $q(\\theta|x_{{\\rm obs}})$')
    plotdir = join(out, 'plots')
    os.makedirs(plotdir, exist_ok=True)
    plot_bands(blocks, title, join(plotdir, 'ppc_bands.png'))
    print(f'Wrote {join(plotdir, "ppc_bands.png")}')

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
    ensemble = load_ensemble(exp_path, cfg.infer.Nnets, plot=False,
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
