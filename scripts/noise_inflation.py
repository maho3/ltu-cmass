"""
Posterior stdev on Omega_m, sigma8 at fiducial cosmology vs kmax, for models
trained with fixed vs marginalized observational noise.

(A) each model on its own test set (std averaged over the test noise draws)
(B) every model on the reference (fixed-noise) suite's test x, same rows
"""
import argparse
import os
import pickle
from os.path import join, exists

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from cmass.infer.tools import iter_kcuts, kcut_dirname, resolve_kmax

matplotlib.use('Agg')

CASES = [
    ('fixed $\\sigma_r,\\sigma_t$', 'fastpm_charm7_noisefix'),
    ('fixed $\\sigma_r$, varied $\\sigma_t$', 'fastpm_charm7_noisetrans'),
    ('varied $\\sigma_r,\\sigma_t$ (reparam)', 'fastpm_charm7_cosmoHOD_reparam'),
    ('varied $\\sigma_r,\\sigma_t$', 'fastpm_charm7_cosmoHOD'),
]
COLORS = ['#2a78d6', '#eb6834', '#eda100', '#1baf7a']
MARKERS = ['o', 's', 'D', '^']
PLOT_CASES = [0, 1, 3]  # reparam (2) kept in the npz, not plotted
THETAFID = np.array([0.3, 0.8])  # Omega_m, sigma8
PIDX = [0, 4]
PNAMES = ['\\Omega_m', '\\sigma_8']


def fidmask(theta, nbar, rtol, nbar_lo, nbar_hi):
    nbar = nbar.mean(axis=1) if nbar.ndim > 1 else nbar
    m = np.all(np.isclose(theta[:, PIDX], THETAFID, rtol=rtol), axis=1)
    return m & (nbar > nbar_lo) & (nbar < nbar_hi)


def load_test(d):
    return {k: np.load(join(d, f'{k}_test.npy'))
            for k in ['x', 'theta', 'nbar', 'ids']}


def std_own(d, args):
    """(A): std over saved posterior samples at fiducial test rows."""
    if not exists(join(d, 'posterior_samples.npy')):
        return None
    t = load_test(d)
    m = fidmask(t['theta'], t['nbar'], args.rtol, args.nbar_lo, args.nbar_hi)
    s = np.load(join(d, 'posterior_samples.npy'), mmap_mode='r')
    return np.asarray(s[:, m][..., PIDX]).std(axis=0)


def std_common(d, ref, m, args):
    """(B): std of fresh samples conditioned on the reference rows."""
    if not exists(join(d, 'posterior.pkl')):
        return None
    assert (open(join(d, 'x_startidx.txt')).read() ==
            open(join(ref, 'x_startidx.txt')).read()), f'feature mismatch {d}'
    ids_ref = np.load(join(ref, 'ids_test.npy'))[m]
    train_ids = np.concatenate([np.load(join(d, f'ids_{k}.npy'))
                                for k in ['train', 'val']])
    assert not np.isin(ids_ref, train_ids).any(), f'train leak {d}'
    with open(join(d, 'posterior.pkl'), 'rb') as f:
        post = pickle.load(f)
    x = torch.Tensor(np.load(join(ref, 'x_test.npy'))[m])
    torch.manual_seed(args.seed)
    s = post.sample_batched((args.nsamples,), x,
                            show_progress_bars=False).cpu().numpy()
    return s[..., PIDX].std(axis=0)


def summarize(std):
    return np.percentile(std, [16, 50, 84], axis=0)  # (3, 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wdir', default='/work/hdd/bdne/maho3/cmass-ili')
    p.add_argument('--nbody', default='abacuslike')
    p.add_argument('--tracer', default='galaxy')
    p.add_argument('--infer', default='cmass/conf/infer/noisetest.yaml')
    p.add_argument('--nsamples', type=int, default=2000)
    p.add_argument('--rtol', type=float, default=0.1)
    p.add_argument('--nbar_lo', type=float, default=np.log10(1e-4))
    p.add_argument('--nbar_hi', type=float, default=np.log10(5e-4))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='figures/noise_inflation_abacuslike_charm7')
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_ON_NODE', 4)))

    exps = OmegaConf.load(args.infer).experiments
    rows = []  # (summary, kcut_dir, pk_kmax, bk_kmax, case_idx, method, std)
    for exp in exps:
        name = '+'.join(exp.summary)
        for kmin, kmax in iter_kcuts(exp):
            kd = kcut_dirname(kmin, kmax)
            pk = resolve_kmax(kmax, 'zPk0')
            bk = resolve_kmax(kmax, 'zBk0') if 'zBk0' in exp.summary else np.nan
            dirs = [join(args.wdir, args.nbody, sim, 'models', args.tracer,
                         name, kd) for _, sim in CASES]
            ref = dirs[0]
            if not exists(join(ref, 'x_test.npy')):
                print(f'skip {name} {kd}: no reference test set')
                continue
            t = load_test(ref)
            m = fidmask(t['theta'], t['nbar'], args.rtol,
                        args.nbar_lo, args.nbar_hi)
            print(f'{name} {kd}: {m.sum()} fiducial rows, '
                  f'{len(np.unique(t["ids"][m]))} lhids')
            for i, d in enumerate(dirs):
                for method, fn in [('A', lambda: std_own(d, args)),
                                   ('B', lambda: std_common(d, ref, m, args))]:
                    s = fn()
                    if s is not None:
                        rows.append((name, kd, pk, bk, i, method, s))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out + '.npz',
             summary=[r[0] for r in rows], kcut=[r[1] for r in rows],
             pk_kmax=[r[2] for r in rows], bk_kmax=[r[3] for r in rows],
             case=[r[4] for r in rows], method=[r[5] for r in rows],
             std=np.array([r[6] for r in rows], dtype=object),
             case_labels=[c[0] for c in CASES],
             case_sims=[c[1] for c in CASES])

    for method in ['A', 'B']:
        plot(rows, exps, method, f'{args.out}_{method}.png')


def plot(rows, exps, method, fname):
    names = ['+'.join(e.summary) for e in exps]
    fig, axs = plt.subplots(len(names), 4, figsize=(17, 3.6*len(names)),
                            squeeze=False)
    for r, name in enumerate(names):
        sel = [x for x in rows if x[0] == name and x[5] == method]
        ref = {(x[1]): x[6] for x in sel if x[4] == 0}
        # Bk rows: zBk=0.2 series solid, zBk=zPk series dotted
        groups = [('', lambda x: True)] if 'zBk0' not in name else [
            ('$k_B$=0.2', lambda x: x[3] == 0.2),
            ('$k_B$=$k_P$', lambda x: x[3] == x[2])]
        for g, (glab, gsel) in enumerate(groups):
            for n, i in enumerate(PLOT_CASES):
                clab = CASES[i][0]
                pts = sorted([x for x in sel if x[4] == i and gsel(x)],
                             key=lambda x: x[2])
                if not pts:
                    continue
                kx = np.array([x[2] for x in pts]) * (1 + 0.025*(n-1))
                q = np.array([summarize(x[6]) for x in pts])  # (n, 3, 2)
                kw = dict(color=COLORS[i], marker=MARKERS[i], ms=6, lw=1.5,
                          ls='-' if g == 0 else ':',
                          label=clab if g == 0 else None)
                for j in range(2):
                    axs[r, j].errorbar(kx, q[:, 1, j],
                                       yerr=[q[:, 1, j]-q[:, 0, j],
                                             q[:, 2, j]-q[:, 1, j]],
                                       capsize=2, **kw)
                    if method == 'B':  # paired rows: median per-row ratio
                        rat = [np.median(x[6][:, j] / ref[x[1]][:, j])
                               if x[1] in ref else np.nan for x in pts]
                    else:
                        rat = [summarize(x[6])[1, j] / summarize(ref[x[1]])[1, j]
                               if x[1] in ref else np.nan for x in pts]
                    axs[r, 2+j].plot(kx, rat, **kw)
        for j in range(2):
            axs[r, j].set_ylabel(f'std$({PNAMES[j]})$')
            axs[r, j].set_ylim(bottom=0)
            axs[r, 2+j].set_ylabel(f'std$({PNAMES[j]})$ / fixed-noise')
            axs[r, 2+j].axhline(1, color='#52514e', lw=0.8)
        axs[r, 0].set_title(name, loc='left', fontsize=11)
        for ax in axs[r]:
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_xlabel('$k_{\\max}^{P}$ [h/Mpc]')
    axs[0, 3].legend(fontsize=9, loc='upper left')
    if 'zBk0' in names[-1]:
        axs[-1, 3].legend(
            [Line2D([], [], color='k', ls='-'), Line2D([], [], color='k', ls=':')],
            ['$k_B$=0.2', '$k_B$=$k_P$'], fontsize=9, loc='upper left')
    desc = ('(B) all models on fixed-noise test x ($\\sigma_r=\\sigma_t=1.5$)'
            if method == 'B' else '(A) each model on its own test set')
    fig.suptitle('abacuslike/fastpm_charm7 galaxies, z-space, fiducial '
                 f'$\\Omega_m$=0.3, $\\sigma_8$=0.8 (±10%): {desc}', y=1.0)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'saved {fname}')


if __name__ == '__main__':
    main()
