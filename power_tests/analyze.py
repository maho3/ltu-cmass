"""Analyze pylians vs pypower comparison outputs and make report figures.

Run from repo root:
    PYTHONPATH=. python power_tests/analyze.py
"""
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

K_EDGES = np.arange(0.01, 0.601, 0.01)  # fixed production-style grid


def rebin_fixed(k, Pk, Nmodes):
    """Mode-weighted rebin onto K_EDGES (same logic as cmass rebin_pk, but
    with edges independent of each measurement's k.max())."""
    kc = 0.5 * (K_EDGES[:-1] + K_EDGES[1:])
    out = np.full((len(kc), Pk.shape[1]), np.nan)
    for i, (lo, hi) in enumerate(zip(K_EDGES[:-1], K_EDGES[1:])):
        m = (k >= lo) & (k < hi)
        if m.any():
            out[i] = np.average(Pk[m], weights=Nmodes[m], axis=0)
    return kc, out


TRUTH = 'pypower_n512_i2'
TRUTH_CHECK = 'pylians_n512'
CONFIGS = ['pylians_n128', 'pypower_n128_i0', 'pypower_n128_i2',
           'pylians_n256', 'pypower_n256_i2']
LABELS = {
    'pylians_n128': 'pylians N=128',
    'pylians_n256': 'pylians N=256 (high_res)',
    'pypower_n128_i0': 'pypower N=128, no interlacing',
    'pypower_n128_i2': 'pypower N=128, interlacing=2',
    'pypower_n256_i2': 'pypower N=256, interlacing=2',
    'pypower_n512_i2': 'pypower N=512, interlacing=2 (truth)',
    'pylians_n512': 'pylians N=512',
}
COLORS = {
    'pylians_n128': 'tab:blue', 'pylians_n256': 'tab:cyan',
    'pypower_n128_i0': 'tab:red', 'pypower_n128_i2': 'tab:orange',
    'pypower_n256_i2': 'tab:green',
}
ALL = CONFIGS + [TRUTH, TRUTH_CHECK]
ELLS = [0, 2, 4]


def load_all(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return [np.load(f) for f in files], files


def rebinned(d, space, cfg):
    """Rebin one measurement onto the fixed production grid."""
    key = f'{space}_{cfg}'
    return rebin_fixed(d[f'{key}_k'], d[f'{key}_Pk'], d[f'{key}_Nmodes'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', default='./data/scratch/power_tests')
    ap.add_argument('--figdir', default='./figures')
    args = ap.parse_args()

    data, files = load_all(f'{args.indir}/compare_lhid*.npz')
    lhids = [int(d['lhid']) for d in data]
    print(f'{len(data)} lhids: {lhids}')

    # --- Ratio to truth, per config -------------------------------------
    for space in ['real', 'zspace']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
        for iell, ell in enumerate(ELLS):
            ax = axes[iell]
            kt = None
            for cfg in CONFIGS:
                ratios = []
                for d in data:
                    kt, Pt = rebinned(d, space, TRUTH)
                    kc, Pc = rebinned(d, space, cfg)
                    ratios.append(Pc[:, iell] / Pt[:, iell])
                ratios = np.array(ratios)
                mean, std = np.nanmean(ratios, 0), np.nanstd(ratios, 0)
                ax.plot(kt, mean, color=COLORS[cfg], label=LABELS[cfg])
                ax.fill_between(kt, mean - std, mean + std,
                                color=COLORS[cfg], alpha=0.2, lw=0)
            ax.axhline(1, color='k', lw=0.5)
            ax.axhspan(0.99, 1.01, color='gray', alpha=0.15)
            ax.set_xlabel(r'$k$ [$h$/Mpc]')
            ax.set_title(rf'$P_{ell}$')
            ax.set_ylim(0.9, 1.1) if ell == 0 else ax.set_ylim(0.7, 1.3)
        axes[0].set_ylabel(r'$P_\ell / P_\ell^{\rm truth}$')
        axes[0].legend(fontsize=8)
        fig.suptitle(
            f'Aliasing test vs N=512 interlaced truth — {space}, '
            f'{len(data)} Quijote halo boxes (lhids {min(lhids)}–{max(lhids)})')
        fig.tight_layout()
        fig.savefig(f'{args.figdir}/pk_backend_ratio_{space}.png', dpi=150)
        print(f'saved {args.figdir}/pk_backend_ratio_{space}.png')

    # --- Truth self-consistency -----------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for space, c in [('real', 'tab:blue'), ('zspace', 'tab:red')]:
        ratios = []
        for d in data:
            kt, Pt = rebinned(d, space, TRUTH)
            kc, Pc = rebinned(d, space, TRUTH_CHECK)
            ratios.append(Pc[:, 0] / Pt[:, 0])
        mean = np.nanmean(ratios, 0)
        ax.plot(kt, mean, color=c, label=f'{space} P0')
    ax.axhline(1, color='k', lw=0.5)
    ax.set_xlabel(r'$k$ [$h$/Mpc]')
    ax.set_ylabel('pylians N=512 / pypower N=512 i2')
    ax.set_title('Truth cross-check (P0, mean over lhids)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{args.figdir}/pk_backend_truthcheck.png', dpi=150)
    print(f'saved {args.figdir}/pk_backend_truthcheck.png')

    # --- Production-settings difference: pylians n128 vs pypower n128 i2 --
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, space in zip(axes, ['real', 'zspace']):
        for iell, ell in enumerate(ELLS[:2]):
            diffs = []
            for d in data:
                k1, P1 = rebinned(d, space, 'pylians_n128')
                k2, P2 = rebinned(d, space, 'pypower_n128_i2')
                # normalize by P0 so near-zero multipoles don't blow up
                diffs.append((P2[:, iell] - P1[:, iell]) / P1[:, 0])
            diffs = np.array(diffs)
            ax.plot(k1, 100 * np.nanmean(diffs, 0), label=rf'$P_{ell}$')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_title(space)
    axes[0].set_ylabel(r'(pypower i2 − pylians) / $P_0^{\rm pylians}$ [%]')
    axes[0].legend()
    fig.suptitle('Change in production N=128 data vector if backend swapped')
    fig.tight_layout()
    fig.savefig(f'{args.figdir}/pk_backend_prod_diff.png', dpi=150)
    print(f'saved {args.figdir}/pk_backend_prod_diff.png')

    # --- Raw spectra (not ratios), single sim ------------------------------
    d0 = data[0]
    lhid0 = int(d0['lhid'])
    for space in ['real', 'zspace']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for iell, ell in enumerate(ELLS):
            ax = axes[iell]
            for cfg in CONFIGS + [TRUTH]:
                color = COLORS.get(cfg, 'k')
                kc, Pc = rebinned(d0, space, cfg)
                if ell == 0:
                    ax.loglog(kc, Pc[:, iell], color=color, label=LABELS[cfg])
                else:
                    ax.semilogx(kc, Pc[:, iell], color=color, label=LABELS[cfg])
                    ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel(r'$k$ [$h$/Mpc]')
            ax.set_title(rf'$P_{ell}$')
        axes[0].set_ylabel(r'$P_\ell(k)$ [$(\mathrm{Mpc}/h)^3$]')
        axes[0].legend(fontsize=8)
        fig.suptitle(f'Raw power spectra — {space}, lhid {lhid0}')
        fig.tight_layout()
        fig.savefig(f'{args.figdir}/pk_backend_raw_{space}.png', dpi=150)
        print(f'saved {args.figdir}/pk_backend_raw_{space}.png')

    # --- z-space monopole: raw P(k) on top, ratio to truth on bottom ------
    space, iell = 'zspace', 0
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 6.5), sharex=True,
        gridspec_kw={'height_ratios': [2, 1]})
    d0 = data[0]
    lhid0 = int(d0['lhid'])
    kt = None
    for cfg in CONFIGS + [TRUTH]:
        color = COLORS.get(cfg, 'k')
        _, P0 = rebinned(d0, space, cfg)
        ratios = []
        for d in data:
            kt, Pt = rebinned(d, space, TRUTH)
            kc, Pc = rebinned(d, space, cfg)
            ratios.append(Pc[:, iell] / Pt[:, iell])
        ratios = np.array(ratios)
        ax_top.loglog(kc, P0[:, iell], color=color, label=LABELS[cfg])
        mean, std = np.nanmean(ratios, 0), np.nanstd(ratios, 0)
        ax_bot.plot(kt, mean, color=color)
        ax_bot.fill_between(kt, mean - std, mean + std,
                            color=color, alpha=0.2, lw=0)
    ax_top.set_ylabel(r'$P_0(k)$ [$(\mathrm{Mpc}/h)^3$]')
    ax_top.set_title(f'z-space monopole — lhid {lhid0}')
    ax_top.legend(fontsize=8)
    ax_bot.axhline(1, color='k', lw=0.5)
    ax_bot.axhspan(0.999, 1.001, color='gray', alpha=0.15)
    ax_bot.set_ylim(0.995, 1.005)
    ax_bot.set_xlabel(r'$k$ [$h$/Mpc]')
    ax_bot.set_ylabel(r'$P_0 / P_0^{\rm truth}$')
    fig.tight_layout()
    fig.savefig(f'{args.figdir}/pk_backend_zspace_p0.png', dpi=150)
    print(f'saved {args.figdir}/pk_backend_zspace_p0.png')

    # --- Timings ----------------------------------------------------------
    print('\nTimings (best-of-3, mean ± std over lhids, seconds):')
    rows = []
    for cfg in ALL:
        t = np.array([min(d[f'real_{cfg}_times']) for d in data])
        rows.append((cfg, t.mean(), t.std()))
        print(f'  {cfg:22s} {t.mean():7.2f} ± {t.std():.2f}')
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [LABELS[c] for c, _, _ in rows]
    ax.barh(names[::-1], [m for _, m, _ in rows][::-1],
            xerr=[s for _, _, s in rows][::-1], color='tab:blue')
    ax.set_xlabel('wall time per P(k) measurement [s]')
    ax.set_title('Timing (16 threads, best of 3, mean over lhids)')
    fig.tight_layout()
    fig.savefig(f'{args.figdir}/pk_backend_timing.png', dpi=150)
    print(f'saved {args.figdir}/pk_backend_timing.png')

    np.savez(f'{args.indir}/timing_summary.npz',
             configs=[r[0] for r in rows],
             mean=[r[1] for r in rows], std=[r[2] for r in rows])


if __name__ == '__main__':
    main()
