"""
Voxelization diagnostic for Quijote halos.

Pipeline
--------
Branch A (G_truth):  halos -> HOD -> RSD(z) -> Gaussian noise(sigma_r, sigma_t) -> P_l(k)
Branch B (G_voxel):  halos -> snap-to-voxel-center -> uniform redistribution in voxel
                     -> HOD -> RSD(z) -> Gaussian noise(sigma_r, sigma_t) -> P_l(k)

For each branch, sweep (sigma_r, sigma_t) in {0, 0.5, 1, 2, 3, 4}^2 Mpc/h (36 configs each).
Compute P_0 and P_2 up to k <= 0.4 h/Mpc for all 72 measurements.

Seed: 42 (HOD populate_mock, voxel redistribution, Gaussian noise RNG).

Outputs
-------
- figures/voxel_diagnostic_results.npz  : P(k) measurements (with timestamp suffix if exists)
- figures/voxel_diagnostic_Pk_families.png
- figures/voxel_diagnostic_bestfit_residual.png

Usage
-----
    python scripts/voxel_devoxelization_diagnostic.py [--halo-path PATH] [--seed 42] ...
"""

from cmass.diagnostics.calculations import MA, MAz, calcPk
from cmass.diagnostics.tools import noise_positions
from cmass.bias.tools.hod import parse_hod
from cmass.bias.apply_hod import populate_hod
from astropy.cosmology import FlatLambdaCDM
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import argparse
import logging
import os
from os.path import join

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')


SIGMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
DEFAULT_HALO_PATH = '/work/hdd/bdne/maho3/cmass-ili/quijote/nbody/L1000-N128/48/halos.h5'
DEFAULT_CONFIG_PATH = '/work/hdd/bdne/maho3/cmass-ili/quijote/nbody/L1000-N128/48/config.yaml'
DEFAULT_BIAS_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cmass', 'conf', 'bias', 'zheng_biased.yaml'
)


def load_halos(halo_path):
    """Load Quijote halos from the L1000-N128/48 snapshot."""
    with h5py.File(halo_path, 'r') as f:
        a_keys = list(f.keys())
        a_str = a_keys[0]
        g = f[a_str]
        hpos = g['pos'][...].astype(np.float64)
        hvel = g['vel'][...].astype(np.float64)
        hmass = g['mass'][...].astype(np.float64)
    a = float(a_str)
    z = 1.0 / a - 1.0
    logging.info(f'Loaded halos: a={a:.6f} (z={z:.4f}), N_h={len(hpos)}')
    return hpos, hvel, hmass, a, z


def voxelize_uniform(hpos, L, N, rng):
    """Snap positions to nearest voxel center then draw a uniform offset within the voxel."""
    delta = L / N
    # fractional index, clipped to [0, N)
    idx = np.floor(hpos / delta).astype(np.int64)
    idx = np.clip(idx, 0, N - 1)
    centers = (idx + 0.5) * delta
    # independent uniform offsets on each axis
    offsets = rng.uniform(-delta / 2.0, delta / 2.0, size=hpos.shape)
    pos_new = centers + offsets
    pos_new %= L
    return pos_new


def build_hod_cfg(bias_yaml_path, cosmo_list, L, N, lhid, redshift):
    """Build a minimal OmegaConf with bias and nbody sections, then run parse_hod."""
    bias_cfg = OmegaConf.load(bias_yaml_path)
    # ensure noise_uniform is False — we handle voxelization manually
    bias_cfg.hod.noise_uniform = False
    # seed=0 means "use defaults from reid2014_cmass" for HOD params
    bias_cfg.hod.seed = 0

    cfg = OmegaConf.create({
        'meta': {'wdir': '.'},
        'sim': 'quijote',
        'nbody': {
            'suite': 'quijote',
            'L': L,
            'N': N,
            'lhid': lhid,
            'zf': redshift,
            'cosmo': list(cosmo_list),
        },
        'bias': OmegaConf.to_container(bias_cfg, resolve=True),
    })
    cfg = parse_hod(cfg)
    return cfg


def run_hod(hpos, hvel, hmass, cosmo, L, redshift, cfg, populate_seed):
    """Run HOD populate using cmass.bias.apply_hod.populate_hod with the given seed."""
    galcat = populate_hod(
        hpos, hvel, hmass,
        cosmo, L, redshift,
        cfg.bias.hod.model, cfg.bias.hod.theta,
        hmeta=None,
        seed=populate_seed,
        mdef=cfg.bias.hod.mdef,
        zpivot=getattr(cfg.bias.hod, 'zpivot', None),
        assem_bias=getattr(cfg.bias.hod, 'assem_bias', False),
        vel_assem_bias=getattr(cfg.bias.hod, 'vel_assem_bias', False),
        custom_prior=getattr(cfg.bias.hod, 'custom_prior', None),
    )
    gpos = np.array([galcat['x'], galcat['y'], galcat['z']]).T
    gvel = np.array([galcat['vx'], galcat['vy'], galcat['vz']]).T
    return gpos.astype(np.float64), gvel.astype(np.float64)


def apply_rsd_z(pos, vel, L, cosmo, z):
    """Apply RSD along z by calling MAz-style redshift_space shift; return shifted positions only."""
    import copy
    import redshift_space_library as RSL
    pos_out = copy.deepcopy(np.ascontiguousarray(pos.astype(np.float32)))
    vel_f = np.ascontiguousarray(vel.astype(np.float32))
    RSL.pos_redshift_space(pos_out, vel_f, L, cosmo.H(z).value / cosmo.h, z, 2)
    pos_out = pos_out.astype(np.float64) % L
    return pos_out


def measure_pk(pos, L, N_mesh, MAS='TSC', threads=16):
    """Compute P(k) multipoles (P_0, P_2) for positions. Returns (k, P0, P2)."""
    field = MA(pos.astype(np.float32), L, N_mesh, MAS=MAS).astype(np.float32)
    k, Pk = calcPk(field, L, axis=2, MAS=MAS, threads=threads)
    # PKL.Pk .Pk has columns [monopole, quadrupole, hexadecapole]
    P0 = Pk[:, 0]
    P2 = Pk[:, 1]
    return k, P0, P2


def sweep_noise(gpos_rsd, L, N_mesh, sigmas, rng_seed, threads=16):
    """Sweep (sigma_r, sigma_t) grid; return dict (sr, st) -> (k, P0, P2)."""
    results = {}
    n_total = len(sigmas) ** 2
    count = 0
    for sr in sigmas:
        for st in sigmas:
            count += 1
            # dedicated RNG per config so results are reproducible and independent
            seed = int(rng_seed) + count
            np.random.seed(seed)
            if sr == 0.0 and st == 0.0:
                pos_noisy = gpos_rsd.copy()
            else:
                pos_noisy = gpos_rsd.copy()
                pos_noisy = noise_positions(
                    pos_noisy, ra=0.0, dec=90.0,
                    noise_radial=sr, noise_transverse=st,
                )
                pos_noisy %= L
            k, P0, P2 = measure_pk(pos_noisy, L, N_mesh, threads=threads)
            results[(float(sr), float(st))] = (k, P0, P2)
            logging.info(
                f'  [{count}/{n_total}] sigma_r={sr}, sigma_t={st} -> '
                f'P0(k~0.1)~{np.interp(0.1, k, P0):.1f}')
    return results


def _palette(cmap_name, sigma_pairs):
    """Return solid colors only: blue for truth, red for voxel."""
    color = 'blue' if 'Blue' in cmap_name else 'red'
    return [color] * len(sigma_pairs), None


def _family_curves(results, sigma_pairs, k_grid, k_ref, P0_ref, P2_ref, kind):
    """Return array shape (n_sp, len(k_grid)) of the curve for given `kind` on k_grid.

    kind:
        'P0'         -> P_0(k)
        'P2'         -> P_2(k)
        'kP2'        -> k * P_2(k)
        'P0_ratio'   -> P_0(k) / P_0_ref(k)
        'dkP2'       -> k * P_2(k) - k * P_2_ref(k)
    """
    out = np.zeros((len(sigma_pairs), len(k_grid)))
    P0_ref_g = np.interp(k_grid, k_ref, P0_ref)
    P2_ref_g = np.interp(k_grid, k_ref, P2_ref)
    for i, sp in enumerate(sigma_pairs):
        k, P0, P2 = results[sp]
        P0_g = np.interp(k_grid, k, P0)
        P2_g = np.interp(k_grid, k, P2)
        if kind == 'P0':
            out[i] = P0_g
        elif kind == 'P2':
            out[i] = P2_g
        elif kind == 'kP2':
            out[i] = k_grid * P2_g
        elif kind == 'P0_ratio':
            out[i] = P0_g / P0_ref_g
        elif kind == 'dkP2':
            out[i] = k_grid * P2_g - k_grid * P2_ref_g
        else:
            raise ValueError(kind)
    return out


def _bin_avg(k, arr, factor=2):
    """Downsample by averaging every `factor` consecutive bins.

    arr may be 1D (shape (nk,)) or 2D (shape (n_curves, nk)).
    """
    nk = len(k)
    m = (nk // factor) * factor
    k = k[:m]
    k_new = k.reshape(-1, factor).mean(axis=1)
    if arr.ndim == 1:
        arr = arr[:m]
        arr_new = arr.reshape(-1, factor).mean(axis=1)
    else:
        arr = arr[:, :m]
        arr_new = arr.reshape(arr.shape[0], -1, factor).mean(axis=2)
    return k_new, arr_new


def _shade_overlap(ax, k_grid, arr_a, arr_b, color='purple', alpha=0.3, label=None):
    """Shade the k-range where family A's envelope intersects family B's envelope."""
    a_lo, a_hi = arr_a.min(axis=0), arr_a.max(axis=0)
    b_lo, b_hi = arr_b.min(axis=0), arr_b.max(axis=0)
    lo = np.maximum(a_lo, b_lo)
    hi = np.minimum(a_hi, b_hi)
    overlap = hi >= lo
    if not np.any(overlap):
        return
    lo_plot = np.where(overlap, lo, np.nan)
    hi_plot = np.where(overlap, hi, np.nan)
    ax.fill_between(k_grid, lo_plot, hi_plot, color=color, alpha=alpha,
                    label=label, linewidth=0)


def load_results_npz(npz_path):
    """Reload results dicts saved by the sweep. Returns (results_truth, results_voxel)."""
    data = np.load(npz_path)
    keys = list(data.keys())
    results = {'truth': {}, 'voxel': {}}
    for key in keys:
        if key in ('sigmas', 'seed', 'kmax'):
            continue
        # expected format: '<tag>_sr<X>_st<Y>_<field>' where X,Y use 'p' for '.'
        try:
            tag, sr_part, st_field = key.split('_', 2)
            # st_field is like 'st0p0_k' or 'st0p5_P0'
            sr_val = float(sr_part.replace('sr', '').replace('p', '.'))
            st_part, field = st_field.split('_', 1)
            st_val = float(st_part.replace('st', '').replace('p', '.'))
        except Exception:
            continue
        sp = (sr_val, st_val)
        if sp not in results[tag]:
            results[tag][sp] = {}
        results[tag][sp][field] = data[key]
    # convert inner dicts to (k, P0, P2) tuples
    out_t = {sp: (d['k'], d['P0'], d['P2'])
             for sp, d in results['truth'].items()}
    out_v = {sp: (d['k'], d['P0'], d['P2'])
             for sp, d in results['voxel'].items()}
    return out_t, out_v


def make_figure_1(results_truth, results_voxel, kmax, out_path,
                  bin_factor=2):
    """Four-panel money plot: P_0, k·P_2 and their deviations for both families.

    Only the diagonal (sigma_r == sigma_t) subset of curves is plotted, at
    higher opacity. Curves are downsampled by `bin_factor` via bin-averaging.
    """
    # Subset: diagonal sigma_pairs (sr == st) — 6 curves per family
    all_pairs = list(results_truth.keys())
    idxs = np.random.choice(len(all_pairs), size=15, replace=False)
    sigma_pairs = [all_pairs[i] for i in idxs]
    sigma_pairs.sort()
    colors_t, _ = _palette('Blues_r', sigma_pairs)
    colors_v, _ = _palette('Reds_r', sigma_pairs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Reference: truth at sigma=(0,0); bin-averaged version for plotting
    k_ref_raw, P0_ref_raw, P2_ref_raw = results_truth[(0.0, 0.0)]
    kmask = k_ref_raw <= kmax
    k_grid_full = k_ref_raw[kmask]
    k_grid, _ = _bin_avg(k_grid_full, k_grid_full, bin_factor)
    P0_ref = np.interp(k_grid, k_ref_raw, P0_ref_raw)
    P2_ref = np.interp(k_grid, k_ref_raw, P2_ref_raw)

    # Envelope shading uses the FULL sigma grid so overlap is conservative
    curves_P0_t_full = _family_curves(results_truth, all_pairs, k_grid,
                                      k_ref_raw, P0_ref_raw, P2_ref_raw, 'P0')
    curves_P0_v_full = _family_curves(results_voxel, all_pairs, k_grid,
                                      k_ref_raw, P0_ref_raw, P2_ref_raw, 'P0')
    curves_kP2_t_full = _family_curves(results_truth, all_pairs, k_grid,
                                       k_ref_raw, P0_ref_raw, P2_ref_raw, 'kP2')
    curves_kP2_v_full = _family_curves(results_voxel, all_pairs, k_grid,
                                       k_ref_raw, P0_ref_raw, P2_ref_raw, 'kP2')
    curves_rP0_t_full = _family_curves(results_truth, all_pairs, k_grid,
                                       k_ref_raw, P0_ref_raw, P2_ref_raw, 'P0_ratio')
    curves_rP0_v_full = _family_curves(results_voxel, all_pairs, k_grid,
                                       k_ref_raw, P0_ref_raw, P2_ref_raw, 'P0_ratio')
    curves_dkP2_t_full = _family_curves(results_truth, all_pairs, k_grid,
                                        k_ref_raw, P0_ref_raw, P2_ref_raw, 'dkP2')
    curves_dkP2_v_full = _family_curves(results_voxel, all_pairs, k_grid,
                                        k_ref_raw, P0_ref_raw, P2_ref_raw, 'dkP2')

    alpha_line = 0.8

    def _curve(results, sp, kind):
        """Return bin-averaged (k, y) for a given (results, sigma_pair, kind)."""
        k, P0, P2 = results[sp]
        m = k <= kmax
        k_m = k[m]
        if kind == 'P0':
            y = P0[m]
        elif kind == 'kP2':
            y = k_m * P2[m]
        elif kind == 'P0_ratio':
            y = P0[m] / np.interp(k_m, k_ref_raw, P0_ref_raw)
        elif kind == 'dkP2':
            y = k_m * P2[m] - k_m * np.interp(k_m, k_ref_raw, P2_ref_raw)
        else:
            raise ValueError(kind)
        k_b, y_b = _bin_avg(k_m, y, bin_factor)
        return k_b, y_b

    # Top-left: P_0 (loglog)
    ax = axes[0, 0]
    _shade_overlap(ax, k_grid, curves_P0_t_full, curves_P0_v_full,
                   label='envelope overlap')
    for sp, ct, cv in zip(sigma_pairs, colors_t, colors_v):
        kt, yt = _curve(results_truth, sp, 'P0')
        kv, yv = _curve(results_voxel, sp, 'P0')
        ax.plot(kt, yt, color=ct, alpha=alpha_line, lw=1.3)
        ax.plot(kv, yv, color=cv, alpha=alpha_line, lw=1.3)
    ax.plot(k_grid, P0_ref, color='k', lw=1.8, label=r'G_truth $\sigma=0$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(right=kmax)
    ax.set_xlabel(r'$k\ [h/\mathrm{Mpc}]$')
    ax.set_ylabel(r'$P_0(k)\ [(\mathrm{Mpc}/h)^3]$')
    ax.set_title(r'Monopole $P_0(k)$')
    ax.legend(loc='lower left', fontsize=9)

    # Top-right: k*P_2 (semilogx, linear y)
    ax = axes[0, 1]
    _shade_overlap(ax, k_grid, curves_kP2_t_full, curves_kP2_v_full)
    for sp, ct, cv in zip(sigma_pairs, colors_t, colors_v):
        kt, yt = _curve(results_truth, sp, 'kP2')
        kv, yv = _curve(results_voxel, sp, 'kP2')
        ax.plot(kt, yt, color=ct, alpha=alpha_line, lw=1.3)
        ax.plot(kv, yv, color=cv, alpha=alpha_line, lw=1.3)
    ax.plot(k_grid, k_grid * P2_ref, color='k', lw=1.8)
    ax.axhline(0.0, color='k', lw=0.6, ls=':')
    ax.set_xscale('log')
    ax.set_xlim(right=kmax)
    ax.set_xlabel(r'$k\ [h/\mathrm{Mpc}]$')
    ax.set_ylabel(r'$k\, P_2(k)\ [(\mathrm{Mpc}/h)^2]$')
    ax.set_title(r'Quadrupole $k\, P_2(k)$')

    # Bottom-left: P_0 ratio (linear x)
    ax = axes[1, 0]
    _shade_overlap(ax, k_grid, curves_rP0_t_full, curves_rP0_v_full)
    for sp, ct, cv in zip(sigma_pairs, colors_t, colors_v):
        kt, yt = _curve(results_truth, sp, 'P0_ratio')
        kv, yv = _curve(results_voxel, sp, 'P0_ratio')
        ax.plot(kt, yt, color=ct, alpha=alpha_line, lw=1.3)
        ax.plot(kv, yv, color=cv, alpha=alpha_line, lw=1.3)
    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_xlim(0, kmax)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel(r'$k\ [h/\mathrm{Mpc}]$')
    ax.set_ylabel(r'$P_0(k) / P_0^{\mathrm{truth},\sigma=0}(k)$')
    ax.set_title('Monopole ratio')

    # Bottom-right: difference k*P_2 - (k*P_2)_ref (linear x, zoom to overlap)
    ax = axes[1, 1]
    _shade_overlap(ax, k_grid, curves_dkP2_t_full, curves_dkP2_v_full)
    for sp, ct, cv in zip(sigma_pairs, colors_t, colors_v):
        kt, yt = _curve(results_truth, sp, 'dkP2')
        kv, yv = _curve(results_voxel, sp, 'dkP2')
        ax.plot(kt, yt, color=ct, alpha=alpha_line, lw=1.3)
        ax.plot(kv, yv, color=cv, alpha=alpha_line, lw=1.3)
    ax.axhline(0.0, color='k', lw=0.8, ls=':')
    ax.set_xlim(0, kmax)
    # zoom y to the overlap band, expanded by 30% margin
    olo = np.maximum(curves_dkP2_t_full.min(axis=0),
                     curves_dkP2_v_full.min(axis=0))
    ohi = np.minimum(curves_dkP2_t_full.max(axis=0),
                     curves_dkP2_v_full.max(axis=0))
    valid = ohi >= olo
    if np.any(valid):
        y0, y1 = olo[valid].min(), ohi[valid].max()
        pad = 0.3 * (y1 - y0 + 1e-12)
        ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_xlabel(r'$k\ [h/\mathrm{Mpc}]$')
    ax.set_ylabel(r'$k\, [P_2(k) - P_2^{\mathrm{truth},\sigma=0}(k)]$')
    ax.set_title(r'$k\, P_2$ deviation (zoom on overlap)')

    # Legends: one swatch per family + overlap band
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color=plt.get_cmap('Blues_r')(0.5), lw=2,
               label='G_truth family'),
        Line2D([0], [0], color=plt.get_cmap('Reds_r')(0.5), lw=2,
               label='G_voxel family'),
        Patch(facecolor='purple', alpha=0.25, label='envelope overlap'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(
        'Voxelization diagnostic: P(k) multipoles (diagonal sigma subset; '
        'purple = min/max overlap of full 36×2 grid)',
        y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'Saved figure 1 to {out_path}')


def make_figure_pointwise_bestfit(results_truth, results_voxel, kmax, out_path,
                                  match_on='P0',
                                  k_stars=(0.02, 0.05, 0.1, 0.2, 0.3, 0.4)):
    """For each reference k*, find the (truth, voxel) pair that are closest 
    to each other at k* on the chosen multipole (`match_on`).
    Then plot, across all k, the P_0 ratio and k*P_2 difference between 
    that best matched voxel and truth pair.
    """
    sigma_pairs_t = sorted(results_truth.keys())
    sigma_pairs_v = sorted(results_voxel.keys())

    k_ref = results_truth[(0.0, 0.0)][0]
    mask = k_ref <= kmax
    k_u = k_ref[mask]

    def _resample(results, kg):
        out = {}
        for sp, (k, P0, P2) in results.items():
            out[sp] = (np.interp(kg, k, P0), np.interp(kg, k, P2))
        return out

    T = _resample(results_truth, k_u)
    V = _resample(results_voxel, k_u)

    # Filter k_stars that are inside k_u
    k_stars = [ks for ks in k_stars if (ks >= k_u.min() and ks <= k_u.max())]

    colors = plt.get_cmap('viridis')(np.linspace(0.1, 0.9, len(k_stars)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # --- Extra: global best fit over all k (mode-weighted chi^2 on `match_on`) ---
    dk_mode = np.median(np.diff(k_u)) if len(k_u) > 1 else 1.0
    w_mode = k_u ** 2 * dk_mode
    w_mode = w_mode / w_mode.sum()

    best_all_key_t = None
    best_all_key_v = None
    best_all_chi2 = np.inf

    for sp_t in sigma_pairs_t:
        P0t, P2t = T[sp_t]
        for sp_v in sigma_pairs_v:
            P0v, P2v = V[sp_v]

            if match_on == 'P0':
                resid2 = (P0v - P0t) ** 2 / (P0t ** 2 + 1e-30)
            else:
                resid2 = (P2v - P2t) ** 2 / (P0t ** 2 + 1e-30)

            chi2 = float(np.sum(w_mode * resid2))
            if chi2 < best_all_chi2:
                best_all_chi2 = chi2
                best_all_key_t = sp_t
                best_all_key_v = sp_v

    P0t_all, P2t_all = T[best_all_key_t]
    P0v_all, P2v_all = V[best_all_key_v]

    ratio_P0_all = P0v_all / P0t_all
    diff_kP2_all = k_u * (P2v_all - P2t_all)

    label_all = (rf'all-$k$ $\to$ T:{best_all_key_t}, V:{best_all_key_v}')
    axes[0].plot(k_u, ratio_P0_all, color='k', lw=2.2, ls='--', label=label_all)
    axes[1].plot(k_u, diff_kP2_all, color='k', lw=2.2, ls='--')

    logging.info(
        f'  all-k match ({match_on}): truth {best_all_key_t} -> '
        f'voxel {best_all_key_v}, chi2={best_all_chi2:.3g}')

    # --- Pointwise best fit matching for each k* ---
    matched_info = []
    for ks, color in zip(k_stars, colors):
        best_key_t = None
        best_key_v = None
        best_diff = np.inf

        for sp_t in sigma_pairs_t:
            P0t, P2t = T[sp_t]
            val_t = np.interp(
                ks, k_u, P0t) if match_on == 'P0' else np.interp(ks, k_u, P2t)

            for sp_v in sigma_pairs_v:
                P0v, P2v = V[sp_v]
                val_v = np.interp(
                    ks, k_u, P0v) if match_on == 'P0' else np.interp(ks, k_u, P2v)

                diff = abs(val_t - val_v)
                if diff < best_diff:
                    best_diff = diff
                    best_key_t = sp_t
                    best_key_v = sp_v

        matched_info.append((ks, best_key_t, best_key_v))

        P0t, P2t = T[best_key_t]
        P0v, P2v = V[best_key_v]

        ratio_P0 = P0v / P0t
        diff_kP2 = k_u * (P2v - P2t)

        label = rf'$k^*={ks:.2f} \to$ T:({best_key_t[0]:g},{best_key_t[1]:g}), V:({best_key_v[0]:g},{best_key_v[1]:g})'
        axes[0].plot(k_u, ratio_P0, color=color, lw=1.5, label=label)
        axes[1].plot(k_u, diff_kP2, color=color, lw=1.5)

        # mark the reference k*
        axes[0].axvline(ks, color=color, lw=0.6, ls=':', alpha=0.7)
        axes[1].axvline(ks, color=color, lw=0.6, ls=':', alpha=0.7)

    for ax in axes:
        # ax.set_xscale('log')
        ax.set_xlim(k_u.min(), kmax)
        kvox = np.pi / (1000.0 / 128.0)
        ax.axvline(kvox, color='gray', lw=0.8, ls='--',
                   label=rf'$k_{{\rm vox}}\approx{kvox:.2f}$'
                   if ax is axes[0] else None)

    axes[0].axhline(1.0, color='k', lw=0.6, ls=':')
    axes[0].set_ylabel(r'$P_0^{\rm voxel}(k) / P_0^{\rm truth\,best-fit}(k)$')
    axes[0].set_title(
        f'Pointwise All-to-All best fit at $k^*$ (matched on ${match_on}$)')

    # Legend can get large, bounding box outside might help if there are many k*
    axes[0].legend(fontsize=8, loc='best')

    axes[1].axhline(0.0, color='k', lw=0.6, ls=':')
    axes[1].set_ylabel(
        r'$k\,[P_2^{\rm voxel}(k) - P_2^{\rm truth\,best-fit}(k)]$')
    axes[1].set_xlabel(r'$k\ [h/\mathrm{Mpc}]$')

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logging.info(f'Saved pointwise best-fit ({match_on}) figure to {out_path}')
    for ks, bkt, bkv in matched_info:
        logging.info(f'  k*={ks:.3f} -> truth {bkt} matched with voxel {bkv}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--halo-path', default=DEFAULT_HALO_PATH)
    parser.add_argument('--config-path', default=DEFAULT_CONFIG_PATH,
                        help='Existing config.yaml with nbody.cosmo for the lhid')
    parser.add_argument('--bias-yaml', default=DEFAULT_BIAS_YAML)
    parser.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kmax', type=float, default=0.4)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--lhid', type=int, default=48)
    parser.add_argument('--n-mesh', type=int, default=256,
                        help='mesh size for P(k) measurement (default 256 for k<=0.4)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    os.makedirs(args.out_dir, exist_ok=True)
    logging.info(f'Seed: {args.seed}')

    npz_path = os.path.join(args.out_dir, 'voxel_diagnostic_results.npz')

    if os.path.exists(npz_path):
        logging.info(
            f'Found existing {npz_path} — loading instead of recomputing.')
        results_truth, results_voxel = load_results_npz(npz_path)
    else:
        # Load cosmo from existing config
        src_cfg = OmegaConf.load(args.config_path)
        cosmo_list = list(src_cfg.nbody.cosmo)
        logging.info(f'Cosmo [Om, Ob, h, ns, s8]: {cosmo_list}')
        cosmo_astropy = FlatLambdaCDM(
            H0=cosmo_list[2] * 100, Om0=cosmo_list[0], Ob0=cosmo_list[1])

        # Load halos
        hpos, hvel, hmass, a, z = load_halos(args.halo_path)
        L = 1000.0
        N_vox = 128
        assert abs(z - 0.5) < 1e-3, f'Expected z=0.5, got z={z}'

        # Build HOD cfg (shared between branches)
        cfg = build_hod_cfg(args.bias_yaml, cosmo_list, int(L), N_vox,
                            lhid=args.lhid, redshift=z)
        logging.info('HOD model: ' + str(cfg.bias.hod.model))
        logging.info('HOD theta: ' + str(dict(cfg.bias.hod.theta)))

        rng = np.random.default_rng(args.seed)

        # --- Branch A: ground truth ---
        logging.info('Branch A: running HOD on non-voxelized halos...')
        gpos_A, gvel_A = run_hod(hpos, hvel, hmass, cosmo_astropy, L, z,
                                 cfg, populate_seed=args.seed)
        logging.info(f'Branch A: N_gal={len(gpos_A)}')

        # --- Branch B: voxelized ---
        logging.info(
            'Branch B: voxelizing halos and redistributing uniformly...')
        hpos_vox = voxelize_uniform(hpos, L, N_vox, rng)
        logging.info('Branch B: running HOD on voxelized halos...')
        gpos_B, gvel_B = run_hod(hpos_vox, hvel, hmass, cosmo_astropy, L, z,
                                 cfg, populate_seed=args.seed)
        logging.info(f'Branch B: N_gal={len(gpos_B)}')

        # Apply RSD along z
        logging.info('Applying RSD along z-axis to both branches...')
        gpos_A_rsd = apply_rsd_z(gpos_A, gvel_A, L, cosmo_astropy, z)
        gpos_B_rsd = apply_rsd_z(gpos_B, gvel_B, L, cosmo_astropy, z)

        # Sweep noise and measure P(k)
        logging.info('Sweeping noise grid for Branch A (G_truth)...')
        results_truth = sweep_noise(
            gpos_A_rsd, L, args.n_mesh, SIGMAS,
            rng_seed=args.seed * 1000, threads=args.threads)
        logging.info('Sweeping noise grid for Branch B (G_voxel)...')
        results_voxel = sweep_noise(
            gpos_B_rsd, L, args.n_mesh, SIGMAS,
            rng_seed=args.seed * 1000 + 500, threads=args.threads)

        # Save results .npz
        save_d = {}
        for tag, res in [('truth', results_truth), ('voxel', results_voxel)]:
            for (sr, st), (k, P0, P2) in res.items():
                key = f'{tag}_sr{sr}_st{st}'.replace('.', 'p')
                save_d[f'{key}_k'] = k
                save_d[f'{key}_P0'] = P0
                save_d[f'{key}_P2'] = P2
        save_d['sigmas'] = np.array(SIGMAS)
        save_d['seed'] = np.array([args.seed])
        save_d['kmax'] = np.array([args.kmax])
        np.savez(npz_path, **save_d)
        logging.info(f'Saved {len(save_d)} arrays to {npz_path}')

    # Figures
    f1 = os.path.join(args.out_dir, 'voxel_diagnostic_Pk_families.png')
    f_p0 = os.path.join(args.out_dir, 'voxel_diagnostic_bestfit_P0match.png')
    f_p2 = os.path.join(args.out_dir, 'voxel_diagnostic_bestfit_P2match.png')
    make_figure_1(results_truth, results_voxel, args.kmax, f1)
    make_figure_pointwise_bestfit(
        results_truth, results_voxel, args.kmax, f_p0, match_on='P0')
    make_figure_pointwise_bestfit(
        results_truth, results_voxel, args.kmax, f_p2, match_on='P2')

    logging.info('Done.')


if __name__ == '__main__':
    main()
