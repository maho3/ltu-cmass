"""
Pair-count histograms for galaxies from three halo catalogs, in real space.

For each of `num_rows` random Quijote lhids, build three galaxy catalogs and
compute the pair-count histogram on a common random subsample:

  (a) Quijote          : HOD on original Quijote halos.
  (b) voxelized Quijote : HOD on Quijote halos snapped to voxel centers and
                          uniformly redistributed within the voxel.
  (c) CHARM            : HOD on CHARM halos living at the same path convention
                          under quijotelike/fastpm/L1000-N128/{lhid}/halos.h5.

Pair counts use periodic boundary conditions (L=1000 Mpc/h) via scipy's
cKDTree and are binned in 1 Mpc/h bins from 0 to 20 Mpc/h.

Each panel title reports the average number of galaxies per voxel
(N_gal / N^3 with N=128) for each of the three catalogs. A dotted vertical
line at the voxel size Δ = 1000/128 ≈ 7.8125 Mpc/h is drawn on every panel.

Output: figures/galaxy_pair_counts.png
"""

from cmass.bias.tools.hod import parse_hod
from cmass.bias.apply_hod import populate_hod
from omegaconf import OmegaConf
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse
import copy
import logging
import os
import random
from os.path import join

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')


DEFAULT_WDIR = '/work/hdd/bdne/maho3/cmass-ili'
DEFAULT_BIAS_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'cmass', 'conf', 'bias', 'zheng_biased.yaml'
)

L = 1000.0
N = 128
N_VOX_TOTAL = N ** 3
DELTA = L / N


def load_halos(path):
    with h5py.File(path, 'r') as f:
        a_key = list(f.keys())[0]
        g = f[a_key]
        hpos = g['pos'][...].astype(np.float64)
        hvel = g['vel'][...].astype(np.float64)
        hmass = g['mass'][...].astype(np.float64)
    a = float(a_key)
    z = 1.0 / a - 1.0
    return hpos, hvel, hmass, a, z


def load_cosmo(wdir, lhid):
    path = join(wdir, 'quijote', 'nbody', f'L1000-N128', str(lhid),
                'config.yaml')
    cfg = OmegaConf.load(path)
    return list(cfg.nbody.cosmo)


def voxelize_uniform(hpos, rng):
    idx = np.clip(np.floor(hpos / DELTA).astype(np.int64), 0, N - 1)
    centers = (idx + 0.5) * DELTA
    offsets = rng.uniform(-DELTA / 2.0, DELTA / 2.0, size=hpos.shape)
    return (centers + offsets) % L


def build_hod_cfg(bias_yaml_path, cosmo_list, lhid, redshift):
    bias_cfg = OmegaConf.load(bias_yaml_path)
    bias_cfg.hod.noise_uniform = False
    bias_cfg.hod.seed = 0
    cfg = OmegaConf.create({
        'meta': {'wdir': '.'},
        'sim': 'quijote',
        'nbody': {
            'suite': 'quijote',
            'L': int(L), 'N': int(N), 'lhid': lhid, 'zf': redshift,
            'cosmo': list(cosmo_list),
        },
        'bias': OmegaConf.to_container(bias_cfg, resolve=True),
    })
    return parse_hod(cfg)


def run_hod(hpos, hvel, hmass, cosmo, redshift, cfg, populate_seed):
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
    gpos = np.array([galcat['x'], galcat['y'], galcat['z']]
                    ).T.astype(np.float64)
    return gpos % L


def pair_count_hist(pos, bin_edges, boxsize):
    """DD pair-count histogram with periodic BCs. Returns counts per bin."""
    tree = cKDTree(pos, boxsize=boxsize)
    cum = tree.count_neighbors(tree, bin_edges)
    hist = np.diff(cum).astype(np.float64)
    # Remove self-pairs from the first bin (N of them, distance 0 ≤ edge)
    hist[0] -= len(pos)
    # Each pair (i,j) counted in both orders (i,j) and (j,i); de-duplicate.
    hist /= 2.0
    return hist


def process_lhid(lhid, wdir, bias_yaml, seed, n_sub):
    """Returns dict: label -> (pair_hist, n_gal_per_voxel)."""
    logging.info(f'=== lhid {lhid} ===')
    quij_path = join(wdir, 'quijote', 'nbody', f'L1000-N128',
                     str(lhid), 'halos.h5')
    charm_path = join(wdir, 'quijotelike', 'fastpm', f'L1000-N128',
                      str(lhid), 'halos.h5')

    cosmo_list = load_cosmo(wdir, lhid)
    cosmo = FlatLambdaCDM(H0=cosmo_list[2] * 100,
                          Om0=cosmo_list[0], Ob0=cosmo_list[1])

    hpos_q, hvel_q, hmass_q, _, z = load_halos(quij_path)
    hpos_c, hvel_c, hmass_c, _, _ = load_halos(charm_path)
    logging.info(f'  N_halos: quijote={len(hpos_q)}, CHARM={len(hpos_c)}')

    cfg = build_hod_cfg(bias_yaml, cosmo_list, lhid, z)

    # (a) Quijote galaxies
    gpos_quij = run_hod(hpos_q, hvel_q, hmass_q, cosmo, z, cfg,
                        populate_seed=seed)
    # (b) voxelized-Quijote galaxies
    rng = np.random.default_rng(int(seed) + int(lhid))
    hpos_q_vox = voxelize_uniform(hpos_q, rng)
    gpos_vox = run_hod(hpos_q_vox, hvel_q, hmass_q, cosmo, z, cfg,
                       populate_seed=seed)
    # (c) CHARM galaxies
    hpos_c_vox = voxelize_uniform(hpos_c, rng)
    gpos_charm = run_hod(hpos_c_vox, hvel_c, hmass_c, cosmo, z, cfg,
                         populate_seed=seed)

    catalogs = {
        'Quijote': gpos_quij,
        'voxelized Quijote': gpos_vox,
        'CHARM': gpos_charm,
    }
    logging.info('  N_gal: '
                 + ', '.join(f'{k}={len(v)}' for k, v in catalogs.items()))

    bin_edges = np.arange(0.0, 21.0, 1.0)  # 0..20 Mpc/h in 1 Mpc/h bins
    rng_sub = np.random.default_rng(int(seed) * 31 + int(lhid))

    out = {}
    for name, gpos in catalogs.items():
        n_per_vox = len(gpos) / N_VOX_TOTAL
        # Common-size random subsample for fair shape comparison
        k = min(n_sub, len(gpos))
        idx = rng_sub.choice(len(gpos), size=k, replace=False)
        sub = gpos[idx]
        hist = pair_count_hist(sub, bin_edges, boxsize=L)
        out[name] = (hist, n_per_vox, k)
    return out, bin_edges


def make_figure(per_lhid, lhids, bin_edges, out_path, n_sub):
    nrow = len(lhids)
    fig, axes = plt.subplots(nrow, 1, figsize=(9, 3.3 * nrow), sharex=True)
    if nrow == 1:
        axes = np.array([axes])

    colors = {'Quijote': 'black',
              'voxelized Quijote': 'darkorange',
              'CHARM': 'purple'}
    bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for ir, lhid in enumerate(lhids):
        ax = axes[ir]
        out = per_lhid[lhid]
        nvox_parts = []
        for name, (hist, n_per_vox, k_used) in out.items():
            ax.plot(bin_mid, hist, color=colors[name], lw=1.6,
                    label=f'{name} (subsample {k_used})')
            nvox_parts.append(f'{name}: {n_per_vox:.3f}')
        ax.axvline(DELTA, color='black', lw=0.9, ls=':',
                   label=rf'$\Delta = L/N \approx {DELTA:.3f}$ Mpc/h')
        # ax.set_yscale('log')
        ax.set_ylim(0)
        ax.set_ylabel('DD pair count / bin')
        ax.set_title(
            f'lhid={lhid}  |  avg gal / voxel — '
            + ', '.join(nvox_parts),
            fontsize=10,
        )
        if ir == 0:
            ax.legend(fontsize=8, loc='upper left')

    axes[-1].set_xlabel(r'separation $r$ [Mpc/h]')
    fig.suptitle(
        'Real-space pair-count histograms of galaxy catalogs '
        f'(subsample size up to {n_sub})',
        fontsize=12,
    )
    fig.subplots_adjust(top=0.95)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'Saved figure to {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wdir', default=DEFAULT_WDIR)
    parser.add_argument('--bias-yaml', default=DEFAULT_BIAS_YAML)
    parser.add_argument('--out-dir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'figures'))
    parser.add_argument('--num-rows', type=int, default=5)
    parser.add_argument('--lhids', type=int, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-sub', type=int, default=10000,
                        help='max galaxies per catalog used in the pair count')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s')
    os.makedirs(args.out_dir, exist_ok=True)

    # Discover lhids that exist in BOTH the Quijote and CHARM halo dirs
    quij_base = join(args.wdir, 'quijote', 'nbody', 'L1000-N128')
    charm_base = join(args.wdir, 'quijotelike', 'fastpm', 'L1000-N128')
    have_quij = set(int(x) for x in os.listdir(quij_base) if x.isdigit())
    have_charm = set(
        int(x) for x in os.listdir(charm_base) if x.isdigit()
        and os.path.isfile(join(charm_base, x, 'halos.h5'))
    )
    available = sorted(have_quij & have_charm)
    logging.info(f'Found {len(available)} lhids with both catalogs.')

    if args.lhids:
        lhids = args.lhids
    else:
        rng = random.Random(args.seed)
        lhids = rng.sample(available, args.num_rows)
    logging.info(f'Using lhids: {lhids}')

    per_lhid = {}
    bin_edges = None
    for lhid in lhids:
        out, bin_edges = process_lhid(
            lhid, args.wdir, args.bias_yaml, args.seed, args.n_sub)
        per_lhid[lhid] = out

    out_path = join(args.out_dir, 'galaxy_pair_counts.png')
    make_figure(per_lhid, lhids, bin_edges, out_path, args.n_sub)


if __name__ == '__main__':
    main()
