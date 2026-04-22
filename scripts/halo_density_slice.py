"""
Visualize Quijote halo/galaxy distributions against the underlying DM density.

For `num_rows` random Quijote lhids, produce a `num_rows x 2` grid:

- Column 1: original halo positions + galaxies from the standard HOD.
- Column 2: halo positions snapped to voxel centers and uniformly redistributed,
  then galaxies from the HOD on those voxelized halos.

Each subplot shows an 8x8 voxel (x,y) slice, 1 voxel thick in z, with:
  - DM overdensity log(1+delta) in the background (Quijotelike FastPM field,
    matched ICs),
  - halos as black circles of physical radius R200c,
  - centrals as red dots, satellites as blue dots.

Real space by default; pass --rsd to view in redshift space along z.

Data paths (matched ICs between the two suites):
  halos:   <wdir>/quijote/nbody/L1000-N128/{lhid}/halos.h5
  density: <wdir>/quijotelike/fastpm/L1000-N128/{lhid}/nbody.h5
"""

from halotools.empirical_models import halo_mass_to_halo_radius
from cmass.bias.tools.hod import parse_hod
from cmass.bias.apply_hod import populate_hod
from omegaconf import OmegaConf
from astropy.cosmology import FlatLambdaCDM
from matplotlib.patches import Circle
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


def load_halos(wdir, lhid):
    path = join(wdir, 'quijote', 'nbody', f'L1000-N128', str(lhid), 'halos.h5')
    with h5py.File(path, 'r') as f:
        a_key = list(f.keys())[0]
        g = f[a_key]
        hpos = g['pos'][...].astype(np.float64)
        hvel = g['vel'][...].astype(np.float64)
        hmass = g['mass'][...].astype(np.float64)
    a = float(a_key)
    z = 1.0 / a - 1.0
    return hpos, hvel, hmass, a, z


def load_density(wdir, lhid):
    path = join(wdir, 'quijotelike', 'fastpm', f'L1000-N128',
                str(lhid), 'nbody.h5')
    with h5py.File(path, 'r') as f:
        a_key = list(f.keys())[0]
        rho = f[a_key]['rho'][...].astype(np.float32)
    return rho


def load_cosmo(wdir, lhid):
    path = join(wdir, 'quijote', 'nbody', f'L1000-N128', str(lhid),
                'config.yaml')
    cfg = OmegaConf.load(path)
    return list(cfg.nbody.cosmo)


def voxelize_uniform(hpos, L, N, rng):
    delta = L / N
    idx = np.clip(np.floor(hpos / delta).astype(np.int64), 0, N - 1)
    centers = (idx + 0.5) * delta
    offsets = rng.uniform(-delta / 2.0, delta / 2.0, size=hpos.shape)
    pos_new = (centers + offsets) % L
    return pos_new


def build_hod_cfg(bias_yaml_path, cosmo_list, L, N, lhid, redshift):
    bias_cfg = OmegaConf.load(bias_yaml_path)
    bias_cfg.hod.noise_uniform = False
    bias_cfg.hod.seed = 0
    cfg = OmegaConf.create({
        'meta': {'wdir': '.'},
        'sim': 'quijote',
        'nbody': {
            'suite': 'quijote',
            'L': L, 'N': N, 'lhid': lhid, 'zf': redshift,
            'cosmo': list(cosmo_list),
        },
        'bias': OmegaConf.to_container(bias_cfg, resolve=True),
    })
    cfg = parse_hod(cfg)
    return cfg


def run_hod(hpos, hvel, hmass, cosmo, L, redshift, cfg, populate_seed):
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
    gvel = np.array([galcat['vx'], galcat['vy'], galcat['vz']]
                    ).T.astype(np.float64)
    gtype = np.array(galcat['gal_type'])
    return gpos, gvel, gtype


def apply_rsd_z(pos, vel, L, cosmo, z):
    """Apply RSD along z. `pos` modified & returned."""
    import redshift_space_library as RSL
    pos_out = copy.deepcopy(np.ascontiguousarray(pos.astype(np.float32)))
    vel_f = np.ascontiguousarray(vel.astype(np.float32))
    RSL.pos_redshift_space(
        pos_out, vel_f, L, cosmo.H(z).value / cosmo.h, z, 2)
    return (pos_out.astype(np.float64)) % L


def compute_R200c(log10_mass_msun_h, cosmo_astropy, redshift):
    """R200c in Mpc/h from halotools."""
    masses = 10 ** log10_mass_msun_h
    # halotools returns Mpc/h for Msun/h input with mdef='200c'
    R = halo_mass_to_halo_radius(mass=masses, cosmology=cosmo_astropy,
                                 redshift=redshift, mdef='200c')
    return np.asarray(R)


def pick_slice_center(L, N, lhid, margin=8, depth=3):
    """Pick integer (i0, j0, k0) voxel corner seeded by lhid.

    i0, j0 are in [0, N-margin]; k0 in [0, N-depth] (no wrap).
    """
    rng = np.random.default_rng(int(lhid) * 17 + 3)
    i0 = int(rng.integers(0, N - margin))
    j0 = int(rng.integers(0, N - margin))
    k0 = int(rng.integers(0, N - depth))
    return i0, j0, k0


def plot_panel(ax, rho_slice, extent_xy, halo_xy, halo_R,
               cen_xy, sat_xy, title):
    # Background: log(1+delta)
    mean = rho_slice.mean() if rho_slice.mean() > 0.5 else None
    if mean is None:
        # treat as overdensity (delta)
        one_plus_delta = 1.0 + rho_slice
    else:
        one_plus_delta = rho_slice / mean
    # protect log
    one_plus_delta = np.clip(one_plus_delta, 1e-3, None)
    bg = np.log10(one_plus_delta)

    im = ax.imshow(
        bg.T, origin='lower',
        extent=extent_xy, cmap='bone',
        interpolation='nearest',
    )

    # Halos as unfilled circles of radius R200c (data units). Use a bright
    # edge color so they stand out against the bone colormap.
    for (hx, hy), r in zip(halo_xy, halo_R):
        ax.add_patch(Circle(
            (hx, hy), radius=r,
            fill=False, edgecolor='yellow', lw=1.1, alpha=0.95))

    # Galaxies
    if len(cen_xy) > 0:
        ax.scatter(cen_xy[:, 0], cen_xy[:, 1], s=8,
                   color='red', edgecolor='none')
    if len(sat_xy) > 0:
        ax.scatter(sat_xy[:, 0], sat_xy[:, 1], s=6,
                   color='deepskyblue', edgecolor='none')

    ax.set_xlim(extent_xy[0], extent_xy[1])
    ax.set_ylim(extent_xy[2], extent_xy[3])
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x [Mpc/h]')
    ax.set_ylabel('y [Mpc/h]')
    return im


def select_in_slice(pos, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, L):
    """Return boolean mask for positions in an axis-aligned box (periodic in z)."""
    x, y, zc = pos[:, 0], pos[:, 1], pos[:, 2]
    # x and y: non-periodic (slice always interior given 8 < N margin)
    in_xy = (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)
    # z: allow periodic wrap (rare, but k0 may be near the edge)
    if z_lo >= 0 and z_hi <= L:
        in_z = (zc >= z_lo) & (zc < z_hi)
    else:
        in_z = ((zc >= z_lo % L) | (zc < z_hi % L))
    return in_xy & in_z


def process_lhid(lhid, L, N, bias_yaml, wdir, rsd, seed):
    logging.info(f'=== lhid {lhid} ===')
    hpos, hvel, hmass, a, z = load_halos(wdir, lhid)
    rho = load_density(wdir, lhid)
    cosmo_list = load_cosmo(wdir, lhid)
    cosmo = FlatLambdaCDM(H0=cosmo_list[2] * 100,
                          Om0=cosmo_list[0], Ob0=cosmo_list[1])
    logging.info(f'  N_halos={len(hpos)}, z={z:.3f}, cosmo={cosmo_list}')

    cfg = build_hod_cfg(bias_yaml, cosmo_list, int(L), int(N), lhid, z)

    # Branch A
    gpos_A, gvel_A, gtype_A = run_hod(
        hpos, hvel, hmass, cosmo, L, z, cfg, populate_seed=seed)
    # Branch B
    rng = np.random.default_rng(int(seed) + int(lhid))
    hpos_vox = voxelize_uniform(hpos, L, N, rng)
    gpos_B, gvel_B, gtype_B = run_hod(
        hpos_vox, hvel, hmass, cosmo, L, z, cfg, populate_seed=seed)

    # Halo radii (use original masses for both branches)
    R200c = compute_R200c(hmass, cosmo, z)

    if rsd:
        hpos = apply_rsd_z(hpos, hvel, L, cosmo, z)
        hpos_vox = apply_rsd_z(hpos_vox, hvel, L, cosmo, z)
        gpos_A = apply_rsd_z(gpos_A, gvel_A, L, cosmo, z)
        gpos_B = apply_rsd_z(gpos_B, gvel_B, L, cosmo, z)

    return dict(
        hpos=hpos, hpos_vox=hpos_vox, R200c=R200c,
        gpos_A=gpos_A, gtype_A=gtype_A,
        gpos_B=gpos_B, gtype_B=gtype_B,
        rho=rho, cosmo=cosmo, z=z,
    )


def make_figure(per_lhid, lhids, L, N, out_path, rsd):
    delta = L / N
    nrow = len(lhids)
    fig, axes = plt.subplots(nrow, 2, figsize=(10, 4.8 * nrow))
    if nrow == 1:
        axes = axes[None, :]

    last_im = None
    depth = 3  # voxels in z, averaged along LOS
    for ir, lhid in enumerate(lhids):
        data = per_lhid[lhid]
        i0, j0, k0 = pick_slice_center(L, N, lhid, margin=8, depth=depth)
        x_lo, x_hi = i0 * delta, (i0 + 8) * delta
        y_lo, y_hi = j0 * delta, (j0 + 8) * delta
        z_lo, z_hi = k0 * delta, (k0 + depth) * delta
        extent = (x_lo, x_hi, y_lo, y_hi)
        # Density: 8x8 voxels in (x,y), depth voxels in z averaged along LOS
        rho_slice = data['rho'][i0:i0 + 8, j0:j0 + 8,
                                k0:k0 + depth].mean(axis=2)

        for ic, which in enumerate(['A', 'B']):
            hp = data['hpos'] if which == 'A' else data['hpos_vox']
            gp = data['gpos_A'] if which == 'A' else data['gpos_B']
            gt = data['gtype_A'] if which == 'A' else data['gtype_B']

            h_mask = select_in_slice(hp, x_lo, x_hi, y_lo, y_hi,
                                     z_lo, z_hi, L)
            g_mask = select_in_slice(gp, x_lo, x_hi, y_lo, y_hi,
                                     z_lo, z_hi, L)
            halo_xy = hp[h_mask][:, :2]
            halo_R = data['R200c'][h_mask]
            gp_sel = gp[g_mask]
            gt_sel = gt[g_mask]
            # gal_type is 'centrals'/'satellites' (byte or string)
            if gt_sel.dtype.kind in ('S', 'O'):
                gt_str = np.array([g.decode() if isinstance(g, bytes) else str(g)
                                   for g in gt_sel])
            else:
                gt_str = gt_sel.astype(str)
            is_cen = gt_str == 'centrals'
            is_sat = gt_str == 'satellites'
            cen_xy = gp_sel[is_cen][:, :2]
            sat_xy = gp_sel[is_sat][:, :2]

            title = (f'lhid={lhid}  ' +
                     ('original' if which == 'A' else 'voxelized+uniform') +
                     f'  | N_h={h_mask.sum()}, N_c={is_cen.sum()}, '
                     f'N_s={is_sat.sum()}'
                     f'\n' r'z $\in$ '
                     f'[{z_lo:.1f}, {z_hi:.1f}] Mpc/h '
                     f'({depth} vox)' +
                     ('  (z-space)' if rsd else '  (real-space)'))
            im = plot_panel(axes[ir, ic], rho_slice, extent,
                            halo_xy, halo_R, cen_xy, sat_xy, title)
            last_im = im

    # Figure-level legend placed between the suptitle and the axes so it
    # does not overlap any panel. Explicit handles so all three entries
    # always appear, regardless of slice contents.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=8,
               markerfacecolor='none', markeredgecolor='yellow',
               markeredgewidth=1.2, label='halo ($R_{200c}$)'),
        Line2D([0], [0], marker='o', linestyle='', markersize=7,
               color='red', label='central'),
        Line2D([0], [0], marker='o', linestyle='', markersize=6,
               color='deepskyblue', label='satellite'),
    ]
    fig.legend(handles=legend_handles, ncol=3, loc='upper center',
               bbox_to_anchor=(0.5, 0.965), fontsize=10, framealpha=0.85)

    fig.suptitle(
        'Quijote halos vs DM density — original (left) vs voxelized (right)  '
        f'[{"z-space" if rsd else "real-space"}, 8x8x{depth} voxel slice, '
        r'$\rho$ averaged along LOS]',
        fontsize=12,
    )
    if last_im is not None:
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
        fig.colorbar(last_im, cax=cax,
                     label=r'$\log_{10}(1+\delta)$')
    # Reserve space at the top for suptitle + legend
    fig.subplots_adjust(top=0.92)
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
    parser.add_argument('--lhids', type=int, nargs='*', default=None,
                        help='explicit list of lhids (overrides random pick)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for lhid selection and HOD populate')
    parser.add_argument('--rsd', action='store_true',
                        help='show positions in redshift space along z')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s')

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover available lhids
    base = join(args.wdir, 'quijote', 'nbody', 'L1000-N128')
    available = sorted([int(x) for x in os.listdir(base) if x.isdigit()])
    # require matching density file
    dens_base = join(args.wdir, 'quijotelike', 'fastpm', 'L1000-N128')
    available_dens = set(int(x) for x in os.listdir(dens_base) if x.isdigit())
    available = [i for i in available if i in available_dens]
    logging.info(f'Found {len(available)} lhids with both halos and density.')

    if args.lhids:
        lhids = args.lhids
    else:
        rng = random.Random(args.seed)
        lhids = rng.sample(available, args.num_rows)
    logging.info(f'Using lhids: {lhids}')

    L, N = 1000.0, 128
    per_lhid = {}
    for lhid in lhids:
        per_lhid[lhid] = process_lhid(
            lhid, L, N, args.bias_yaml, args.wdir, args.rsd, args.seed)

    out_name = 'halo_density_slice'
    if args.rsd:
        out_name += '_rsd'
    out_name += '.png'
    out_path = os.path.join(args.out_dir, out_name)
    make_figure(per_lhid, lhids, L, N, out_path, args.rsd)


if __name__ == '__main__':
    main()
