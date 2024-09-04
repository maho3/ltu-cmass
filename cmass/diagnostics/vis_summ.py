"""
A script to visualise basic summary statistics for all fields generated during
the simulation.
Can be also used to compare between different runs.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config


def check_diagnostics_exist(path, diag_dir, diag_file):
    source_file = join(path, 'diag', 'halos.h5')
    if os.path.isfile(source_file):
        logging.info('Halo diagnostics already computed. Proceeding with visualisation.')
        return True
    else:
        logging.error(f'{source_file} with halo diagnostics not found.')
        return False


def extract_halo_diagnostics(file_path, group):
    with h5py.File(file_path, 'r') as f:
        # Load
        k = f[group]['Pk_k'][...]
        Pk = f[group]['Pk'][...]
        kz = f[group]['zPk_k'][...]
        Pkz = f[group]['zPk'][...]
        mass_bins = f[group]['mass_bins'][...]
        mass_hist = f[group]['mass_hist'][...]

        return k, Pk, kz, Pkz, mass_bins, mass_hist


def plot_halo_sum(source_path, L, N, out_dir, lhids):
    # TODO: Implement use of mpl style files
    # plt.style.use('/home/sding/PhD/codes/ltu-cmass/cmass/conf/diag/aa_one_column.mplstyle')

    if len(source_path) == 1:
        source_path = [source_path]
        lhids = [lhids]

    # check if diagnostics is computed
    all_diagnostics_exist = np.all([check_diagnostics_exist(path, 'diag', 'halos.h5')
                                    for path in source_path])
    if not all_diagnostics_exist:
        return False

    source_files = [join(path, 'diag', 'halos.h5') for path in source_path]
    # check for file keys
    with h5py.File(source_files[0], 'r') as f:
        alist = list(f.keys())

    outpath = join(out_dir, 'diag')
    os.makedirs(outpath, exist_ok=True)
    logging.info(f'Saving halo diagnostics to {outpath}')

    # extract diagnostics and plot them
    for a in alist:
        all_k = []
        all_Pk = []
        all_kz = []
        all_Pkz = []
        all_mass_bins = []
        all_mass_hist = []

        fig, axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
        cmap = mpl.colormaps['winter']

        # Take colors at regular intervals spanning the colormap.
        colors = cmap(np.linspace(0, 1, len(lhids)))
        cmap = (mpl.colors.ListedColormap(colors))

        for i, source_file in enumerate(source_files):
            k, Pk, kz, Pkz, mass_bins, mass_hist = extract_halo_diagnostics(source_file, a)
            all_k.append(k)
            all_Pk.append(Pk)
            all_kz.append(kz)
            all_Pkz.append(Pkz)
            all_mass_bins.append(mass_bins)
            all_mass_hist.append(mass_hist)

            axs[0].loglog(k, Pk[:, 0], color=colors[i], label=f'lhid = {lhids[i]}')
            axs[1].loglog(kz, Pkz[:, 0], color=colors[i], label=f'lhid = {lhids[i]}')
            centered_bins = 0.5 * (mass_bins[:-1] + mass_bins[1:])
            axs[2].loglog(10 ** centered_bins, mass_hist / L ** 3 / np.diff(mass_bins),
                          color=colors[i], label=f'lhid = {lhids[i]}')

            # TOASK: do we want the mean behaviour as well?

        out_file = join(outpath, f'halo_summ_group_{a}_lh_id_{lhids[0]}_to_{lhids[-1]}.svg')

        k_nyquist = np.pi / (L / N)
        axs[0].set_xlim(right=k_nyquist * 0.7)
        axs[1].set_xlim(right=k_nyquist * 0.7)

        bounds = (np.arange(len(lhids) + 1) - 0.5).tolist()
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        fig.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=axs[:],
            boundaries=bounds,
            ticks=[lhids[0], lhids[-1]],
            spacing='uniform',
            orientation='vertical',
            label='latin hypercube ids',
            location='right'
        )
        fig.savefig(out_file, format='svg')
    return True


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'sim', 'nbody', 'bias', 'diag'])
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    cfg = parse_nbody_config(cfg)

    wdir = cfg.meta.wdir
    odir = cfg.meta.odir
    suite = cfg.nbody.suite
    sim = cfg.sim
    L = cfg.nbody.L
    N = cfg.nbody.N
    n_lhid_seeds = cfg.diag.n_seeds

    if n_lhid_seeds == 1:
        lhids = cfg.nbody.lhid
        source_path = get_source_path(
            wdir, suite, sim,
            L, N, lhids
        )

        out_dir = get_source_path(
            odir, suite, sim,
            L, N, lhids,
            mkdir=True
        )
    else:
        lhids = range(cfg.diag.lhid_start, n_lhid_seeds)
        source_path = [get_source_path(
            wdir, suite, sim,
            L, N, lhid=lhid
        ) for lhid in lhids]

        out_dir = get_source_path(
            odir, suite, sim,
            L, N, lhid=0, get_cfg_dir=True
        )

    all_done = True

    # measure halo diagnostics
    done = plot_halo_sum(
        source_path, L, N, out_dir=out_dir, lhids=lhids)
    all_done &= done

    if all_done:
        logging.info('All diagnostics computed successfully')
    else:
        logging.error('Some diagnostics failed to compute')


if __name__ == "__main__":
    main()
