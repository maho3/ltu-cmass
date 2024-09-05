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
    source_file = join(path, diag_dir, diag_file)
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


def check_gen_comparison(config):
    same_wdir = config.meta.wdir == config.diag.compare.wdir
    same_nbody = config.nbody == config.diag.compare.nbody
    same_sim = config.sim == config.diag.compare.sim
    return same_wdir and same_nbody and same_sim


def plot_halo_sum(source_path, L, N, out_dir, lhids, compare_paths):
    # TODO: Implement use of mpl style files
    # plt.style.use('/home/sding/PhD/codes/ltu-cmass/cmass/conf/diag/aa_one_column.mplstyle')

    # check if diagnostics is computed
    all_diagnostics_exist = np.all([check_diagnostics_exist(path, 'diag', 'halos.h5')
                                    for path in source_path])
    if not all_diagnostics_exist:
        return False

    source_files = [join(path, 'diag', 'halos.h5') for path in source_path]
    compare_files = [join(path, 'diag', 'halos.h5') for path in compare_paths]
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

        all_k_comp = []
        all_Pk_comp = []
        all_kz_comp = []
        all_Pkz_comp = []
        all_mass_bins_comp = []
        all_mass_hist_comp = []

        fig, axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
        cmap = mpl.colormaps['winter']

        # Take colors at regular intervals spanning the colormap.
        colors = cmap(np.linspace(0, 1, len(lhids)))
        cmap = (mpl.colors.ListedColormap(colors))

        for i, source_file in enumerate(source_files):
            Pk, Pkz, k, kz, mass_bins, mass_hist = append_halo_summs(a, all_Pk, all_Pkz, all_k, all_kz, all_mass_bins,
                                                                     all_mass_hist, source_file)

            append_halo_summs(a, all_Pk_comp, all_Pkz_comp, all_k_comp,
                              all_kz_comp, all_mass_bins_comp,
                              all_mass_hist_comp, compare_files[i])

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

        fig_comp, axs_comp = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
        fig_comp_ensemble, axs_comp_ensemble = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
        Pk_ratios = []
        Pkz_ratios = []
        hmf_ratios = []

        for i, source_file in enumerate(source_files):
            ratio_Pk_i = all_Pk[i][:, 0] / all_Pk_comp[i][:, 0]
            axs_comp[0].semilogx(all_k[i], ratio_Pk_i, color=colors[i])
            ratio_Pkz_i = all_Pkz[i][:, 0] / all_Pk_comp[i][:, 0]
            axs_comp[1].semilogx(all_kz[i], ratio_Pkz_i, color=colors[i])
            centered_bins = 0.5 * (all_mass_bins[i][:-1] + all_mass_bins[i][1:])

            hmf = all_mass_hist[i] / L ** 3 / np.diff(all_mass_bins[i])
            hmf_comp = all_mass_hist_comp[i] / L ** 3 / np.diff(all_mass_bins[i])
            ratio_hmf_i = hmf / hmf_comp
            axs_comp[2].semilogx(10 ** centered_bins, ratio_hmf_i,
                                 color=colors[i])
            Pk_ratios.append(ratio_Pk_i)
            Pkz_ratios.append(ratio_Pkz_i)
            hmf_ratios.append(ratio_hmf_i)

        out_file = join(outpath, f'halo_comp_group_{a}_lh_id_{lhids[0]}_to_{lhids[-1]}.svg')

        k_nyquist = np.pi / (L / N)
        axs_comp[0].set_xlim(right=k_nyquist * 0.7)
        axs_comp[1].set_xlim(right=k_nyquist * 0.7)

        bounds = (np.arange(len(lhids) + 1) - 0.5).tolist()
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        fig_comp.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=axs_comp[:],
            boundaries=bounds,
            ticks=[lhids[0], lhids[-1]],
            spacing='uniform',
            orientation='vertical',
            label='latin hypercube ids',
            location='right'
        )
        fig_comp.savefig(out_file, format='svg')

        ## Ensemble plot
        mean_pk = np.mean(Pk_ratios, axis=0)
        std_pk = np.std(Pk_ratios, axis=0)
        axs_comp_ensemble[0].semilogx(all_k[0], mean_pk, color='k')
        axs_comp_ensemble[0].fill_between(all_k[0], mean_pk - std_pk, mean_pk + std_pk, color='k', alpha=0.2)

        mean_pk_z = np.mean(Pkz_ratios, axis=0)
        std_pk_z = np.std(Pkz_ratios, axis=0)
        axs_comp_ensemble[1].semilogx(all_kz[0], mean_pk_z, color='k')
        axs_comp_ensemble[1].fill_between(all_kz[0], mean_pk_z - std_pk_z, mean_pk_z + std_pk_z, color='k', alpha=0.2)
        mean_hmf = np.mean(hmf_ratios, axis=0)
        std_hmf = np.std(hmf_ratios, axis=0)
        axs_comp_ensemble[2].semilogx(10 ** centered_bins, mean_hmf, color='k')
        axs_comp_ensemble[2].fill_between(10 ** centered_bins, mean_hmf - std_hmf, mean_hmf + std_hmf, color='k',
                                          alpha=0.2)

        out_file = join(outpath, f'halo_comp_ensemble_group_{a}_lh_id_{lhids[0]}_to_{lhids[-1]}.svg')

        k_nyquist = np.pi / (L / N)
        axs_comp_ensemble[0].set_xlim(right=k_nyquist * 0.7)
        axs_comp_ensemble[1].set_xlim(right=k_nyquist * 0.7)

        bounds = (np.arange(len(lhids) + 1) - 0.5).tolist()
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        fig_comp_ensemble.colorbar(
            mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=axs_comp[:],
            boundaries=bounds,
            ticks=[lhids[0], lhids[-1]],
            spacing='uniform',
            orientation='vertical',
            label='latin hypercube ids',
            location='right'
        )
        fig_comp_ensemble.savefig(out_file, format='svg')

    return True


def append_halo_summs(a, all_Pk, all_Pkz, all_k, all_kz, all_mass_bins, all_mass_hist, source_file):
    k, Pk, kz, Pkz, mass_bins, mass_hist = extract_halo_diagnostics(source_file, a)
    all_k.append(k)
    all_Pk.append(Pk)
    all_kz.append(kz)
    all_Pkz.append(Pkz)
    all_mass_bins.append(mass_bins)
    all_mass_hist.append(mass_hist)
    return Pk, Pkz, k, kz, mass_bins, mass_hist


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

    # generate_comparison = check_gen_comparison(cfg)

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

        compare_paths = [get_source_path(
            wdir, 'quijotelike', 'fastpm',
            L, N, lhid=lhids
        )]

        source_path = [source_path]
        lhids = [lhids]
    else:
        lhids = range(cfg.nbody.lhid, cfg.nbody.lhid + n_lhid_seeds)
        source_path = [get_source_path(
            wdir, suite, sim,
            L, N, lhid=lhid
        ) for lhid in lhids]

        out_dir = get_source_path(
            odir, suite, sim,
            L, N, lhid=0, get_cfg_dir=True
        )

        # FIXME: integrate this into the hydra configuration
        compare_paths = [get_source_path(
            wdir, 'quijote', 'nbody',
            L, N, lhid=lhid
        ) for lhid in lhids]

    all_done = True

    # measure halo diagnostics
    done = plot_halo_sum(
        source_path, L, N, out_dir=out_dir, lhids=lhids, compare_paths=compare_paths)
    all_done &= done

    if all_done:
        logging.info('All diagnostics computed successfully')
    else:
        logging.error('Some diagnostics failed to compute')


if __name__ == "__main__":
    main()
