"""
A script to visualise basic summary statistics for all fields generated during
the simulation.
Can be also used to compare between different runs.
"""

import os
from typing import TypedDict

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


class HalosStdSummary(TypedDict):
    Pk_k: np.ndarray
    Pk: np.ndarray
    zPk_k: np.ndarray
    zPk: np.ndarray
    mass_bins: np.ndarray
    mass_hist: np.ndarray


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
        halo_summ = {}
        for key in HalosStdSummary.__required_keys__:
            halo_summ[key] = f[group][key][...]

        return halo_summ


def check_gen_comparison(config):
    same_wdir = config.meta.wdir == config.diag.compare.wdir
    same_nbody = config.nbody == config.diag.compare.nbody
    same_sim = config.sim == config.diag.compare.sim
    return same_wdir and same_nbody and same_sim


def plot_std_halo_summ(axs, halo_summ: HalosStdSummary, L, lhid, color):
    k, Pk = halo_summ['Pk_k'], halo_summ['Pk']
    axs[0].loglog(k, Pk[:, 0], color=color, label=f'lhid = {lhid}')
    axs[0].set_title(r'Comoving $P(k)$')
    axs[0].set_ylabel(r"$P(k)$ [$h^{-3}\,\mathrm{Mpc}^3$]")
    axs[0].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")

    kz, Pkz = halo_summ['zPk_k'], halo_summ['zPk']
    axs[1].loglog(kz, Pkz[:, 0], color=color, label=f'lhid = {lhid}')
    axs[1].set_title(r'Redshift space $P(k)$')
    axs[1].set_ylabel(r"$P(k)$ [$h^{-3}\,\mathrm{Mpc}^3$]")
    axs[1].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")

    mass_bins, mass_hist = halo_summ['mass_bins'], halo_summ['mass_hist']
    centered_bins = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    axs[2].loglog(10 ** centered_bins, mass_hist / L ** 3 / np.diff(mass_bins),
                  color=color, label=f'lhid = {lhid}')
    axs[2].set_title(r'Halo mass function')
    axs[2].set_ylabel("$n(M)\ [h^{3}\\mathrm{Mpc}^{-3}]$")
    axs[2].set_xlabel(r"$M [M_\odot]$")


def plot_halo_sum(source_path, L, N, out_dir, lhids, compare_paths):
    # TODO: Implement use of mpl style files
    # plt.style.use('/home/sding/PhD/codes/ltu-cmass/cmass/conf/diag/aa_one_column.mplstyle')

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
        halo_summs = []

        fig, axs = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
        cmap = mpl.colormaps['winter']

        # Take colors at regular intervals spanning the colormap.
        colors = cmap(np.linspace(0, 1, len(lhids)))
        cmap = (mpl.colors.ListedColormap(colors))

        for i, source_file in enumerate(source_files):
            halo_summ = extract_halo_diagnostics(source_file, a)
            halo_summs.append(halo_summ)
            plot_std_halo_summ(axs, halo_summ, L, lhids[i], colors[i])

        finalise_halo_summ_fig(L, N, a, axs, cmap, fig, lhids, outpath, prefix='halo_summ')

        if compare_paths is not None:
            compare_files = [join(path, 'diag', 'halos.h5') for path in compare_paths]

            # Plot individual ratios compared to another suite of simulations
            fig_comp, axs_comp = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
            fig_comp_ensemble, axs_comp_ensemble = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
            Pk_ratios = []
            Pkz_ratios = []
            hmf_ratios = []

            for i, compare_file in enumerate(compare_files):
                halo_summ_compare = extract_halo_diagnostics(compare_file, a)

                ratio_Pk_i = halo_summs[i]['Pk'][:, 0] / halo_summ_compare['Pk'][:, 0]
                axs_comp[0].semilogx(halo_summs[i]['Pk_k'], ratio_Pk_i, color=colors[i])

                ratio_Pkz_i = halo_summs[i]['zPk'][:, 0] / halo_summ_compare['zPk'][:, 0]
                axs_comp[1].semilogx(halo_summs[i]['zPk_k'], ratio_Pkz_i, color=colors[i])

                mass_bins = halo_summs[i]['mass_bins']
                centered_bins = 0.5 * (mass_bins[:-1] + mass_bins[1:])

                hmf = halo_summs[i]['mass_hist'] / L ** 3 / np.diff(mass_bins)
                hmf_comp = halo_summ_compare['mass_hist'] / L ** 3 / np.diff(mass_bins)
                ratio_hmf_i = hmf / hmf_comp
                axs_comp[2].semilogx(10 ** centered_bins, ratio_hmf_i, color=colors[i])

                Pk_ratios.append(ratio_Pk_i)
                Pkz_ratios.append(ratio_Pkz_i)
                hmf_ratios.append(ratio_hmf_i)

            axs_comp[0].hlines(1, 0, 2, colors='k', lw=0.4)
            axs_comp[0].set_title(r'Comoving $P(k)$')
            axs_comp[0].set_ylabel(r"$P(k) / P_{\mathrm{ref}}(k)$")
            axs_comp[0].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
            axs_comp[1].hlines(1, 0, 2, colors='k', lw=0.4)
            axs_comp[1].set_title(r'Redshift space $P(k)$')
            axs_comp[1].set_ylabel(r"$P(k) / P_{\mathrm{ref}}(k)$")
            axs_comp[1].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
            # axs_comp[2].hlines(1, 10**13, 10**16, colors='k', lw=0.4)

            finalise_halo_summ_fig(L, N, a, axs_comp, cmap, fig_comp, lhids, outpath, prefix='halo_summ_comparison')

            # Ensemble ratio plot
            mean_pk = np.mean(Pk_ratios, axis=0)
            std_pk = np.std(Pk_ratios, axis=0)
            k = halo_summs[0]['Pk_k']
            axs_comp_ensemble[0].semilogx(k, mean_pk, color='k')
            axs_comp_ensemble[0].fill_between(k, mean_pk - std_pk, mean_pk + std_pk, color='k', alpha=0.2)

            mean_pk_z = np.mean(Pkz_ratios, axis=0)
            std_pk_z = np.std(Pkz_ratios, axis=0)
            kz = halo_summs[0]['zPk_k']
            axs_comp_ensemble[1].semilogx(kz, mean_pk_z, color='k')
            axs_comp_ensemble[1].fill_between(kz, mean_pk_z - std_pk_z, mean_pk_z + std_pk_z, color='k', alpha=0.2)

            mean_hmf = np.mean(hmf_ratios, axis=0)
            std_hmf = np.std(hmf_ratios, axis=0)
            mass_bins = halo_summs[0]['mass_bins']
            centered_bins = 0.5 * (mass_bins[:-1] + mass_bins[1:])
            axs_comp_ensemble[2].semilogx(10 ** centered_bins, mean_hmf, color='k')
            axs_comp_ensemble[2].fill_between(10 ** centered_bins, mean_hmf - std_hmf, mean_hmf + std_hmf, color='k',
                                              alpha=0.2)

            axs_comp_ensemble[0].hlines(1, 0, 2, colors='k', lw=0.4)
            axs_comp_ensemble[0].set_title(r'Comoving $P(k)$')
            axs_comp_ensemble[0].set_ylabel(r"$P(k) / P_{\mathrm{ref}}(k)$")
            axs_comp_ensemble[0].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")
            axs_comp_ensemble[1].hlines(1, 0, 2, colors='k', lw=0.4)
            axs_comp_ensemble[1].set_title(r'Redshift space $P(k)$')
            axs_comp_ensemble[1].set_ylabel(r"$P(k) / P_{\mathrm{ref}}(k)$")
            axs_comp_ensemble[1].set_xlabel(r"$k$ [$h\,\mathrm{Mpc}^{-1}$]")

            finalise_halo_summ_fig(L, N, a, axs_comp_ensemble, cmap, fig_comp_ensemble, lhids, outpath,
                                   prefix='halo_summ_comparison_ensemble', no_cmap=True)

    return True


def finalise_halo_summ_fig(L, N, a, axs, cmap, fig, lhids, outpath, prefix, no_cmap=False):
    if len(lhids) == 1:
        file_name = f'{prefix}_group_{a}_lhid_{lhids[0]}.png'
    else:
        file_name = f'{prefix}_group_{a}_lhid_{lhids[0]}_to_{lhids[-1]}.png'

    out_file = join(outpath, file_name)
    k_nyquist = np.pi / (L / N)
    axs[0].set_xlim(right=k_nyquist * 0.7)
    axs[1].set_xlim(right=k_nyquist * 0.7)

    if len(lhids) > 1 or no_cmap:
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
    else:
        fig.suptitle(f'Latin-hypercube id: {lhids[0]}')
    logging.info(f'Saving figure as {out_file}')
    fig.savefig(out_file, format='png', dpi=300)


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
    make_comparison = cfg.diag.make_comparison

    # FIXME: Maybe solve this more elegantly
    if make_comparison:
        compare_suite = cfg.diag.compare_suite
        compare_sim = cfg.diag.compare_sim
    else:
        compare_paths = None

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
        if make_comparison:
            compare_paths = [get_source_path(
                wdir, compare_suite, compare_sim,
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

        if make_comparison:
            compare_paths = [get_source_path(
                wdir, compare_suite, compare_sim,
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
