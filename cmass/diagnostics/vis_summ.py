"""
A script to visualise basic summary statistics for all fields generated during
the simulation.
Can be also used to compare between different runs.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config


def plot_halo_sum(source_path, L, N, h, z, from_scratch=True, out_dir=None):
    if out_dir is None:
        out_dir = source_path

    # check if diagnostics is computed
    source_file = join(source_path, 'diag', 'halos.h5')
    if os.path.isfile(source_file):
        logging.info('Halo diagnostics already computed. Proceeding with visualisation.')
    else:
        logging.error(f'{source_file} with halo diagnostics not found.')
        return False

    # check for file keys
    with h5py.File(source_file, 'r') as f:
        alist = list(f.keys())

    outpath = join(out_dir, 'diag')
    os.makedirs(outpath, exist_ok=True)
    logging.info(f'Saving halo diagnostics to {outpath}')

    # compute diagnostics and save
    with h5py.File(source_file, 'r') as f:
        for a in alist:
            out_file = join(outpath, f'halo_summ_group{a}.svg')
            # Load
            k = f[a]['Pk_k'][...]
            Pk = f[a]['Pk'][...]
            kz = f[a]['zPk_k'][...]
            Pkz = f[a]['zPk'][...]
            mass_bins = f[a]['mass_bins'][...]
            mass_hist = f[a]['mass_hist'][...]

            fig, axs = plt.subplots(1, 3)
            axs[0].loglog(k, Pk)
            axs[1].loglog(kz, Pkz)
            centered_bins = 0.5 * (mass_bins[:-1] + mass_bins[1:])
            axs[2].loglog(centered_bins, mass_hist/L**3)

            fig.savefig(out_file, format='svg')
    return True

@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'sim', 'nbody', 'bias', 'diag'])
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    cfg = parse_nbody_config(cfg)
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )

    out_dir = get_source_path(
        cfg.meta.odir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid,
        mkdir=True
    )

    from_scratch = cfg.diag.from_scratch

    all_done = True

    # measure halo diagnostics
    done = plot_halo_sum(
        source_path, cfg.nbody.L, cfg.nbody.N, cfg.nbody.cosmo[2],
        cfg.nbody.zf, from_scratch=from_scratch, out_dir=out_dir)
    all_done &= done

    if all_done:
        logging.info('All diagnostics computed successfully')
    else:
        logging.error('Some diagnostics failed to compute')


if __name__ == "__main__":
    main()
