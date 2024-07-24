"""
A script to compute basic summary statistics for all fields generated during
the simulation.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import h5py

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config
from .tools import MA, MAz, calcPk


def rho_summ(source_path, L, threads=16, from_scratch=True):
    # check if diagnostics already computed
    outpath = join(source_path, 'diag', 'rho.h5')
    if (not from_scratch) and os.path.isfile(outpath):
        logging.info('Rho diagnostics already computed')
        return True

    # check for file keys
    filename = join(source_path, 'nbody.h5')
    if not os.path.isfile(filename):
        logging.error(f'rho file {filename} not found')
        return False
    with h5py.File(filename, 'r') as f:
        alist = list(f.keys())

    logging.info(f'Saving rho diagnostics to {outpath}')
    os.makedirs(join(source_path, 'diag'), exist_ok=True)

    # compute diagnostics and save
    with h5py.File(filename, 'r') as f:
        with h5py.File(outpath, 'w') as o:
            for a in alist:
                rho = f[a]['rho'][...].astype(np.float32)
                k, Pk = calcPk(rho, L, threads=threads)
                group = o.create_group(a)
                group.create_dataset('k', data=k)
                group.create_dataset('Pk', data=Pk)
    return True


def halo_summ(source_path, L, N, h, z, threads=16, from_scratch=True):
    # check if diagnostics already computed
    outpath = join(source_path, 'diag', 'halos.h5')
    if (not from_scratch) and os.path.isfile(outpath):
        logging.info('Halo diagnostics already computed')
        return True

    # check for file keys
    filename = join(source_path, 'halos.h5')
    if not os.path.isfile(filename):
        logging.error(f'halo file not found: {filename}')
        return False
    with h5py.File(filename, 'r') as f:
        alist = list(f.keys())

    logging.info(f'Saving halo diagnostics to {outpath}')
    os.makedirs(join(source_path, 'diag'), exist_ok=True)

    # compute diagnostics and save
    with h5py.File(filename, 'r') as f:
        with h5py.File(outpath, 'w') as o:
            for a in alist:
                # Load
                hpos = f[a]['pos'][...]
                hvel = f[a]['vel'][...]
                hmass = f[a]['mass'][...]

                # measure halo Pk in comoving space
                delta = MA(hpos, L, N, MAS='NGP')
                k, Pk = calcPk(delta, L, MAS='NGP', threads=threads)

                # measure halo Pk in redshift space
                delta = MAz(hpos, hvel, L, N, h, z, MAS='NGP')
                kz, Pkz = calcPk(delta, L, MAS='NGP', threads=threads)

                # measure halo mass function
                be = np.linspace(13, 16, 100)
                hist, _ = np.histogram(hmass, bins=be)

                # Save
                group = o.create_group(a)
                group.create_dataset('Pk_k', data=k)
                group.create_dataset('Pk', data=Pk)
                group.create_dataset('zPk_k', data=kz)
                group.create_dataset('zPk', data=Pkz)
                group.create_dataset('mass_bins', data=be)
                group.create_dataset('mass_hist', data=hist)
    return True


def gal_summ(source_path, hod_seed, L, N, h, z, threads=16,
             from_scratch=True):
    # check if diagnostics already computed
    outpath = join(source_path, 'diag', 'galaxies', f'hod{hod_seed:03}.h5')
    if (not from_scratch) and os.path.isfile(outpath):
        logging.info('Gal diagnostics already computed')
        return True

    # check for file keys
    filename = join(source_path, 'galaxies', f'hod{hod_seed:03}.h5')
    if not os.path.isfile(filename):
        logging.error(f'gal file not found: {filename}')
        return False
    with h5py.File(filename, 'r') as f:
        alist = list(f.keys())

    logging.info(f'Saving gal diagnostics to {outpath}')
    os.makedirs(join(source_path, 'diag'), exist_ok=True)
    os.makedirs(join(source_path, 'diag', 'galaxies'), exist_ok=True)

    # compute diagnostics and save
    with h5py.File(filename, 'r') as f:
        with h5py.File(outpath, 'w') as o:
            for a in alist:
                # Load
                gpos = f[a]['pos'][...]
                gvel = f[a]['vel'][...]

                # measure gal Pk
                delta = MA(gpos, L, N, MAS='NGP')
                k, Pk = calcPk(delta, L, MAS='NGP', threads=threads)

                # measure gal zPk
                delta = MAz(gpos, gvel, L, N, h, z, MAS='NGP')
                kz, Pkz = calcPk(delta, L, MAS='NGP', threads=threads)

                # Save
                group = o.create_group(a)
                group.create_dataset('Pk_k', data=k)
                group.create_dataset('Pk', data=Pk)
                group.create_dataset('zPk_k', data=kz)
                group.create_dataset('zPk', data=Pkz)
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

    threads = cfg.diag.threads
    from_scratch = cfg.diag.from_scratch

    all_done = True

    # measure rho diagnostics
    done = rho_summ(
        source_path, cfg.nbody.L, threads=threads,
        from_scratch=from_scratch)
    all_done &= done

    # measure halo diagnostics
    done = halo_summ(
        source_path, cfg.nbody.L, cfg.nbody.N, cfg.nbody.cosmo[2],
        cfg.nbody.zf, threads=threads, from_scratch=from_scratch)
    all_done &= done

    # measure gal diagnostics
    done = gal_summ(
        source_path, cfg.bias.hod.seed, cfg.nbody.L, cfg.nbody.N,
        cfg.nbody.cosmo[2], cfg.nbody.zf,
        threads=threads, from_scratch=from_scratch)
    all_done &= done

    if all_done:
        logging.info('All diagnostics computed successfully')
    else:
        logging.error('Some diagnostics failed to compute')


if __name__ == "__main__":
    main()
