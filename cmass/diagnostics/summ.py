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

from summarizer.data import BoxCatalogue
import summarizer

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config
from .tools import MA, MAz, calcPk, get_redshift_space_pos


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

def get_box_catalogue(pos, z, L, N):
    return BoxCatalogue(
        galaxies_pos=pos,
        redshift=z,
        boxsize=L,
        n_mesh=N,
    )

def get_box_catalogue_rsd(pos, vel, z, L, h, axis, N):
    pos = get_redshift_space_pos(pos=pos, vel=vel, z=z, h=h, axis=axis, L=L, N=N,)
    return BoxCatalogue(
        galaxies_pos=pos,
        redshift=z,
        boxsize=L,
        n_mesh=N,
    )

def get_binning(summary, L, N, threads, rsd=False,):
    #TODO: make sure using gpu if available
    ells = [0,] if not rsd else [0,2,4]
    if summary == 'Pk':
        return {
            'n_mesh': N,
            'los': 'z',
            'compensation': 'nsc',
            'ells': ells,
            'n_threads': threads,
        }
    if summary == 'Bk':
        k_min = 2 * np.pi / L
        k_max = np.pi * N / L 
        num_bins = 30
        return {
            'k_bins': np.logspace(np.log10(k_min), np.log10(k_max), num_bins),
            'n_mesh': N,
            'lmax': 2,
            'ells': ells,
        }
    if summary == 'TwoPCF':
        num_bins = 60
        return {
            'r_bins': np.logspace(-2, np.log10(150.), num_bins),
            'mu_bins': np.linspace(-1.,1.,201),
            'ells': ells,
            'n_threads': threads,
            'los': 'z',
        }
    if summary == 'WST':
        num_bins = 60
        return {
            'J_3d': 3,
            'L_3d': 3,
            'integral_powers': [0.8,],
            'sigma': 0.8,
            'n_mesh': N,
        }
    if summary == 'DensitySplit':
        num_bins = 60
        return {
            'r_bins': np.logspace(-1, np.log10(150.), num_bins),
            'mu_bins': np.linspace(-1.,1.,201),
            'n_quantiles': 5,
            'smoothing_radius': 10.0,
            'ells': ells,
            'n_threads': threads,
            'los': 'z',
            'n_mesh': N,
        }
    if summary == 'kNN':
        num_bins = 60
        return {
            'r_bins': np.logspace(-2, np.log10(30.), num_bins),
            'k': [1,3,7,11],
            'n_threads': threads,
        }
    else:
        raise NotImplementedError(f'{summary} not implemented')



def halo_summ(source_path, L, N, h, z, threads=16, from_scratch=True, summaries=['Pk', 'Bk', 'TwoPCF', 'WST', 'DensitySplit', 'KNN']):
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
                # Get summaries in comoving space
                group = o.create_group(a)
                box_catalogue = get_box_catalogue(pos=hpos, z=z, L=L, N=N)
                for summ in summaries:
                    binning = get_binning(summary, L, N, threads, rsd=False,)
                    summary = getattr(summarizer, summ)(**binning)(box_catalogue)
                    #TODO: check this is storing both coordinates and values
                    for key, value in summary.coords.items():
                        group.create_dataset(f'{summ}_{key}', data=value)
                    group.create_dataset(f'{summ}', data=summary.values)

                # Get summaries in redshift space
                zbox_catalogue = get_box_catalogue(pos=hpos, vel=hvel, h=h, z=z, axis=2, L=L, N=N)
                for summ in summaries:
                    binning = get_binning(summary, L, N, threads, rsd=True,)
                    summary = getattr(summarizer, summ)(**binning)(zbox_catalogue)
                    for key, value in summary.coords.items():
                        group.create_dataset(f'z{summ}_{key}', data=value)
                    group.create_dataset(f'z{summ}', data=summary.values)

                # measure halo mass function
                be = np.linspace(13, 16, 100)
                hist, _ = np.histogram(hmass, bins=be)

                # Save
                group.create_dataset('mass_bins', data=be)
                group.create_dataset('mass_hist', data=hist)
    return True


def gal_summ(source_path, hod_seed, L, N, h, z, threads=16,
             from_scratch=True):
    # check if diagnostics already computed
    outpath = join(source_path, 'diag', 'galaxies', f'hod{hod_seed:05}.h5')
    if (not from_scratch) and os.path.isfile(outpath):
        logging.info('Gal diagnostics already computed')
        return True

    # check for file keys
    filename = join(source_path, 'galaxies', f'hod{hod_seed:05}.h5')
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
