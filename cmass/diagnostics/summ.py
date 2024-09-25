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
                logging.info(f'Processing density field a={a}')
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
    pos = get_redshift_space_pos(pos=pos, vel=vel, z=z, h=h, axis=axis, L=L,)
    return BoxCatalogue(
        galaxies_pos=pos,
        redshift=z,
        boxsize=L,
        n_mesh=N,
    )

def get_binning(summary, L, N, threads, rsd=False,):
    ells = [0,] if not rsd else [0,2,4]
    if summary == 'Pk':
        return {
            'k_edges': np.linspace(0, 1., 31),
            'n_mesh': N,
            'los': 'z',
            'compensations': 'ngp',
            'ells': ells,
        }
    if summary == 'Bk':
        k_min = 1.05*2* np.pi / L
        n_mesh = 64
        k_max = 0.95 * np.pi * n_mesh / L 
        num_bins = 15
        return {
            'k_bins': np.logspace(np.log10(k_min), np.log10(k_max), num_bins),
            'n_mesh': n_mesh,
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
        }
    if summary == 'KNN':
        num_bins = 60
        return {
            'r_bins': np.logspace(-2, np.log10(30.), num_bins),
            'k': [1,3,7,11],
            'n_threads': threads,
        }
    else:
        raise NotImplementedError(f'{summary} not implemented')

def store_summary(catalog, group, summary_name, box_size, num_bins, num_threads, use_rsd=False):
    binning_config = get_binning(summary_name, box_size, num_bins, num_threads, rsd=use_rsd)
    
    logging.info(f'Computing Summary: {summary_name}, with binning:')
    logging.info(binning_config)
    
    summary_function = getattr(summarizer, summary_name)(**binning_config)
    summary_data = summary_function(catalog)
    summary_dataset = summary_function.to_dataset(summary_data)
    for coord_name, coord_value in summary_dataset.coords.items():
        dataset_key = f'{summary_name}_{coord_name}' if not use_rsd else f'z{summary_name}_{coord_name}'
        group.create_dataset(dataset_key, data=coord_value.values)
    summary_key = summary_name if not use_rsd else f'z{summary_name}'
    group.create_dataset(summary_key, data=summary_dataset.values)

def halo_summ(source_path, L, N, h, z, threads=16, from_scratch=True, summaries=['Pk','TwoPCF','KNN']): #['WST', 'Bk', 'DensitySplit']
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
                zi = 1/float(a) - 1
                logging.info(f'Processing halo catalog a={a}')
                # Load
                hpos = f[a]['pos'][...].astype(np.float32)
                hvel = f[a]['vel'][...].astype(np.float32)
                hmass = f[a]['mass'][...].astype(np.float32)
                # Ensure all halos inside box
                m = np.all((hpos >= 0) & (hpos < L), axis=1)
                hpos = hpos[m]
                hvel = hvel[m]
                hmass = hmass[m]
                # Get summaries in comoving space
                group = o.create_group(a)
                box_catalogue = get_box_catalogue(pos=hpos, z=z, L=L, N=N)
                for summ in summaries:
                    store_summary(box_catalogue, group, summ, L, N, threads, use_rsd=False,)

                # Get summaries in redshift space
                zbox_catalogue = get_box_catalogue_rsd(pos=hpos, vel=hvel, h=h, z=z, axis=2, L=L, N=N)
                for summ in summaries:
                    store_summary(zbox_catalogue, group, summ, L, N, threads, use_rsd=True,)
                    
                # measure halo mass function
                be = np.linspace(12.5, 16, 100)
                hist, _ = np.histogram(hmass, bins=be)

                # Save
                group.create_dataset('mass_bins', data=be)
                group.create_dataset('mass_hist', data=hist)
    return True


def gal_summ(source_path, hod_seed, L, N, h, z, threads=16,
             from_scratch=True,  summaries=['Pk','TwoPCF','KNN']): #WST
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
                zi = 1/float(a) - 1
                logging.info(f'Processing galaxy catalog a={a}')
                # Load
                gpos = f[a]['pos'][...].astype(np.float32)
                gvel = f[a]['vel'][...].astype(np.float32)

                # Ensure all galaxies inside box
                m = np.all((gpos >= 0) & (gpos < L), axis=1)
                gpos = gpos[m]
                gvel = gvel[m]

                # Get summaries in comoving space
                group = o.create_group(a)
                box_catalogue = get_box_catalogue(pos=gpos, z=z, L=L, N=N)
                for summ in summaries:
                    store_summary(box_catalogue, group, summ, L, N, threads, use_rsd=False,)

                # Get summaries in redshift space
                zbox_catalogue = get_box_catalogue_rsd(pos=gpos, vel=gvel, h=h, z=z, axis=2, L=L, N=N)
                for summ in summaries:
                    store_summary(zbox_catalogue, group, summ, L, N, threads, use_rsd=True,)

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
    N = (cfg.nbody.L//1000)*128  # 128 cells per 1000 Mpc/h  TODO: should this stay fixed?
    done = halo_summ(
        source_path, cfg.nbody.L, N, cfg.nbody.cosmo[2],
        cfg.nbody.zf, threads=threads, from_scratch=from_scratch)
    all_done &= done

    # measure gal diagnostics
    done = gal_summ(
        source_path, cfg.bias.hod.seed, cfg.nbody.L, N,
        cfg.nbody.cosmo[2], cfg.nbody.zf,
        threads=threads, from_scratch=from_scratch)
    all_done &= done

    if all_done:
        logging.info('All diagnostics computed successfully')
    else:
        logging.error('Some diagnostics failed to compute')


if __name__ == "__main__":
    main()
