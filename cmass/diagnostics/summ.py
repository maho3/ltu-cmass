"""
A script to compute basic summary statistics for all fields generated during
the simulation.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig
import h5py

from ..utils import get_source_path, timing_decorator
from ..nbody.tools import parse_nbody_config
from .tools import MA, MAz, calcPk


def rho_summ(source_path, L, threads=16):
    outpath = join(source_path, 'diag', 'rho.h5')
    if os.path.isfile(outpath):
        logging.info('Rho diagnostics already computed')
        return True

    filename = join(source_path, 'rho.npy')
    if not os.path.isfile(filename):
        logging.error(f'rho file {filename} not found')
        return False
    rho = np.load(filename)
    k, Pk = calcPk(rho, L, threads=threads)

    logging.info(f'Saving rho diagnostics to {outpath}')
    os.makedirs(join(source_path, 'diag'), exist_ok=True)
    with h5py.File(outpath, 'w') as f:
        f.create_dataset('k', data=k)
        f.create_dataset('Pk', data=Pk)
    return True


def halo_summ(source_path, L, N, h, z, threads=16):
    outpath = join(source_path, 'diag', 'halo.h5')
    if os.path.isfile(outpath):
        logging.info('Halo diagnostics already computed')
        return True

    if not os.path.isfile(join(source_path, 'halo_mass.npy')):
        logging.error('halo files not found')
        return False
    hpos = np.load(join(source_path, 'halo_pos.npy'))
    hvel = np.load(join(source_path, 'halo_vel.npy'))
    hmass = np.load(join(source_path, 'halo_mass.npy'))

    # measure halo Pk
    delta = MA(hpos, L, N, MAS='NGP')
    k, Pk = calcPk(delta, L, MAS='NGP', threads=threads)

    # measure halo zPk
    delta = MAz(hpos, hvel, L, N, h, z, MAS='NGP')
    kz, Pkz = calcPk(delta, L, MAS='NGP', threads=threads)

    # measure halo mass function
    be = np.linspace(13, 16, 100)
    hist, _ = np.histogram(hmass, bins=be)

    logging.info(f'Saving halo diagnostics to {outpath}')
    os.makedirs(join(source_path, 'diag'), exist_ok=True)
    with h5py.File(outpath, 'w') as f:
        group = f.create_group('Pk')
        group.create_dataset('k', data=k)
        group.create_dataset('Pk', data=Pk)
        group = f.create_group('zPk')
        group.create_dataset('k', data=kz)
        group.create_dataset('Pk', data=Pkz)
        group = f.create_group('mass')
        group.create_dataset('bins', data=be)
        group.create_dataset('hist', data=hist)
    return True


def gal_summ(source_path, hod_seed, L, N, h, z, threads=16):
    outpath = join(source_path, 'diag', 'gal.h5')
    if os.path.isfile(outpath):
        logging.info('Gal diagnostics already computed')
        return True

    if not os.path.isfile(join(source_path, 'hod', f'hod{hod_seed}_pos.npy')):
        logging.error('gal files not found')
        return False
    gpos = np.load(join(source_path, 'hod', f'hod{hod_seed}_pos.npy'))
    gvel = np.load(join(source_path, 'hod', f'hod{hod_seed}_vel.npy'))

    # measure gal Pk
    delta = MA(gpos, L, N, MAS='NGP')
    k, Pk = calcPk(delta, L, MAS='NGP', threads=threads)

    # measure gal zPk
    delta = MAz(gpos, gvel, L, N, h, z, MAS='NGP')
    kz, Pkz = calcPk(delta, L, MAS='NGP', threads=threads)

    logging.info(f'Saving gal diagnostics to {outpath}')
    os.makedirs(join(source_path, 'diag'), exist_ok=True)
    with h5py.File(outpath, 'w') as f:
        group = f.create_group('Pk')
        group.create_dataset('k', data=k)
        group.create_dataset('Pk', data=Pk)
        group = f.create_group('zPk')
        group.create_dataset('k', data=kz)
        group.create_dataset('Pk', data=Pkz)
    return True


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs

    # logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    cfg = parse_nbody_config(cfg)
    source_path = get_source_path(cfg, cfg.sim)

    threads = 16

    # measure rho diagnostics
    done = rho_summ(source_path, cfg.nbody.L, threads=threads)
    if not done:
        return

    # measure halo diagnostics
    done = halo_summ(
        source_path, cfg.nbody.L, cfg.nbody.N, cfg.nbody.cosmo[2],
        cfg.nbody.zf, threads=threads)
    if not done:
        return

    # measure gal diagnostics
    done = gal_summ(
        source_path, cfg.bias.hod.seed, cfg.nbody.L, cfg.nbody.N, cfg.nbody.cosmo[2],
        cfg.nbody.zf, threads=threads)
    if not done:
        return

    logging.info('All diagnostics computed successfully')


if __name__ == "__main__":
    main()
