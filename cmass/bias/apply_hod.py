"""
Sample an HOD realization from the halo catalog.
HOD model and parameters defined in the bias configuration file.

Input:
    - halos.h5
        - pos: halo positions
        - vel: halo velocities
        - mass: halo masses

Output:
    - galaxies/hod{hod_seed}.h5
        - pos: halo positions
        - vel: halo velocities
        - gal_type: galaxy type (central or satellite)
        - hostid: host halo ID

NOTE:
    - TODO: Implement Zheng+ex 10-parameter model
    - TODO: Allow cosmology-dependent HOD priors
"""

import numpy as np
import logging
import os
from os.path import join
import hydra
import h5py
from omegaconf import DictConfig, OmegaConf, open_dict
from .tools.hod import (
    build_halo_catalog, build_HOD_model, parse_hod)
from ..utils import (
    get_source_path, timing_decorator, load_params, cosmo_to_astropy, save_cfg)
from ..nbody.tools import parse_nbody_config


@ timing_decorator
def populate_hod(
    hpos, hvel, hmass,
    cosmo, cfg, seed=0, mdef='vir'
):
    cosmo = cosmo_to_astropy(cosmo)

    BoxSize = cfg.nbody.L*np.ones(3)
    catalog = build_halo_catalog(
        hpos, hvel, 10**hmass, cfg.nbody.zf, BoxSize, cosmo,
        mdef=mdef
    )

    hod = build_HOD_model(cosmo, cfg, mdef=mdef)
    hod.populate_mock(catalog, seed=seed)
    galcat = hod.mock.galaxy_table.as_array()

    return galcat


def run_snapshot(pos, vel, mass, cfg):
    # Populate HOD
    logging.info('Populating HOD...')
    hod = populate_hod(
        pos, vel, mass,
        cfg.nbody.cosmo, cfg,
        seed=cfg.bias.hod.seed
    )

    # Organize outputs
    gpos = np.array([hod['x'], hod['y'], hod['z']]).T
    gvel = np.array([hod['vx'], hod['vy'], hod['vz']]).T
    meta = {'gal_type': hod['gal_type'], 'hostid': hod['halo_id']}
    return gpos, gvel, meta


def load_snapshot(source_path, a):
    with h5py.File(join(source_path, 'halos.h5'), 'r') as f:
        group = f[f'{a:.6f}']
        hpos = group['pos'][...]
        hvel = group['vel'][...]
        hmass = group['mass'][...]
    return hpos, hvel, hmass


def delete_outputs(outpath):
    if os.path.isfile(outpath):
        os.remove(outpath)


def save_snapshot(outpath, a, gpos, gvel, **meta):
    with h5py.File(outpath, 'a') as f:
        group = f.create_group(f'{a:.6f}')
        group.create_dataset('pos', data=gpos)
        group.create_dataset('vel', data=gvel)
        for key, value in meta.items():
            group.create_dataset(key, data=value)


@ timing_decorator
@ hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'multisnapshot', 'nbody', 'bias'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    # Setup save directory
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    save_path = join(source_path, 'galaxies')
    os.makedirs(save_path, exist_ok=True)
    save_file = join(save_path, f'hod{cfg.bias.hod.seed:05}.h5')
    logging.info(f'Saving to {save_file}...')

    # Delete existing outputs
    delete_outputs(save_file)

    # Run each snapshot
    for i, a in enumerate(cfg.nbody.asave):
        logging.info(f'Running snapshot {i} at a={a:.6f}...')

        # Load snapshot
        hpos, hvel, hmass = load_snapshot(source_path, a)

        # Populate HOD
        gpos, gvel, meta = run_snapshot(hpos, hvel, hmass, cfg)

        # Save snapshot
        save_snapshot(save_file, a, gpos, gvel, **meta)

    save_cfg(source_path, cfg, field='bias')
    logging.info('Done!')


if __name__ == '__main__':
    main()
