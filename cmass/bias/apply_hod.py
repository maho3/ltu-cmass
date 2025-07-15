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
    - TODO: Allow cosmology-dependent HOD priors
"""

import numpy as np
import logging
import os
from os.path import join
import hydra
import h5py
from omegaconf import DictConfig, OmegaConf
from .tools.hod import (
    build_halo_catalog, build_HOD_model, parse_hod)
from ..utils import (
    get_source_path, timing_decorator, cosmo_to_astropy, save_cfg)
from ..nbody.tools import parse_nbody_config


@timing_decorator
def populate_hod(
    hpos, hvel, hmass,
    cosmo, L, redshift,
    model, theta,
    hmeta=None,
    seed=0, mdef='vir',
    zpivot=None,
    assem_bias=False,
    vel_assem_bias=False,
    custom_prior=None,
):
    cosmo = cosmo_to_astropy(cosmo)

    if (hmeta is not None) and ('concentration' in hmeta):
        logging.info('Using saved halo concentration...')
        hconc = hmeta['concentration']
    else:
        logging.info('Using halo-concentration relation...')
        hconc = None

    if (hmeta is not None) and ('redshift' in hmeta):
        hredshift = hmeta['redshift']
    else:
        hredshift = redshift

    BoxSize = L*np.ones(3)
    catalog = build_halo_catalog(
        hpos, hvel, 10**hmass, redshift, BoxSize, cosmo,
        mdef=mdef, conc=hconc, halo_redshift=hredshift
    )

    hod = build_HOD_model(
        cosmo,
        model=model,
        theta=theta,
        zf=redshift,
        mdef=mdef,
        zpivot=zpivot,
        assem_bias=assem_bias,
        vel_assem_bias=vel_assem_bias,
        custom_prior=custom_prior,
    )
    hod.populate_mock(catalog, seed=seed, halo_mass_column_key=f'halo_m{mdef}')
    galcat = hod.mock.galaxy_table.as_array()

    return galcat


def run_snapshot(hpos, hvel, hmass, a, cfg, hmeta=None):
    # Populate HOD
    logging.info('Populating HOD...')
    hod = populate_hod(
        hpos, hvel, hmass,
        cfg.nbody.cosmo, cfg.nbody.L, (1/a)-1,
        cfg.bias.hod.model, cfg.bias.hod.theta,
        seed=cfg.bias.hod.seed,
        hmeta=hmeta if cfg.bias.hod.use_conc else None,
        mdef=cfg.bias.hod.mdef,
        zpivot=getattr(cfg.bias.hod, "zpivot", None),
        assem_bias=getattr(cfg.bias.hod, "assem_bias", False),
        vel_assem_bias=getattr(cfg.bias.hod, "vel_assem_bias", False),
        custom_prior=getattr(cfg.bias.hod, "custom_prior", None),
    )

    # Organize outputs
    gpos = np.array(
        [hod['x'], hod['y'], hod['z']]).T  # comoving positions [Mpc/h]
    gvel = np.array(
        [hod['vx'], hod['vy'], hod['vz']]).T  # physical velocities [km/s]
    gmeta = {'gal_type': hod['gal_type'], 'hostid': hod['halo_id']}
    return gpos, gvel, gmeta


def load_snapshot(source_path, a):
    with h5py.File(join(source_path, 'halos.h5'), 'r') as f:
        group = f[f'{a:.6f}']
        hpos = group['pos'][...]    # comoving positions [Mpc/h]
        hvel = group['vel'][...]    # physical velocities [km/s]
        hmass = group['mass'][...]  # log10(Mass) [Msun/h]

        hmeta = {}
        for key in group.keys():
            if key not in ['pos', 'vel', 'mass']:
                hmeta[key] = group[key][...]
    return hpos, hvel, hmass, hmeta


def delete_outputs(outpath):
    if os.path.isfile(outpath):
        os.remove(outpath)


def save_parameters(outpath, **params):
    with h5py.File(outpath, 'w') as f:
        for key, value in params.items():
            f.attrs[key] = value


def save_snapshot(outpath, a, gpos, gvel, **meta):
    with h5py.File(outpath, 'a') as f:
        group = f.create_group(f'{a:.6f}')
        group.create_dataset('pos', data=gpos)
        group.create_dataset('vel', data=gvel)
        for key, value in meta.items():
            group.create_dataset(key, data=value)


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'multisnapshot', 'nbody', 'bias'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    # Save with original hod_seed (parse_hod modifies it to lhid*1e6 + hod_seed)
    hod_seed = int(cfg.bias.hod.seed - cfg.nbody.lhid * 1e6)

    # Setup save directory
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    save_path = join(source_path, 'galaxies')
    os.makedirs(save_path, exist_ok=True)
    save_file = join(save_path, f'hod{hod_seed:05}.h5')
    logging.info(f'Saving to {save_file}...')

    # Delete existing outputs
    delete_outputs(save_file)

    # Save parameters
    save_parameters(save_file, **cfg.bias.hod.theta)

    # Run each snapshot
    for i, a in enumerate(cfg.nbody.asave):
        logging.info(f'Running snapshot {i} at a={a:.6f}...')

        # Load snapshot
        hpos, hvel, hmass, hmeta = load_snapshot(source_path, a)

        # Populate HOD
        gpos, gvel, gmeta = run_snapshot(hpos, hvel, hmass, a, cfg, hmeta=hmeta)

        # Save snapshot
        save_snapshot(save_file, a, gpos, gvel, **gmeta)

    save_cfg(source_path, cfg, field='bias')
    logging.info('Done!')


if __name__ == '__main__':
    main()
