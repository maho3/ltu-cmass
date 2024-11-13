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
from .tools import MA, calcPk
from ..survey.tools import sky_to_xyz
from ..bias.apply_hod import parse_hod
import astropy


def lc_summ(source_path, hod_seed, aug_seed, L, N, cosmo, is_North=True,
            threads=16, from_scratch=True, **header_kwargs):
    pfx = 'ngc' if is_North else 'sgc'

    # check if diagnostics already computed
    outpath = join(source_path, 'diag', f'{pfx}_lightcone',
                   f'hod{hod_seed:05}_aug{aug_seed:05}.h5')
    if (not from_scratch) and os.path.isfile(outpath):
        logging.info('Gal diagnostics already computed')
        return True

    # check for file keys
    filename = join(source_path, f'{pfx}_lightcone',
                    f'hod{hod_seed:05}_aug{aug_seed:05}.h5')
    if not os.path.isfile(filename):
        logging.error(f'gal file not found: {filename}')
        return False

    logging.info(f'Saving lc diagnostics to {outpath}')
    os.makedirs(join(source_path, 'diag'), exist_ok=True)
    os.makedirs(join(source_path, 'diag', f'{pfx}_lightcone'), exist_ok=True)

    # compute diagnostics and save
    with h5py.File(filename, 'r') as f:
        with h5py.File(outpath, 'w') as o:
            logging.info(
                f'Processing {pfx} lightcone catalog hod{hod_seed:05}_aug{aug_seed:05}')
            # Load
            ra = f['ra'][...]
            dec = f['dec'][...]
            z = f['z'][...]
            rdz = np.vstack([ra, dec, z]).T

            # convert to comoving
            xyz = sky_to_xyz(rdz, cosmo)

            # offset to center
            xyz += [1800, 1650, 150]

            # # noise positions
            xyz += np.random.randn(*xyz.shape)  # * 20 # 8/np.sqrt(3)

            # convert to float32
            xyz = xyz.astype(np.float32)

            # measure gal Pk
            delta = MA(xyz, L, N, MAS='NGP')
            delta[delta <= 0] = 1e-5  # avoid log(0)
            k, Pk = calcPk(delta, L, MAS='NGP', threads=threads)

            # Save
            o.create_dataset('Pk_k', data=k)
            o.create_dataset('Pk', data=Pk)

            for k, v in header_kwargs.items():
                o.attrs[k] = v
    return True


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'multisnapshot', 'nbody', 'bias', 'diag'])
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    is_North = cfg.survey.is_North

    threads = cfg.diag.threads
    from_scratch = cfg.diag.from_scratch

    all_done = True

    # set grid
    L = 3200  # Mpc/h
    N = 384
    cosmo = astropy.cosmology.Planck15
    threads = os.cpu_count()

    # measure gal diagnostics
    done = lc_summ(
        source_path, cfg.bias.hod.seed, cfg.survey.aug_seed,
        L, N, cosmo, is_North=is_North,
        threads=threads, from_scratch=from_scratch,
        cosmology=cfg.nbody.cosmo, **cfg.bias.hod.theta)
    all_done &= done

    if all_done:
        logging.info('All diagnostics computed successfully')
    else:
        logging.error('Some diagnostics failed to compute')


if __name__ == "__main__":
    main()
