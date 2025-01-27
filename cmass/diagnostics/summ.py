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
from astropy.cosmology import Planck18

from ..utils import get_source_path, timing_decorator, cosmo_to_astropy
from ..nbody.tools import parse_nbody_config
from ..bias.apply_hod import parse_hod
from .tools import MA, MAz, get_box_catalogue, get_box_catalogue_rsd
from .tools import calcPk, calcBk
from ..survey.tools import sky_to_xyz


def get_binning(summary, L, N, threads, rsd=False):
    ells = [0,] if not rsd else [0, 2, 4]
    if summary == 'Pk':
        return {
            'k_edges': np.linspace(0, 1., 31),
            'n_mesh': N,
            'los': 'z',
            'compensations': 'ngp',
            'ells': ells,
        }
    if summary == 'Bk':
        k_min = 1.05*2 * np.pi / L
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
            'mu_bins': np.linspace(-1., 1., 201),
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
            'mu_bins': np.linspace(-1., 1., 201),
            'n_quantiles': 5,
            'smoothing_radius': 10.0,
            'ells': ells,
            'n_threads': threads,
        }
    if summary == 'KNN':
        num_bins = 60
        return {
            'r_bins': np.logspace(-2, np.log10(30.), num_bins),
            'k': [1, 3, 7, 11],
            'n_threads': threads,
        }
    else:
        raise NotImplementedError(f'{summary} not implemented')


def store_summary(
    catalog, group, summary_name,
    box_size, num_bins, num_threads, use_rsd=False
):
    # get summary binning
    binning_config = get_binning(
        summary_name, box_size, num_bins, num_threads, rsd=use_rsd)

    logging.info(f'Computing Summary: {summary_name}')

    # compute summary
    import summarizer  # only import if needed. TODO: get working
    summary_function = getattr(summarizer, summary_name)(**binning_config)
    summary_data = summary_function(catalog)

    # store summary
    summary_dataset = summary_function.to_dataset(summary_data)
    for coord_name, coord_value in summary_dataset.coords.items():
        dataset_key = f"{'z' if use_rsd else ''}{summary_name}_{coord_name}"
        group.create_dataset(dataset_key, data=coord_value.values)
    summary_key = summary_name if not use_rsd else f'z{summary_name}'
    group.create_dataset(summary_key, data=summary_dataset.values)


def run_pylians(
    field, group, summaries,
    box_size, axis, num_threads, use_rsd,
    MAS='CIC'
):
    # Only for power spectrum
    accept_summaries = ['Pk', 'Bk']

    for summary_name in summaries:
        pfx = 'z' if use_rsd else ''
        if summary_name == 'Pk':
            k, Pk = calcPk(field, box_size, axis=axis,
                           MAS=MAS, threads=num_threads)
            group.create_dataset(pfx+'Pk_k3D', data=k)
            group.create_dataset(pfx+'Pk', data=Pk)
        elif summary_name == 'Bk':
            k123, Bk, Qk, k, Pk = calcBk(
                field, box_size, axis=axis,
                MAS=MAS, threads=num_threads)
            group.create_dataset(pfx+'Bk_k123', data=k123)
            group.create_dataset(pfx+'Bk', data=Bk)
            group.create_dataset(pfx+'Qk', data=Qk)
            group.create_dataset(pfx+'bPk_k3D', data=k123)
            group.create_dataset(pfx+'bPk', data=Pk)
        elif summary_name not in accept_summaries:
            logging.error(f'{summary_name} not yet implemented in Pylians')
            continue


def run_summarizer(
    pos, vel, h, redshift, box_size, grid_size,
    group, summaries, threads
):
    # For two-point correlation, bispectrum, wavelets, density split, and
    # k-nearest neighbors

    # Get summaries in comoving space
    box_catalogue = get_box_catalogue(
        pos=pos, z=redshift, L=box_size, N=grid_size)
    for summ in summaries:
        store_summary(
            box_catalogue, group, summ,
            box_size, grid_size, threads, use_rsd=False)

    # Get summaries in redshift space
    zbox_catalogue = get_box_catalogue_rsd(
        pos=pos, vel=vel, h=h, z=redshift, axis=2, L=box_size, N=grid_size)
    for summ in summaries:
        store_summary(
            zbox_catalogue, group, summ,
            box_size, grid_size, threads, use_rsd=True)


def save_configuration(file, config, save_HOD=True):
    file.attrs['config'] = OmegaConf.to_yaml(config)
    file.attrs['cosmo_names'] = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma8']
    file.attrs['cosmo_params'] = config.nbody.cosmo

    if save_HOD:
        file.attrs['HOD_model'] = config.bias.hod.model
        file.attrs['HOD_seed'] = config.bias.hod.seed

        keys = sorted(list(config.bias.hod.theta.keys()))
        file.attrs['HOD_names'] = keys
        file.attrs['HOD_params'] = [config.bias.hod.theta[k] for k in keys]


def summarize_rho(
    source_path, L,
    threads=16, from_scratch=True,
    config=None
):
    # check if diagnostics already computed
    outpath = join(source_path, 'diag', 'rho.h5')
    if (not from_scratch) and os.path.isfile(outpath):
        logging.info('rho diagnostics already computed')
        return True
    logging.info(f'Computing diagnostics to save to: {outpath}')

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
            if config is not None:
                save_configuration(o, config, save_HOD=False)
            for a in alist:
                logging.info(f'Processing density field a={a}')
                rho = f[a]['rho'][...].astype(np.float32)
                group = o.create_group(a)
                if 'Pk' in config.diag.summaries:
                    run_pylians(
                        rho, group, ['Pk'], L, axis=0, MAS='CIC',
                        num_threads=threads, use_rsd=False
                    )
                if 'Bk' in config.diag.summaries:
                    run_pylians(
                        rho, group, ['Bk'], L, axis=0, MAS='CIC',
                        num_threads=threads, use_rsd=False
                    )
    return True


def summarize_tracer(
    source_path, L, N, cosmo,
    density=None, proxy=None,
    threads=16, from_scratch=True,
    type='halo', hod_seed=None,
    summaries=['Pk'],
    config=None
):
    postfix = 'halos.h5' if type == 'halo' else f'galaxies/hod{hod_seed:05}.h5'

    # check if diagnostics already computed
    outpath = join(source_path, 'diag', postfix)
    if (not from_scratch) and os.path.isfile(outpath):
        logging.info(f'{type} diagnostics already computed')
        return True
    logging.info(f'Computing diagnostics to save to: {outpath}')

    # check for file keys
    filename = join(source_path, postfix)
    if not os.path.isfile(filename):
        logging.error(f'File not found: {filename}')
        return False
    with h5py.File(filename, 'r') as f:
        alist = list(f.keys())

    logging.info(f'Computing diagnostics to save to: {outpath}')
    os.makedirs(join(source_path, 'diag'), exist_ok=True)
    if type == 'galaxy':
        os.makedirs(join(source_path, 'diag', 'galaxies'), exist_ok=True)

    # compute overdensity cut
    Ncut = None if density is None else int(density * L**3)

    # compute diagnostics and save
    with h5py.File(filename, 'r') as f:
        with h5py.File(outpath, 'w') as o:
            if config is not None:
                save_configuration(o, config, save_HOD=(type == 'galaxy'))
            for a in alist:
                z = 1/float(a) - 1
                logging.info(f'Processing {type} catalog a={a}')

                # Load
                pos = f[a]['pos'][...].astype(np.float32)
                vel = f[a]['vel'][...].astype(np.float32)
                pos %= L  # Ensure all tracers inside box

                # Create output group
                group = o.create_group(a)

                # Mask out low mass tracers (to match number density)
                mass = None
                if density is not None:
                    if proxy is None:
                        logging.warning(
                            'Proxy is set to None. Not rank-ordering.')
                        mass = np.arange(len(pos))
                        np.random.shuffle(mass)
                    else:
                        if proxy not in f[a].keys():
                            logging.error(
                                f'{proxy} not found in {type} file at a={a}')
                        if len(pos) <= Ncut:
                            logging.warning(f'Not enough {type} tracers in {a}')
                        logging.info(
                            f'Cutting top {Ncut} out of {len(pos)} {type} '
                            'tracers to match number density')
                        mass = f[a][proxy][...].astype(np.float32)
                    mask = np.argsort(mass)[-Ncut:]  # Keep top Ncut tracers
                    pos = pos[mask]
                    vel = vel[mask]
                    mass = mass[mask]

                    group.attrs['density'] = float(density)
                else:
                    group.attrs['density'] = np.nan

                # Noise out positions (we do not probe less than Lnoise)
                Lnoise = (1000/128)/np.sqrt(3)  # Set by CHARM resolution
                pos += np.random.randn(*pos.shape) * Lnoise

                # Compute P(k)
                if 'Pk' in summaries:
                    # real space
                    MAS = 'NGP'
                    field = MA(pos, L, N, MAS=MAS).astype(np.float32)
                    run_pylians(
                        field, group, ['Pk'], L, axis=0, MAS=MAS,
                        num_threads=threads, use_rsd=False
                    )

                    # redshift space
                    field = MAz(pos, vel, L, N, cosmo, z, MAS=MAS,
                                axis=0).astype(np.float32)
                    run_pylians(
                        field, group, ['Pk'], L, axis=0, MAS=MAS,
                        num_threads=threads, use_rsd=True
                    )
                if 'Bk' in summaries:
                    # real space
                    MAS = 'TSC'
                    field = MA(pos, L, N, MAS=MAS).astype(np.float32)
                    run_pylians(
                        field, group, ['Bk'], L, axis=0, MAS=MAS,
                        num_threads=threads, use_rsd=False
                    )

                    # redshift space
                    field = MAz(pos, vel, L, N, cosmo, z, MAS=MAS,
                                axis=0).astype(np.float32)
                    run_pylians(
                        field, group, ['Bk'], L, axis=0, MAS=MAS,
                        num_threads=threads, use_rsd=True
                    )

                # Compute other summaries
                others = [s for s in summaries if (
                    ('Pk' not in s) and ('Bk' not in s))]
                if len(others) > 0:
                    run_summarizer(
                        pos, vel, cosmo.h, z, L, N,
                        group, others,
                        threads
                    )

                if mass is not None:
                    # measure halo mass function
                    be = np.linspace(12.5, 16, 100)
                    hist, _ = np.histogram(mass, bins=be)
                    group.create_dataset('mass_bins', data=be)
                    group.create_dataset('mass_hist', data=hist)
    return True


def summarize_lightcone(
    source_path, L, N, cosmo,
    cap='ngc',
    threads=16, from_scratch=True,
    hod_seed=None, aug_seed=None,
    summaries=['Pk'],
    config=None
):
    postfix = f'{cap}_lightcone/hod{hod_seed:05}_aug{aug_seed:05}.h5'

    # check if diagnostics already computed
    outpath = join(source_path, 'diag', postfix)
    if (not from_scratch) and os.path.isfile(outpath):
        logging.info(f'{cap}_lightcone diagnostics already computed')
        return True
    logging.info(f'Computing diagnostics to save to: {outpath}')

    # check for file keys
    filename = join(source_path, postfix)
    if not os.path.isfile(filename):
        logging.error(f'File not found: {filename}')
        return False

    os.makedirs(join(source_path, 'diag'), exist_ok=True)
    os.makedirs(join(source_path, 'diag', f'{cap}_lightcone'), exist_ok=True)

    # compute diagnostics and save
    with h5py.File(filename, 'r') as f:
        with h5py.File(outpath, 'w') as o:
            if config is not None:
                save_configuration(o, config, save_HOD=True)

            # Load
            ra = f['ra'][...]
            dec = f['dec'][...]
            z = f['z'][...]
            rdz = np.vstack([ra, dec, z]).T

            # convert to comoving
            pos = sky_to_xyz(rdz, cosmo)

            # Noise out positions (we do not probe less than Lnoise)
            Lnoise = (1000/128)/np.sqrt(3)  # Set by CHARM resolution
            pos += np.random.randn(*pos.shape) * Lnoise

            # convert to float32
            pos = pos.astype(np.float32)

            if cap == 'ngc':
                # offset to center (min is about -1870, -1750, -120)
                pos += [2000, 1800, 250]
                # set length scale of grid (range is about 1750, 3350, 1900)
                L = 3500
            elif cap == 'sgc':
                # offset to center (min is about 800, -1275, -375)
                pos += [-600, 1400, 400]
                # set length scale of grid (range is about 1750, 3350, 1900)
                L = 2750
            elif cap == 'mtng':
                pos += [100, 100, 100]
                L = 2000
            else:
                raise ValueError

            # Check if all tracers are inside the box
            if np.any(pos < 0) or np.any(pos > L):
                logging.error('Error! Some tracers outside of box!')
                return False

            # Set mesh resolution
            N = int(L/1000*128)

            # Compute P(k)
            if 'Pk' in summaries:
                MAS = 'NGP'
                field = MA(pos, L, N, MAS=MAS).astype(np.float32)
                run_pylians(
                    field, o, ['Pk'], L, axis=0, MAS=MAS,
                    num_threads=threads, use_rsd=False
                )
            if 'Bk' in summaries:
                MAS = 'TSC'
                field = MA(pos, L, N, MAS=MAS).astype(np.float32)
                run_pylians(
                    field, o, ['Bk'], L, axis=0, MAS=MAS,
                    num_threads=threads, use_rsd=False
                )

            # Compute other summaries
            others = [s for s in summaries if (
                ('Pk' not in s) and ('Bk' not in s))]
            if len(others) > 0:
                run_summarizer(
                    pos, np.zeros_like(pos), cosmo[2], z, L, N,
                    o, others,
                    threads
                )
    return True


@ timing_decorator
@ hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )

    from_scratch = cfg.diag.from_scratch
    summaries = cfg.diag.summaries
    threads = cfg.diag.threads
    if threads == -1:
        threads = os.cpu_count()

    cosmo = cosmo_to_astropy(cfg.nbody.cosmo)

    logging.info(f'Computing diagnostics: {summaries}')

    all_done = True

    # measure rho diagnostics
    if cfg.diag.all or cfg.diag.density:
        done = summarize_rho(
            source_path, cfg.nbody.L,
            threads=threads, from_scratch=from_scratch,
            config=cfg
        )
        all_done &= done
    else:
        logging.info('Skipping rho diagnostics')

    # set mesh resolution
    N = (cfg.nbody.L//1000)*128  # fixed resolution at 128 cells per 1000 Mpc/h

    # measure halo diagnostics
    if cfg.diag.all or cfg.diag.halo:
        done = summarize_tracer(
            source_path, cfg.nbody.L, N, cosmo,
            density=cfg.diag.halo_density,
            proxy=cfg.diag.halo_proxy,
            threads=threads, from_scratch=from_scratch,
            type='halo',
            summaries=summaries,
            config=cfg
        )
        all_done &= done
    else:
        logging.info('Skipping halo diagnostics')

    # measure galaxy diagnostics
    if cfg.diag.all or cfg.diag.galaxy:
        done = summarize_tracer(
            source_path, cfg.nbody.L, N, cosmo,
            density=cfg.diag.galaxy_density,
            proxy=cfg.diag.galaxy_proxy,
            threads=threads, from_scratch=from_scratch,
            type='galaxy', hod_seed=cfg.bias.hod.seed,
            summaries=summaries,
            config=cfg
        )
        all_done &= done
    else:
        logging.info('Skipping galaxy diagnostics')

    # measure lightcone diagnostics
    if cfg.diag.all or cfg.diag.ngc:
        done = summarize_lightcone(
            source_path, cfg.nbody.L, N, Planck18,
            cap='ngc',
            threads=threads, from_scratch=from_scratch,
            hod_seed=cfg.bias.hod.seed, aug_seed=cfg.survey.aug_seed,
            summaries=summaries,
            config=cfg
        )
        all_done &= done
    else:
        logging.info('Skipping ngc_lightcone diagnostics')
    if cfg.diag.all or cfg.diag.sgc:
        done = summarize_lightcone(
            source_path, cfg.nbody.L, N, Planck18,
            cap='sgc',
            threads=threads, from_scratch=from_scratch,
            hod_seed=cfg.bias.hod.seed, aug_seed=cfg.survey.aug_seed,
            summaries=summaries,
            config=cfg
        )
        all_done &= done
    else:
        logging.info('Skipping sgc_lightcone diagnostics')
    if cfg.diag.all or cfg.diag.mtng:
        done = summarize_lightcone(
            source_path, cfg.nbody.L, N, Planck18,
            cap='mtng',
            threads=threads, from_scratch=from_scratch,
            hod_seed=cfg.bias.hod.seed, aug_seed=cfg.survey.aug_seed,
            summaries=summaries,
            config=cfg
        )
        all_done &= done
    else:
        logging.info('Skipping mtng_lightcone diagnostics')

    if all_done:
        logging.info('All diagnostics computed successfully')
    else:
        logging.error('Some diagnostics failed to compute')


if __name__ == "__main__":
    main()
