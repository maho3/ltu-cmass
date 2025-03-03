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
from .tools import calcPk, calcBk_bfast, get_mesh_resolution, compute_Wavelets
from .tools import store_summary, check_existing
from ..survey.tools import sky_to_xyz
import datetime


def run_pylians(
    field, summaries,
    box_size, axis, num_threads, use_rsd,
    MAS='CIC', cache_dir=None
):
    # Only for power spectrum
    accept_summaries = ['Pk', 'Bk']

    for summary_name in summaries:
        pfx = 'z' if use_rsd else ''
        if summary_name == 'Pk':
            k, Pk = calcPk(field, box_size, axis=axis,
                           MAS=MAS, threads=num_threads)
            out = {
                pfx+'Pk_k3D': k,
                pfx+'Pk': Pk
            }
        elif summary_name == 'Bk':
            k123, Bk, Qk, k, Pk = calcBk_bfast(
                field, box_size, axis=axis,
                MAS=MAS, threads=num_threads,
                cache_dir=cache_dir)
            out = {
                pfx+'Bk_k123': k123,
                pfx+'Bk': Bk,
                pfx+'Qk': Qk,
                pfx+'bPk_k3D': k123,
                pfx+'bPk': Pk
            }
        elif summary_name not in accept_summaries:
            logging.error(f'{summary_name} not yet implemented in Pylians')
            continue
    return out

def run_wavelets(field):
    order0, order12 = compute_Wavelets(field)
    out ={
        'S0': order0,
        'S12': order12
    }
    return out

def run_summarizer(
    pos, vel, h, redshift, box_size, grid_size,
    group, summaries, threads
):
    raise NotImplementedError('Summarizer not yet implemented')  # TODO
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


def save_group(file, data, attrs=None, a=None, config=None, save_HOD=False):
    logging.info(f'Saving {len(data)} datasets to {file}')
    with h5py.File(file, 'a') as f:
        if a is not None:
            group = f.require_group(a)
        else:
            group = f
        if attrs is not None:
            for key, value in attrs.items():
                group.attrs[key] = value
            group.attrs['timestamp'] = datetime.datetime.now().isoformat()
        for key, value in data.items():
            if key in group:
                del group[key]
            group.create_dataset(key, data=value)

        if config is not None:
            save_configuration(f, config, save_HOD=save_HOD)


def summarize_rho(
    source_path, L,
    threads=16, from_scratch=True,
    config=None
):
    # check for file keys
    filename = join(source_path, 'nbody.h5')
    if not os.path.isfile(filename):
        logging.error(f'rho file {filename} not found')
        return False
    with h5py.File(filename, 'r') as f:
        alist = list(f.keys())

    # check if diagnostics already computed, delete if from_scratch
    outpath = join(source_path, 'diag', 'rho.h5')
    summaries = check_existing(filename, summaries, from_scratch, rsd=False)
    if len(summaries) == 0:
        logging.info('All diagnostics already saved. Skipping...')
        return True
    logging.info(f'Computing diagnostics to save to: {outpath}')

    # compute diagnostics and save
    for a in alist:
        logging.info(f'Processing density field a={a}')
        with h5py.File(filename, 'r') as f:
            rho = f[a]['rho'][...].astype(np.float32)
        out_data = {}
        if 'Pk' in summaries:
            out = run_pylians(
                rho, ['Pk'], L, axis=0, MAS='CIC',
                num_threads=threads, use_rsd=False
            )
            out_data.update(out)
        if 'Bk' in summaries:
            if config is not None:
                cache_dir = join(config.meta.wdir, 'scratch', 'cache')
            else:
                cache_dir = None
            out = run_pylians(
                rho, ['Bk'], L, axis=0, MAS='CIC',
                num_threads=threads, use_rsd=False,
                cache_dir=cache_dir
            )
            out_data.update(out)
        if 'WST' in summaries:
            out = run_wavelets(rho)
            out_data.update(out)
        if len(out) > 0:
            save_group(outpath, out_data, None, a, config)
    return True


def summarize_tracer(
    source_path, L, cosmo,
    density=None, proxy=None, high_res=False,
    threads=16, from_scratch=True,
    type='halo', hod_seed=None,
    summaries=['Pk'],
    config=None
):
    postfix = 'halos.h5' if type == 'halo' else f'galaxies/hod{hod_seed:05}.h5'

    # check for file keys
    filename = join(source_path, postfix)
    if not os.path.isfile(filename):
        logging.error(f'File not found: {filename}')
        return False
    with h5py.File(filename, 'r') as f:
        alist = list(f.keys())

    # check if diagnostics already computed
    if type == 'galaxy':
        os.makedirs(join(source_path, 'diag', 'galaxies'), exist_ok=True)
    outpath = join(source_path, 'diag', postfix)
    summaries = check_existing(outpath, summaries, from_scratch, rsd=True)
    if len(summaries) == 0:
        logging.info('All diagnostics already saved. Skipping...')
        return True
    logging.info(f'Computing diagnostics to save to: {outpath}')

    # compute overdensity cut
    Ncut = None if density is None else int(density * L**3)

    # compute diagnostics and save
    for a in alist:
        z = 1/float(a) - 1
        logging.info(f'Processing {type} catalog a={a}')

        # Load
        with h5py.File(filename, 'r') as f:
            pos = f[a]['pos'][...].astype(np.float32)
            vel = f[a]['vel'][...].astype(np.float32)
            if str(proxy) in f[a].keys():
                mass = f[a][proxy][...].astype(np.float32)
            else:
                mass = None
        pos %= L  # Ensure all tracers inside box

        # Mask out low mass tracers (to match number density)
        out_attrs = {}
        if density is not None:
            if proxy is None:
                logging.warning(
                    'Proxy is set to None. Not rank-ordering.')
                mass = np.arange(len(pos))
                np.random.shuffle(mass)
            else:
                if mass is None:
                    raise KeyError(
                        f'{proxy} not found in {type} file at a={a}')
                if len(pos) <= Ncut:
                    logging.warning(f'Not enough {type} tracers in {a}')
                logging.info(
                    f'Cutting top {Ncut} out of {len(pos)} {type} '
                    'tracers to match number density')
            mask = np.argsort(mass)[-Ncut:]  # Keep top Ncut tracers
            pos = pos[mask]
            vel = vel[mask]
            mass = mass[mask]

            out_attrs['density'] = float(density)
        else:
            out_attrs['density'] = np.nan

        # Noise out positions (we do not probe less than Lnoise)
        Lnoise = (1000/128)/np.sqrt(3)  # Set by CHARM resolution
        pos += np.random.randn(*pos.shape) * Lnoise

        # Compute P(k)
        out_data = {}
        if 'Pk' in summaries:
            N, MAS = get_mesh_resolution(L, high_res)

            # real space
            field = MA(pos, L, N, MAS=MAS).astype(np.float32)
            out = run_pylians(
                field, ['Pk'], L, axis=0, MAS=MAS,
                num_threads=threads, use_rsd=False
            )
            out_data.update(out)

            # redshift space
            field = MAz(pos, vel, L, N, cosmo, z, MAS=MAS,
                        axis=0).astype(np.float32)
            out = run_pylians(
                field, ['Pk'], L, axis=0, MAS=MAS,
                num_threads=threads, use_rsd=True
            )
            out_data.update(out)
        # Compute B(k)
        if 'Bk' in summaries:
            N, MAS = get_mesh_resolution(L, high_res=False)  # No high-res
            MAS = 'TSC'
            if config is not None:
                cache_dir = join(config.meta.wdir, 'scratch', 'cache')
            else:
                cache_dir = None

            # real space
            field = MA(pos, L, N, MAS=MAS).astype(np.float32)
            out = run_pylians(
                field, ['Bk'], L, axis=0, MAS=MAS,
                num_threads=threads, use_rsd=False,
                cache_dir=cache_dir
            )
            out_data.update(out)

            # redshift space
            field = MAz(pos, vel, L, N, cosmo, z, MAS=MAS,
                        axis=0).astype(np.float32)
            out = run_pylians(
                field, ['Bk'], L, axis=0, MAS=MAS,
                num_threads=threads, use_rsd=True,
                cache_dir=cache_dir
            )
            out_data.update(out)
        # Compute Wavelets
        if 'WST' in summaries: # check
            N, MAS = get_mesh_resolution(L, high_res=False)  # No high-res MAS='CIC'
            
            if config is not None:    # do we need this?
                cache_dir = join(config.meta.wdir, 'scratch', 'cache')
            else:
                cache_dir = None

            # real space
            field = MA(pos, L, N, MAS=MAS).astype(np.float32)
            out = run_wavelets(field)
            out_data.update(out)
            
            # redshift space
            field = MAz(pos, vel, L, N, cosmo, z, MAS=MAS,
                        axis=0).astype(np.float32)
            out = run_wavelets(field)
            out_data.update(out)
            
        # Compute other summaries
        others = [s for s in summaries if (('Pk' not in s) and ('Bk' not in s) and ('WST' not in s))]
        if len(others) > 0:
            out = run_summarizer(
                pos, vel, cosmo.h, z, L, N, others,
                threads
            )
            out_data.update(out)

        if mass is not None:
            # measure halo mass function
            be = np.linspace(12.5, 16, 100)
            hist, _ = np.histogram(mass, bins=be)
            out_data['mass_bins'] = be
            out_data['mass_hist'] = hist

        save_group(outpath, out_data, out_attrs, a,
                   config, save_HOD=(type == 'galaxy'))
    return True


def summarize_lightcone(
    source_path, L, cosmo,
    cap='ngc', high_res=False,
    threads=16, from_scratch=True,
    hod_seed=None, aug_seed=None,
    summaries=['Pk'],
    config=None
):
    postfix = f'{cap}_lightcone/hod{hod_seed:05}_aug{aug_seed:05}.h5'

    # check for file keys
    filename = join(source_path, postfix)
    if not os.path.isfile(filename):
        logging.error(f'File not found: {filename}')
        return False

    # check if diagnostics already computed
    os.makedirs(join(source_path, 'diag', f'{cap}_lightcone'), exist_ok=True)
    outpath = join(source_path, 'diag', postfix)
    summaries = check_existing(outpath, summaries, from_scratch, rsd=False)
    if len(summaries) == 0:
        logging.info('All diagnostics already saved. Skipping...')
        return True
    logging.info(f'Computing diagnostics to save to: {outpath}')

    # Load
    with h5py.File(filename, 'r') as f:
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

    out_data = {}
    # Compute P(k)
    if 'Pk' in summaries:
        N, MAS = get_mesh_resolution(L, high_res)

        field = MA(pos, L, N, MAS=MAS).astype(np.float32)
        out = run_pylians(
            field, ['Pk'], L, axis=0, MAS=MAS,
            num_threads=threads, use_rsd=False
        )
        out_data.update(out)
    # Compute B(k)
    if 'Bk' in summaries:
        N, MAS = get_mesh_resolution(L, high_res=False)  # No high-res
        MAS = 'TSC'
        if config is not None:
            cache_dir = join(config.meta.wdir, 'scratch', 'cache')
        else:
            cache_dir = None

        field = MA(pos, L, N, MAS=MAS).astype(np.float32)
        out = run_pylians(
            field, ['Bk'], L, axis=0, MAS=MAS,
            num_threads=threads, use_rsd=False,
            cache_dir=cache_dir
        )
        out_data.update(out)

    # Compute Wavelets
    if 'WST' in summaries:
        N, MAS = get_mesh_resolution(L, high_res=False)  # No high-res
        
        if config is not None:
            cache_dir = join(config.meta.wdir, 'scratch', 'cache')
        else:
            cache_dir = None

        field = MA(pos, L, N, MAS=MAS).astype(np.float32)
        out = run_wavelets(field)
        out_data.update(out)
            
    # Compute other summaries
    others = [s for s in summaries if (
        ('Pk' not in s) and ('Bk' not in s) and ('WST' not in s))]
    if len(others) > 0:
        out = run_summarizer(
            pos, np.zeros_like(pos), cosmo[2], z, L, N,
            others,
            threads
        )
        out_data.update(out)

    save_group(outpath, out_data, None, None,
               config, save_HOD=True)
    return True


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    os.makedirs(join(source_path, 'diag'), exist_ok=True)

    from_scratch = cfg.diag.from_scratch
    summaries = cfg.diag.summaries
    threads = cfg.diag.threads
    if threads == -1:
        threads = os.cpu_count()

    cosmo = cosmo_to_astropy(cfg.nbody.cosmo)

    logging.info(f'Selected diagnostics: {summaries}')

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

    # measure halo diagnostics
    if cfg.diag.all or cfg.diag.halo:
        done = summarize_tracer(
            source_path, cfg.nbody.L, cosmo,
            density=cfg.diag.halo_density,
            proxy=cfg.diag.halo_proxy,
            high_res=cfg.diag.high_res,
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
            source_path, cfg.nbody.L, cosmo,
            density=cfg.diag.galaxy_density,
            proxy=cfg.diag.galaxy_proxy,
            high_res=cfg.diag.high_res,
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
            source_path, cfg.nbody.L, Planck18,
            cap='ngc', high_res=cfg.diag.high_res,
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
            source_path, cfg.nbody.L, Planck18,
            cap='sgc', high_res=cfg.diag.high_res,
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
            source_path, cfg.nbody.L, Planck18,
            cap='mtng', high_res=cfg.diag.high_res,
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
