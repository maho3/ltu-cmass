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

from ..utils import (
    get_source_path, timing_decorator, cosmo_to_astropy, save_configuration_h5)
from ..nbody.tools import parse_nbody_config
from ..bias.apply_hod import parse_hod
from .tools import MA, MAz, get_box_catalogue, get_box_catalogue_rsd
from .tools import calcPk, calcBk_bfast, get_mesh_resolution
from .tools import store_summary, check_existing
from .tools import parse_noise
from ..survey.tools import sky_to_xyz, sky_to_unit_vectors
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
            save_configuration_h5(f, config, save_HOD=save_HOD)


def summarize_rho(
    source_path, L,
    threads=16, from_scratch=True, focus_z=None,
    summaries=['Pk'], config=None
):
    # check for file keys
    filename = join(source_path, 'nbody.h5')
    if not os.path.isfile(filename):
        logging.error(f'rho file {filename} not found')
        return False
    with h5py.File(filename, 'r') as f:
        alist = list(f.keys())

    # Filter alist to only include the closest to a specified redshift
    if focus_z is not None:
        i = np.argmin(np.abs(np.array(alist, dtype=float) - 1./(1 + focus_z)))
        alist = [alist[i]]

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
        if len(out) > 0:
            save_group(outpath, out_data, None, a, config)
    return True


def summarize_tracer(
    source_path, L, cosmo,
    density=None, proxy=None, high_res=False, use_ngp=False,
    threads=16, from_scratch=True, focus_z=None,
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

    # Filter alist to only include the closest to a specified redshift
    if focus_z is not None:
        i = np.argmin(np.abs(np.array(alist, dtype=float) - 1./(1 + focus_z)))
        alist = [alist[i]]

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

        # Save number density of tracers
        out_attrs['nbar'] = len(pos) / L**3  # Number density (h/Mpc)^3
        out_attrs['log10nbar'] = \
            np.log10(len(pos)) - 3 * np.log10(L)  # for numerical precision
        out_attrs['high_res'] = high_res and not use_ngp
        out_attrs['noise_dist'] = config.noise.dist
        out_attrs['noise_radial'] = config.noise.radial
        out_attrs['noise_transverse'] = config.noise.transverse

        # Noise in-voxel
        if config.bias.hod.noise_uniform:
            delta = L / config.nbody.N
            pos += np.random.uniform(-delta/2, delta/2, size=pos.shape)
            pos = np.mod(pos, L)

        # Get unit vectors and add noise along each direction
        # RSDs are applied along the 0th axis
        r_hat, e_phi, e_theta = np.identity(3)
        noise = np.random.randn(*pos.shape)
        pos += r_hat * noise[:, 0, None] * config.noise.radial
        pos += e_phi * noise[:, 1, None] * config.noise.transverse
        pos += e_theta * noise[:, 2, None] * config.noise.transverse

        # Compute P(k)
        out_data = {}
        if 'Pk' in summaries:
            N, MAS = get_mesh_resolution(L, high_res, use_ngp)

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
        if 'Bk' in summaries:   # high-res takes too much memory
            N, MAS = get_mesh_resolution(L, high_res=False, use_ngp=use_ngp)
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

        # Compute other summaries
        others = [s for s in summaries if (('Pk' not in s) and ('Bk' not in s))]
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
    cap='ngc', high_res=False, use_ngp=False,
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

    # Get unit vectors and add noise along each direction
    r_hat, e_phi, e_theta = sky_to_unit_vectors(ra, dec)
    noise = np.random.randn(*pos.shape)
    pos += r_hat * noise[:, 0, None] * config.noise.radial
    pos += e_phi * noise[:, 1, None] * config.noise.transverse
    pos += e_theta * noise[:, 2, None] * config.noise.transverse

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
    elif cap == 'simbig':
        # offset to center (min is about 850, -650, -175)
        pos += [-650, 800, 250]
        L = 2000
    else:
        raise ValueError

    # Check if all tracers are inside the box
    if np.any(pos < 0) or np.any(pos > L):
        logging.error('Error! Some tracers outside of box!')
        raise ValueError(
            f'position out of bounds for {cap}_lightcone: '
            f'{np.min(pos, axis=0)} {np.max(pos, axis=0)}')

    out_attrs = {}
    # Save number density of tracers
    out_attrs['nbar'] = len(pos) / L**3  # Number density (h/Mpc)^3
    out_attrs['log10nbar'] = np.log10(
        len(pos)) - 3 * np.log10(L)  # for numerical precision
    out_attrs['high_res'] = high_res and not use_ngp
    out_attrs['noise_dist'] = config.noise.dist
    out_attrs['noise_radial'] = config.noise.radial
    out_attrs['noise_transverse'] = config.noise.transverse

    out_data = {}
    # Compute P(k)
    if 'Pk' in summaries:
        N, MAS = get_mesh_resolution(L, high_res, use_ngp)

        field = MA(pos, L, N, MAS=MAS).astype(np.float32)
        out = run_pylians(
            field, ['Pk'], L, axis=0, MAS=MAS,
            num_threads=threads, use_rsd=False
        )
        out_data.update(out)
    # Compute B(k)
    if 'Bk' in summaries:  # high-res takes too much memory
        N, MAS = get_mesh_resolution(L, high_res=False, use_ngp=use_ngp)
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

    # Compute other summaries
    others = [s for s in summaries if (
        ('Pk' not in s) and ('Bk' not in s))]
    if len(others) > 0:
        out = run_summarizer(
            pos, np.zeros_like(pos), cosmo[2], z, L, N,
            others,
            threads
        )
        out_data.update(out)

    # Save n(z)
    zbins = np.linspace(0.4, 0.7, 101)  # spacing in dz = 0.003
    out_data['nz'], out_data['nz_bins'] = np.histogram(rdz[:, -1], bins=zbins)

    save_group(outpath, out_data, out_attrs, None,
               config, save_HOD=True)
    return True


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)

    # parse noise (seeded by lhid and hod seed)
    noise_seed = int(cfg.nbody.lhid*1e6 + cfg.bias.hod.seed)
    cfg.noise.radial, cfg.noise.transverse = \
        parse_noise(seed=noise_seed,
                    dist=cfg.noise.dist,
                    params=cfg.noise.params)

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
            focus_z=cfg.diag.focus_z,
            summaries=summaries, config=cfg
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
            use_ngp=cfg.diag.use_ngp,
            threads=threads, from_scratch=from_scratch,
            focus_z=cfg.diag.focus_z,
            type='halo',
            summaries=summaries,
            config=cfg
        )
        all_done &= done
    else:
        logging.info('Skipping halo diagnostics')

    # Save with original hod_seed
    if cfg.bias.hod.seed == 0:
        hod_seed = cfg.bias.hod.seed
    else:
        # (parse_hod modifies it to lhid*1e6 + hod_seed)
        hod_seed = int(cfg.bias.hod.seed - cfg.nbody.lhid * 1e6)

    # measure galaxy diagnostics
    if cfg.diag.all or cfg.diag.galaxy:
        done = summarize_tracer(
            source_path, cfg.nbody.L, cosmo,
            density=cfg.diag.galaxy_density,
            proxy=cfg.diag.galaxy_proxy,
            high_res=cfg.diag.high_res,
            use_ngp=cfg.diag.use_ngp,
            threads=threads, from_scratch=from_scratch,
            focus_z=cfg.diag.focus_z,
            type='galaxy', hod_seed=hod_seed,
            summaries=summaries,
            config=cfg
        )
        all_done &= done
    else:
        logging.info('Skipping galaxy diagnostics')

    # measure lightcone diagnostics
    for cap in ['ngc', 'sgc', 'mtng', 'simbig']:
        if cfg.diag.all or getattr(cfg.diag, f'{cap}'):
            done = summarize_lightcone(
                source_path, cfg.nbody.L,
                cosmo=Planck18,  # Diagnostics for lightcone stats use fiducial cosmology
                cap=cap,
                high_res=cfg.diag.high_res,
                use_ngp=cfg.diag.use_ngp,
                threads=threads, from_scratch=from_scratch,
                hod_seed=hod_seed, aug_seed=cfg.survey.aug_seed,
                summaries=summaries,
                config=cfg
            )
            all_done &= done
        else:
            logging.info(f'Skipping {cap} lightcone diagnostics')

    if all_done:
        logging.info('All diagnostics computed successfully')
    else:
        logging.error('Some diagnostics failed to compute')


if __name__ == "__main__":
    main()
