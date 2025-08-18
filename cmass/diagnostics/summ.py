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
from scipy.spatial.transform import Rotation as R
import subprocess

from ..utils import (
    get_source_path, timing_decorator, clean_up, cosmo_to_astropy,
    save_configuration_h5
)
from ..nbody.tools import parse_nbody_config
from ..bias.apply_hod import parse_hod
from .tools import (
    get_mesh_resolution, noise_positions, store_summary, check_existing,
    parse_noise, _get_snapshot_alist
)
from .calculations import (
    MA, MAz, calcPk, calcBk_bfast,
    get_box_catalogue, get_box_catalogue_rsd  # not used
)
from .tools import save_group
from ..survey.tools import sky_to_xyz, sky_to_unit_vectors


def run_pylians(
    field, box_size, axis, num_threads, use_rsd, MAS='CIC'
):
    # Only for power spectrum
    pfx = 'z' if use_rsd else ''
    k, Pk = calcPk(field, box_size, axis=axis,
                   MAS=MAS, threads=num_threads)
    out = {
        pfx+'Pk_k3D': k,
        pfx+'Pk': Pk
    }
    return out


def run_bfast(
    field, box_size, axis, num_threads, use_rsd, MAS='CIC', cache_dir=None
):
    # Only for bispectrum
    pfx = 'z' if use_rsd else ''
    k123, Bk, Qk, k, Pk = calcBk_bfast(
        field, box_size, axis=axis,
        MAS=MAS, threads=num_threads,
        cache_dir=cache_dir
    )
    out = {
        pfx+'Bk_k123': k123,
        pfx+'Bk': Bk,
        pfx+'Qk': Qk,
        pfx+'bPk_k3D': k123,
        pfx+'bPk': Pk
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


def summarize_rho(
    source_path, L,
    threads=16, from_scratch=True, focus_z=None,
    summaries=['Pk'], config=None
):
    # get source file
    filename = join(source_path, 'nbody.h5')
    if not os.path.isfile(filename):
        logging.error(f'File not found: {filename}')
        return False
    alist = _get_snapshot_alist(filename, focus_z=focus_z)

    # check if diagnostics already computed, delete if from_scratch
    outpath = join(source_path, 'diag', 'rho.h5')
    summaries = check_existing(filename, summaries, from_scratch, rsd=False)
    if len(summaries) == 0:
        logging.info('All diagnostics already saved. Skipping...')
        return True

    # compute diagnostics and save
    logging.info(f'Computing diagnostics to save to: {outpath}')
    for a in alist:
        logging.info(f'Processing density field a={a}')
        # Load
        with h5py.File(filename, 'r') as f:
            rho = f[a]['rho'][...].astype(np.float32)

        # Compute P(k)
        out_data = {}
        if 'Pk' in summaries:
            out = run_pylians(
                rho, L, axis=0, MAS='CIC',
                num_threads=threads, use_rsd=False
            )
            out_data.update(out)
        if 'Bk' in summaries:
            cache_dir = None
            if config is not None:
                cache_dir = join(config.meta.wdir, 'scratch', 'cache')

            out = run_bfast(
                rho, L, axis=0, MAS='CIC',
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
    # get source file
    postfix = 'halos.h5' if type == 'halo' else f'galaxies/hod{hod_seed:05}.h5'
    filename = join(source_path, postfix)
    if not os.path.isfile(filename):
        logging.error(f'File not found: {filename}')
        return False
    alist = _get_snapshot_alist(filename, focus_z=focus_z)

    # check if diagnostics already computed
    if type == 'galaxy':
        os.makedirs(join(source_path, 'diag', 'galaxies'), exist_ok=True)
    outpath = join(source_path, 'diag', postfix)
    summaries = check_existing(outpath, summaries, from_scratch, rsd=True)
    if len(summaries) == 0:
        logging.info('All diagnostics already saved. Skipping...')
        return True

    # compute overdensity cut
    Ncut = None if density is None else int(density * L**3)

    # compute diagnostics and save
    logging.info(f'Computing diagnostics to save to: {outpath}')
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

        # Save metadata
        out_attrs = {}
        Ngalaxies = pos.shape[0]
        out_attrs['Ngalaxies'] = Ngalaxies
        out_attrs['boxsize'] = L
        out_attrs['nbar'] = Ngalaxies / L**3  # Number density (h/Mpc)^3
        out_attrs['log10nbar'] = \
            np.log10(Ngalaxies) - 3 * np.log10(L)  # for numerical precision
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
        pos = noise_positions(pos, 0, 0,
                              config.noise.radial,
                              config.noise.transverse)

        # Compute P(k)
        out_data = {}
        if 'Pk' in summaries:
            N, MAS = get_mesh_resolution(L, high_res, use_ngp)

            # real space
            field = MA(pos, L, N, MAS=MAS).astype(np.float32)
            out = run_pylians(
                field, L, axis=0, MAS=MAS,
                num_threads=threads, use_rsd=False
            )
            out_data.update(out)

            # redshift space
            field = MAz(pos, vel, L, N, cosmo, z, MAS=MAS,
                        axis=0).astype(np.float32)
            out = run_pylians(
                field, L, axis=0, MAS=MAS,
                num_threads=threads, use_rsd=True
            )
            out_data.update(out)
        # Compute B(k)
        if 'Bk' in summaries:
            # high-res takes too much memory
            N, MAS = get_mesh_resolution(L, high_res=False, use_ngp=use_ngp)
            cache_dir = None
            if config is not None:
                cache_dir = join(config.meta.wdir, 'scratch', 'cache')

            # real space
            field = MA(pos, L, N, MAS=MAS).astype(np.float32)
            out = run_bfast(
                field, L, axis=0, MAS=MAS,
                num_threads=threads, use_rsd=False,
                cache_dir=cache_dir
            )
            out_data.update(out)

            # redshift space
            field = MAz(pos, vel, L, N, cosmo, z, MAS=MAS,
                        axis=0).astype(np.float32)
            out = run_bfast(
                field, L, axis=0, MAS=MAS,
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


def _center_box(pos, boxpad=1.5):
    """
    Finds a box_size which contains all points, and centers the points
    in a box of size box_size*boxpad. NOTE: no longer do this manually.
    """
    voxel_size = 1000/128
    pos_max, pos_min = np.max(pos, axis=0), np.min(pos, axis=0)
    box_size = np.max(pos_max - pos_min)
    box_size *= boxpad
    box_size = np.ceil(box_size / voxel_size) * voxel_size
    center = (pos_max + pos_min) / 2
    pos += (box_size/2 - center)
    return pos, box_size


def summarize_lightcone_pylians(
    source_path, cosmo,
    cap='ngc', high_res=False, use_ngp=False,
    threads=16, from_scratch=True,
    hod_seed=None, aug_seed=None,
    summaries=['Pk'],
    config=None
):
    # get source file
    postfix = f'{cap}_lightcone/hod{hod_seed:05}_aug{aug_seed:05}.h5'
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

    # compute diagnostics and save
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

    # rotate pos so that line of sight is along 0th axis
    v1 = np.mean(pos, axis=0)
    v2 = np.array([1, 0, 0])
    rotation, _ = R.align_vectors([v2], [v1])
    pos = rotation.apply(pos).astype(np.float32)

    # Center the box
    pos, L = _center_box(pos, boxpad=1.1)

    # Check if all tracers are inside the box
    if np.any(pos < 0) or np.any(pos > L):
        logging.error('Error! Some tracers outside of box!')
        raise ValueError(
            f'position out of bounds for {cap}_lightcone: '
            f'{np.min(pos, axis=0)} {np.max(pos, axis=0)}')

    # Save metadata
    out_attrs = {}
    Ngalaxies = pos.shape[0]
    out_attrs['Ngalaxies'] = Ngalaxies
    out_attrs['boxsize'] = L
    out_attrs['nbar'] = Ngalaxies / L**3  # Number density (h/Mpc)^3
    out_attrs['log10nbar'] = \
        np.log10(Ngalaxies) - 3 * np.log10(L)  # for numerical precision
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
            field, L, axis=0, MAS=MAS,
            num_threads=threads, use_rsd=False
        )
        out_data.update(out)
    # Compute B(k)
    if 'Bk' in summaries:
        # high-res takes too much memory
        N, MAS = get_mesh_resolution(L, high_res=False, use_ngp=use_ngp)
        cache_dir = None
        if config is not None:
            cache_dir = join(config.meta.wdir, 'scratch', 'cache')

        field = MA(pos, L, N, MAS=MAS).astype(np.float32)
        out = run_bfast(
            field, L, axis=0, MAS=MAS,
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


def summarize_lightcone_pypower(
    source_path,
    cap='ngc', high_res=False, use_ngp=False,
    threads=16, from_scratch=True,
    hod_seed=None, aug_seed=None,
    summaries=['Pk'],
    cfg=None
):
    """Builds a command and launches the MPI job."""

    # --- 1. Define Paths and Parameters ---
    # get source file
    postfix = f'{cap}_lightcone/hod{hod_seed:05}_aug{aug_seed:05}.h5'
    data_file = join(source_path, postfix)

    # check if diagnostics already computed
    os.makedirs(join(source_path, 'diag', f'{cap}_lightcone'), exist_ok=True)
    outpath = join(source_path, 'diag', postfix)
    summaries = check_existing(outpath, summaries, from_scratch, rsd=False)
    if len(summaries) == 0:
        logging.info('All diagnostics already saved. Skipping...')
        return True

    # get randoms file
    if cap in ['simbig', 'sgc', 'mtng']:
        randoms_path = get_source_path(
            cfg.meta.wdir, 'abacus', 'randoms', 2000, 256, 0)
    elif cap == 'ngc':
        randoms_path = get_source_path(
            cfg.meta.wdir, 'quijote3gpch', 'randoms', 3000, 384, 0)
    else:
        raise ValueError(f'Unknown cap: {cap}.')

    randoms_file = join(randoms_path, f'{cap}_lightcone',
                        f'hod{0:05}_aug{0:05}.h5')

    # Limit the number of processes to avoid overloading the system
    n_processes = min(threads, 8)  # Limit to 8 processes

    codedir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..'))

    # --- 2. Construct the Command ---
    command_string = f"""
    cd {codedir}
    module purge
    module restore cmass
    source /apps/anvil/external/apps/conda/2024.02/bin/activate
    conda activate pmesh
    which python
    mpirun -n {n_processes} python -m cmass.diagnostics.pypower \
        --data-file {data_file} \
        --randoms-file {randoms_file} \
        --output-file {outpath} \
        --cap {cap} \
        --use-fkp \
        {'--high-res' if high_res else ''} \
        --resampler {'ngp' if use_ngp else 'tsc'} \
        --boxpad 1.1 \
        --noise-radial {cfg.noise.radial} \
        --noise-transverse {cfg.noise.transverse}
    """
    logging.info(f"Launching MPI job with command:\n{command_string}")

    # --- 3. Execute the Command ---
    try:
        result = subprocess.run(
            command_string,
            check=True,
            shell=True,
            capture_output=True,
            text=True,
            executable='/bin/bash',
            env=os.environ
        )
        if result.stdout:
            logging.info("MPI job stdout:\n" + result.stdout)
        logging.info(
            f'MPI job completed successfully. Output saved to {outpath}')
    except FileNotFoundError:
        logging.error(
            "mpirun command not found. Please check your MPI installation.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(
            f"An error occurred during the MPI job (Exit Code: {e.returncode})")
        if e.stdout:
            logging.error("--- Standard Output ---")
            logging.error(e.stdout)
        if e.stderr:
            logging.error("--- Standard Error ---")
            logging.error(e.stderr)
        return False

    with h5py.File(outpath, 'a') as f:
        save_configuration_h5(f, cfg, save_HOD=True)
    return True


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)

    # parse noise (seeded by lhid and hod seed)
    noise_seed = int(cfg.nbody.lhid*1e4 + cfg.bias.hod.seed)
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
        # (parse_hod modifies it to lhid*1e4 + hod_seed)
        hod_seed = int(cfg.bias.hod.seed - cfg.nbody.lhid * 1e4)

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
            if cfg.diag.survey_backend == 'pylians':
                done = summarize_lightcone_pylians(
                    source_path,
                    # Diagnostics for lightcone stats use fiducial cosmology
                    cosmo=Planck18,
                    cap=cap,
                    high_res=cfg.diag.high_res,
                    use_ngp=cfg.diag.use_ngp,
                    threads=threads, from_scratch=from_scratch,
                    hod_seed=hod_seed, aug_seed=cfg.survey.aug_seed,
                    summaries=summaries,
                    config=cfg
                )
            elif cfg.diag.survey_backend == 'pypower':
                done = summarize_lightcone_pypower(
                    source_path,
                    cap=cap,
                    high_res=cfg.diag.high_res,
                    use_ngp=cfg.diag.use_ngp,
                    threads=threads, from_scratch=from_scratch,
                    hod_seed=hod_seed, aug_seed=cfg.survey.aug_seed,
                    cfg=cfg
                )
                # # Bk is still done with old code (TODO: update)
                # done &= summarize_lightcone_pylians(
                #     source_path,
                #     # Diagnostics for lightcone stats use fiducial cosmology
                #     cosmo=Planck18,
                #     cap=cap,
                #     high_res=cfg.diag.high_res,
                #     use_ngp=cfg.diag.use_ngp,
                #     threads=threads, from_scratch=from_scratch,
                #     hod_seed=hod_seed, aug_seed=cfg.survey.aug_seed,
                #     summaries=['Bk'],
                #     config=cfg
                # )
            else:
                raise ValueError(
                    f'Unknown survey backend: {cfg.diag.survey_backend}')
            all_done &= done
        else:
            logging.info(f'Skipping {cap} lightcone diagnostics')

    if all_done:
        logging.info('All diagnostics computed successfully')
    else:
        logging.error('Some diagnostics failed to compute')


if __name__ == "__main__":
    main()
