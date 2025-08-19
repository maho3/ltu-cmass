
import functools
import logging
import datetime
import os
import shutil
from os.path import join
from astropy.cosmology import Cosmology, FlatLambdaCDM
from colossus.cosmology import cosmology as csm
from omegaconf import OmegaConf


def get_source_path(wdir, suite, sim, L, N, lhid, check=True):
    # get the path to the source directory, and check at each level
    sim_dir = join(wdir, suite, sim)
    cfg_dir = join(sim_dir, f'L{L}-N{N}')
    lh_dir = join(cfg_dir, str(lhid))

    if check:
        if not os.path.isdir(sim_dir):
            raise ValueError(
                f"Simulation directory {sim_dir} does not exist.")
        if not os.path.isdir(cfg_dir):
            raise ValueError(
                f"Configuration directory {cfg_dir} does not exist.")
        if not os.path.isdir(lh_dir):
            raise ValueError(
                f"Latin hypercube directory {lh_dir} does not exist.")
    return lh_dir


def timing_decorator(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        logging.info(f"Running {func.__name__}...")
        t0 = datetime.datetime.now()
        out = func(*args, **kwargs)
        dt = (datetime.datetime.now() - t0).total_seconds()
        logging.info(
            f"Finished {func.__name__}... "
            f"({int(dt//60)}m{int(dt%60)}s)")
        return out
    return wrapper


def clean_up(hydra):
    """Decorator to clean up the Hydra log directory after function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                try:
                    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                    logging.info(f"Cleaning up log directory: {logdir}")
                    shutil.rmtree(logdir, ignore_errors=True)
                except Exception as e:
                    logging.warning(
                        f"Cleanup failed: {e}")

        return wrapper
    return decorator


def save_cfg(source_path, cfg, field=None):
    if os.path.isfile(join(source_path, 'config.yaml')):
        old_cfg = OmegaConf.load(join(source_path, 'config.yaml'))
        if field is not None:
            cfg = OmegaConf.masked_copy(cfg, field)
            cfg = OmegaConf.merge(old_cfg, cfg)
    filename = join(source_path, 'config.yaml')
    if os.path.isfile(filename):
        os.remove(filename)
    OmegaConf.save(cfg, filename)


def load_params(index, cosmofile):
    # load cosmology parameters
    # [Omega_m, Omega_b, h, n_s, sigma8]
    if index == "fid":
        return [0.3175, 0.049, 0.6711, 0.9624, 0.834]
    with open(cosmofile, 'r') as f:
        content = f.readlines()[index]
    content = [float(x) for x in content.split()]
    return content


def cosmo_to_astropy(params=None, omega_m=None, omega_b=None,
                     h=None, n_s=None, sigma8=None):
    """
    Converts a list of cosmological parameters into an astropy cosmology
    object. Note, ignores s8 and n_s parameters, which are not used in astropy.
    """
    if isinstance(params, Cosmology):
        return params
    try:
        params = list(params)
        return FlatLambdaCDM(H0=params[2]*100, Om0=params[0], Ob0=params[1])
    except TypeError:
        return FlatLambdaCDM(H0=h*100, Om0=omega_m, Ob0=omega_b)


def get_particle_mass(N, L, omega_m, h):
    """
    M_particle = Omega_m * rho_crit * Volume / NumParticles

    Args:
        N (int): number of particles per dimension
        L (float): box side length (Mpc/h)
        omega_m (float): matter density
        h (float): Hubble constant
    """
    cosmo = cosmo_to_astropy(omega_m=omega_m, h=h)
    rho_crit = cosmo.critical_density0.to('Msun/Mpc^3').value
    volume = L**3  # (Mpc/h)^3
    NumParticles = N**3
    return omega_m * rho_crit * volume / (NumParticles * h**2)  # Msun/h


def cosmo_to_colossus(cpars):
    try:
        params = list(cpars)
    except TypeError:
        return params

    params = {'flat': True, 'H0': 100*cpars[2], 'Om0': cpars[0],
              'Ob0': cpars[1], 'sigma8': cpars[4], 'ns': cpars[3]}
    csm.addCosmology('myCosmo', **params)
    cosmo = csm.setCosmology('myCosmo')
    return cosmo


def save_configuration_h5(file, config, save_HOD=True, save_noise=False):
    file.attrs['config'] = OmegaConf.to_yaml(config)
    file.attrs['cosmo_names'] = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma8']
    file.attrs['cosmo_params'] = config.nbody.cosmo

    if save_HOD:
        file.attrs['HOD_model'] = config.bias.hod.model
        file.attrs['HOD_seed'] = config.bias.hod.seed

        keys = sorted(list(config.bias.hod.theta.keys()))
        file.attrs['HOD_names'] = keys
        file.attrs['HOD_params'] = [config.bias.hod.theta[k] for k in keys]
    if save_noise:
        file.attrs['noise_dist'] = config.noise.dist
        file.attrs['noise_radial'] = config.noise.radial
        file.attrs['noise_transverse'] = config.noise.transverse
