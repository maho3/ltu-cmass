
import logging
import datetime
import os
from os.path import join as pjoin
from astropy.cosmology import FlatLambdaCDM


def get_source_path(cfg, simtype, check=True):
    wdir = cfg.meta.wdir
    nbody = cfg.nbody

    # get the path to the source directory, and check at each level
    sim_dir = pjoin(wdir, nbody.suite, simtype)
    cfg_dir = pjoin(sim_dir, f'L{nbody.L}-N{nbody.N}')
    lh_dir = pjoin(cfg_dir, str(nbody.lhid))

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

    # check if params is a list
    if isinstance(params, list):
        return FlatLambdaCDM(H0=params[2]*100, Om0=params[0], Ob0=params[1])
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
