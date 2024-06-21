import os
from os.path import join as pjoin
import logging
import numpy as np
from ..utils import timing_decorator, get_particle_mass
import MAS_library as MASL


@timing_decorator
def gen_white_noise(N, seed=None):
    """Generate ICs in Fourier space."""
    if seed is not None:
        np.random.seed(seed)
    ic = np.fft.rfftn(np.random.randn(N, N, N)) / N ** (1.5)
    return ic


@timing_decorator
def load_white_noise(path_to_ic, N, quijote=False):
    """Loading in Fourier space."""
    logging.info(f"Loading ICs from {path_to_ic}...")
    num_modes_last_d = N // 2 + 1
    with open(path_to_ic, 'rb') as f:
        if quijote:
            _ = np.fromfile(f, np.uint32, 1)[0]
        modes = np.fromfile(f, np.complex128, -1)
        modes = modes.reshape((N, N, num_modes_last_d))
    return modes


@timing_decorator
def save_nbody(savedir, rho, fvel, pos, vel,
               save_particles=True, save_velocities=True):
    os.makedirs(savedir, exist_ok=True)
    np.save(pjoin(savedir, 'rho.npy'), rho)  # density contrast
    if save_velocities:
        np.save(pjoin(savedir, 'fvel.npy'), fvel)  # velocity field [km/s]
    if save_particles:
        np.save(pjoin(savedir, 'ppos.npy'), pos)  # particle positions [Mpc/h]
        np.save(pjoin(savedir, 'pvel.npy'), vel)  # particle velocities [km/s]
    logging.info(f'Saved to {savedir}.')


def assign_field(pos, BoxSize, Ngrid, MAS, value=None, verbose=False):
    """ Assign particle positions to a grid.
    Note: 
        For overdensity and density contrast, divide by the mean and
        subtract 1

    Args:
        pos (np.array): (N, 3) array of particle positions
        BoxSize (float): size of the box
        Ngrid (int): number of grid points
        MAS (str): mass assignment scheme (NGP, CIC, TSC, PCS)
        value (np.array, optional): (N,) array of values to assign
        verbose (bool, optional): print information on progress
    """
    # define 3D density field
    delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)

    # construct 3D density field
    MASL.MA(pos, delta, BoxSize, MAS, W=value, verbose=verbose)

    return delta


def rho_and_vfield(ppos, pvel, BoxSize, Ngrid, MAS, omega_m, h, verbose=False):
    """
    Measure the 3D density and velocity field from particles.

    Args:
        ppos (np.array): (N, 3) array of particle positions
        pvel (np.array): (N, 3) array of particle velocities
        BoxSize (float): size of the box
        Ngrid (int): number of grid points
        MAS (str): mass assignment scheme (NGP, CIC, TSC, PCS)
        omega_m (float): matter density
        h (float): Hubble constant
        verbose (bool, optional): print information on progress
    """
    ppos, pvel = ppos.astype(np.float32), pvel.astype(np.float32)

    Npart = len(ppos)

    # Get particle count field
    count = assign_field(ppos, BoxSize, Ngrid, MAS,
                         value=None, verbose=verbose)

    # Sum velocity field
    vel = np.stack([
        assign_field(ppos, BoxSize, Ngrid, MAS,
                     value=pvel[:, i], verbose=verbose)
        for i in range(3)
    ], axis=-1)

    # Normalize
    m_particle = get_particle_mass(Npart, BoxSize, omega_m, h)
    rho = count*m_particle
    vel /= count[..., None]

    return rho, vel  # TODO: Implement interpolation for NaNs?
