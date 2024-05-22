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
        verbose (bool, optional): print information on progress
    """
    # define 3D density field
    delta = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)

    # construct 3D density field
    MASL.MA(pos, delta, BoxSize, MAS, W=value, verbose=verbose)

    return delta


def vfield(ppos, pvel, BoxSize, Ngrid, MAS, omega_m, h, verbose=False):
    """
    Measure the 3D velocity field from particles. Also returns the
    density field.

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
    Npart = len(ppos)
    m_particle = get_particle_mass(Npart, BoxSize, omega_m, h)
    rho = assign_field(ppos, BoxSize, Ngrid, MAS,
                       value=m_particle, verbose=verbose)

    pmom = m_particle * pvel
    mom = np.stack([
        assign_field(ppos, BoxSize, Ngrid, MAS,
                     value=pmom[..., i], verbose=verbose)
        for i in range(3)
    ], axis=-1)

    vel = mom / rho[..., None]
    return rho, vel  # TODO: Implement interpolation?


@timing_decorator
def vfield_CIC(ppos, pvel, cfg, interp=True):
    raise NotImplementedError

    nbody = cfg.nbody
    N = nbody.N * nbody.supersampling
    ptcl_spacing = nbody.L / N
    ptcl_grid_shape = (N,)*3
    pmconf = Configuration(ptcl_spacing, ptcl_grid_shape)
    ptcl = Particles.from_pos(pmconf, ppos)

    scale = N / nbody.Nvfield
    mesh = np.zeros([nbody.Nvfield]*3)
    rho = scatter(ptcl, pmconf, val=1,
                  mesh=mesh, cell_size=pmconf.cell_size*scale)
    mesh = np.zeros([nbody.Nvfield]*3+[3])
    mom = scatter(ptcl, pmconf, val=pvel,
                  mesh=mesh, cell_size=pmconf.cell_size*scale)

    vel = mom / rho[..., None]
    vel = np.array(vel)

    if interp and np.any(np.isnan(vel)):
        # interpolate nan values using nearest neighbors
        davg = 1
        naninds = np.argwhere(np.all(np.isnan(vel), axis=-1))
        paddedvel = np.pad(vel, ((davg,), (davg,), (davg,), (0,)), mode='wrap')
        for i, j, k in naninds:
            perim = np.array(
                [paddedvel[i:i+2*davg+1, j:j+2*davg+1, k:k+2*davg+1]])
            perim = perim[~np.isnan(perim).all(axis=-1)]
            vel[i, j, k] = np.mean(perim, axis=0)

    return vel
