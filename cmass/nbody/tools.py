import os
from os.path import join as pjoin
import logging
import numpy as np
from ..utils import timing_decorator
from pmwd import Configuration, Particles, scatter


@timing_decorator
def gen_white_noise(N):
    """Generate ICs in Fourier space."""
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
def save_nbody(savedir, rho, fvel, pos, vel, save_particles=True):
    os.makedirs(savedir, exist_ok=True)
    np.save(pjoin(savedir, 'rho.npy'), rho)  # density contrast
    np.save(pjoin(savedir, 'fvel.npy'), fvel)  # velocity field [km/s]
    if save_particles:
        np.save(pjoin(savedir, 'ppos.npy'), pos)  # particle positions [Mpc/h]
        np.save(pjoin(savedir, 'pvel.npy'), vel)  # particle velocities [km/s]
    logging.info(f'Saved to {savedir}.')


@timing_decorator
def vfield_CIC(ppos, pvel, cfg, interp=True):
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
