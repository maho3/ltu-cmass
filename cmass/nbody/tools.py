import os
from os.path import join as pjoin
import logging
import numpy as np
from ..utils import timing_decorator


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
def save_nbody(savedir, rho, pos, vel, save_particles=True):
    os.makedirs(savedir, exist_ok=True)
    np.save(pjoin(savedir, 'rho.npy'), rho)
    if save_particles:
        np.save(pjoin(savedir, 'ppos.npy'), pos)
        np.save(pjoin(savedir, 'pvel.npy'), vel)
    logging.info(f'Saved to {savedir}.')
