import os
from os.path import join as pjoin
import logging
import numpy as np
from ..utils import timing_decorator
from pmwd import Configuration, Particles, scatter
import camb
from classy import Class, CosmoComputationError
import symbolic_pofk.linear

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


def get_camb_pk(k, omega_m, omega_b, h, n_s, sigma8):
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0 = h*100,
                       ombh2 = omega_b * h ** 2,
                       omch2 = (omega_m - omega_b) * h ** 2,
                       mnu = 0.0,
                       omk = 0,
                      )
    As_fid = 2.0e-9
    pars.InitPower.set_params(As=As_fid, ns=n_s, r=0)
    pars.set_matter_power(redshifts=[0.], kmax=k[-1])
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    sigma8_camb = results.get_sigma8()[0]
    As_new = (sigma8 / sigma8_camb) ** 2 * As_fid
    pars.InitPower.set_params(As=As_new, ns=n_s, r=0)
    results = camb.get_results(pars)
    _, _, pk_camb = results.get_matter_power_spectrum(
                            minkh=k.min(), maxkh=k.max(), npoints=len(k))
    pk_camb = pk_camb[0,:]
    
    return pk_camb


def class_compute(class_params):
    '''
    A function to handle CLASS computation and deal with potential errors.

    Args:
        :class_params (dict): Dictionary of CLASS parameters

    Returns:
        :cosmo (CLASS): Instance of the CLASS code
        :isNormal (bool): Whether error occurred in the computation
    '''
    cosmo = Class()
    cosmo.set(class_params)
    try:
        cosmo.compute()
        isNormal=True
    except CosmoComputationError as e:
        if "DeltaNeff < deltaN[0]" in str(e):
            # set YHe to 0.25. Value is from https://arxiv.org/abs/1503.08146 and Plank 2018(Section 7.6.1) https://arxiv.org/abs/1807.06209
            warnings.warn(f"Adjust YHe to 0.25 due to CLASS CosmoComputationError for cosmology {class_params}.")
            class_params['YHe'] = 0.25
            cosmo.set(class_params)
            cosmo.compute()
            isNormal=False
        else:
            raise e
    return cosmo, isNormal


def get_class_pk(k, omega_m, omega_b, h, n_s, sigma8):

    As_fid = 2.0e-9
    class_params = {
        'h': h,
        'omega_b': omega_b * h**2,
        'omega_cdm': (omega_m - omega_b) * h**2,
        'A_s': As_fid,
        'n_s': n_s,
        'output': 'mPk',
        'P_k_max_1/Mpc': k.max() * h,
        'w0_fld': -1.0,
        'wa_fld': 0.0,
        'Omega_Lambda': 0,  # Set to 0 because we're using w0_fld and wa_fld instead
        'z_max_pk': 3.0,  # Max redshift for P(k) output
    }   
    cosmo, isNormal = class_compute(class_params)
    sigma8_class = cosmo.sigma8()
    cosmo.struct_cleanup()
    cosmo.empty()
    As_new = (sigma8 / sigma8_class) ** 2 * As_fid
    class_params['A_s'] = As_new
    
    cosmo, isNormal = class_compute(class_params)
    redshift = 0.0
    plin_class = np.array([cosmo.pk_lin(kk*h, redshift) for kk in k]) * h ** 3
    cosmo.struct_cleanup()
    cosmo.empty()
    
    return plin_class


def get_syren_pk(k, omega_m, omega_b, h, n_s, sigma8):    
    return symbolic_pofk.linear.plin_emulated(k, sigma8, omega_m, omega_b, h, n_s,
        emulator='fiducial', extrapolate=True)
