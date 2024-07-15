import os
from os.path import join as pjoin
import logging
import numpy as np
import h5py
from ..utils import load_params, timing_decorator, get_particle_mass
import warnings
import MAS_library as MASL
from omegaconf import open_dict

# Optional imports
try:
    import camb
except ImportError:
    camb = None
try:
    import classy
    from classy import Class, CosmoComputationError
except ImportError:
    classy = None
try:
    import symbolic_pofk
    import symbolic_pofk.linear
except ImportError:
    symbolic_pofk = None


def parse_nbody_config(cfg):
    with open_dict(cfg):
        nbody = cfg.nbody
        nbody.ai = 1 / (1 + nbody.zi)  # initial scale factor
        nbody.af = 1 / (1 + nbody.zf)  # final scale factor
        nbody.quijote = nbody.matchIC == 2  # whether to match ICs to Quijote
        nbody.matchIC = nbody.matchIC > 0  # whether to match ICs to file

        # default asave
        if not (hasattr(nbody, 'snapshot_mode') and nbody.snapshot_mode):
            nbody.asave = [nbody.af]

        # load cosmology
        nbody.cosmo = load_params(nbody.lhid, cfg.meta.cosmofile)

    if cfg.nbody.quijote:
        logging.info('Matching ICs to Quijote')
        assert cfg.nbody.L == 1000  # enforce same size of quijote

    return cfg


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


def get_ICs(cfg):
    nbody = cfg.nbody
    N = nbody.N*nbody.supersampling
    if nbody.matchIC:
        path_to_ic = f'wn/N{N}/wn_{nbody.lhid}.dat'
        if nbody.quijote:
            path_to_ic = pjoin(cfg.meta.wdir, 'quijote', path_to_ic)
        else:
            path_to_ic = pjoin(cfg.meta.wdir, path_to_ic)
        return load_white_noise(path_to_ic, N, quijote=nbody.quijote)
    else:
        return gen_white_noise(N, seed=nbody.lhid)


@timing_decorator
def save_nbody(savedir, a, rho, fvel, ppos, pvel):
    os.makedirs(savedir, exist_ok=True)
    savefile = pjoin(savedir, 'nbody.h5')

    logging.info(f'Saving to {savefile}...')
    with h5py.File(savefile, 'w') as f:
        key = f'{a:.6f}'
        group = f.create_group(key)
        group.create_dataset('rho', data=rho)  # density contrast
        group.create_dataset('fvel', data=fvel)  # velocity field [km/s]
        if (ppos is not None) and (pvel is not None):
            # particle comoving positions [Mpc/h]
            group.create_dataset('ppos', data=ppos)
            # particle physical velocities [km/s]
            group.create_dataset('pvel', data=pvel)


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

    return rho, vel


def bin_cube(arr, M):
    """Bins a cube of shape (A, B, C) into subcubes of shape (M,M,M).
    Average over each subcube, producing an output of shape (A//M,B//M,C//M)

    Args:
        arr (np.array): cube to bin
        M (int): bin size
    """
    A, B, C = arr.shape
    assert (A % M == 0) and (B % M == 0) and (C % M == 0), \
        "Array shape must be divisible by M"
    reshaped = arr.reshape(A // M, M, B // M, M, C // M, M)
    transposed = reshaped.transpose(0, 2, 4, 1, 3, 5)
    reshaped = transposed.reshape(A // M, B // M, C // M, -1)
    return reshaped.mean(axis=-1)


# power spectrum stuff (for pinnochio)

def get_camb_pk(k, omega_m, omega_b, h, n_s, sigma8, z=0.):
    if camb is None:
        raise ImportError(
            "camb transfer function requested, but camb not installed. "
            "See ltu-cmass installation instructions."
        )

    pars = camb.CAMBparams(DoLensing=False)
    pars.set_cosmology(
        H0=h*100,
        ombh2=omega_b * h ** 2,
        omch2=(omega_m - omega_b) * h ** 2,
        mnu=0.0,
        omk=0,
    )
    As_fid = 2.0e-9
    pars.InitPower.set_params(As=As_fid, ns=n_s, r=0)
    pars.set_matter_power(redshifts=[z], kmax=k[-1])
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    sigma8_camb = results.get_sigma8()[0]
    As_new = (sigma8 / sigma8_camb) ** 2 * As_fid
    pars.InitPower.set_params(As=As_new, ns=n_s, r=0)
    results = camb.get_results(pars)
    _, _, pk_camb = results.get_matter_power_spectrum(
        minkh=k.min(), maxkh=k.max(), npoints=len(k))
    pk_camb = pk_camb[0, :]

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
    if classy is None:
        raise ImportError(
            "CLASS computation requested, but class not installed. "
            "See ltu-cmass installation instructions."
        )
    cosmo = Class()
    cosmo.set(class_params)
    try:
        cosmo.compute()
        isNormal = True
    except CosmoComputationError as e:
        if "DeltaNeff < deltaN[0]" in str(e):
            # set YHe to 0.25. Value is from https://arxiv.org/abs/1503.08146
            # and Plank 2018(Section 7.6.1) https://arxiv.org/abs/1807.06209
            warnings.warn(
                "Adjust YHe to 0.25 due to CLASS CosmoComputationError "
                f"for cosmology {class_params}.")
            class_params['YHe'] = 0.25
            cosmo.set(class_params)
            cosmo.compute()
            isNormal = False
        else:
            raise e
    return cosmo, isNormal


def get_class_pk(k, omega_m, omega_b, h, n_s, sigma8):
    if classy is None:
        raise ImportError(
            "CLASS transfer function requested, but class not installed. "
            "See ltu-cmass installation instructions."
        )

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
    if symbolic_pofk is None:
        raise ImportError(
            "syren transfer function requested, but syren not installed. "
            "See ltu-cmass installation instructions."
        )
    return symbolic_pofk.linear.plin_emulated(
        k, sigma8, omega_m, omega_b, h, n_s,
        emulator='fiducial', extrapolate=True
    )
