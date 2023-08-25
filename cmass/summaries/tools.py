from os.path import join as pjoin
import numpy as np
from astropy.stats import scott_bin_width
from scipy.interpolate import InterpolatedUnivariateSpline
import nbodykit.lab as nblab

from ..utils import timing_decorator


def get_nofz(z, fsky, cosmo=None):
    ''' calculate nbar(z) given redshift values and f_sky (sky coverage
    fraction)
    Parameters
    ----------
    z : array like
        array of redshift values 
    fsky : float 
        sky coverage fraction  
    cosmo : cosmology object 
        cosmology to calculate comoving volume of redshift bins 
    Returns
    -------
    number density at input redshifts: nbar(z) 
    Notes
    -----
    * based on nbdoykit implementation 
    '''
    # calculate nbar(z) for each galaxy
    _, edges = scott_bin_width(z, return_bins=True)

    dig = np.searchsorted(edges, z, "right")
    N = np.bincount(dig, minlength=len(edges)+1)[1:-1]

    R_hi = cosmo.comoving_distance(edges[1:])  # Mpc/h
    R_lo = cosmo.comoving_distance(edges[:-1])  # Mpc/h

    dV = (4./3.) * np.pi * (R_hi**3 - R_lo**3) * fsky

    nofz = InterpolatedUnivariateSpline(
        0.5*(edges[1:] + edges[:-1]), N/dV, ext='const')

    return nofz


@timing_decorator
def load_galaxies_obs(source_dir, seed):
    rdz = np.load(pjoin(source_dir, 'obs', f'rdz{seed}.npy'))
    return rdz


@timing_decorator
def load_randoms_precomputed():
    savepath = pjoin(
        'data', 'obs', 'random0_DR12v5_CMASS_North_PRECOMPUTED.npy')
    return np.load(savepath)


def sky_to_xyz(rdz, cosmo):
    return nblab.transform.SkyToCartesian(*rdz.T, cosmo)
