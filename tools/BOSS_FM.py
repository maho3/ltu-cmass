"""
Functions for implementing BOSS-cmass forward models.
Most functions are from or inspired by: https://github.com/changhoonhahn/simbig/blob/main/src/simbig/forwardmodel.py
"""
# imports
import os
import numpy as np
import pymangle
from astropy.stats import scott_bin_width
from scipy.interpolate import InterpolatedUnivariateSpline

# mask functions


def BOSS_angular(ra, dec):
    ''' Given RA and Dec, check whether the galaxies are within the angular
    mask of BOSS
    '''
    f_poly = os.path.join('data', 'obs', 'mask_DR12v5_CMASS_North.ply')
    mask = pymangle.Mangle(f_poly)

    w = mask.weight(ra, dec)
    inpoly = (w > 0.)
    return inpoly


def BOSS_veto(ra, dec, verbose=False):
    ''' given RA and Dec, find the objects that fall within one of the veto 
    masks of BOSS. At the moment it checks through the veto masks one by one.  
    '''
    in_veto = np.zeros(len(ra)).astype(bool)
    fvetos = [
        'badfield_mask_postprocess_pixs8.ply',
        'badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply',
        'allsky_bright_star_mask_pix.ply',
        'bright_object_mask_rykoff_pix.ply',
        'centerpost_mask_dr12.ply',
        'collision_priority_mask_dr12.ply']

    veto_dir = 'data'
    for fveto in fvetos:
        if verbose:
            print(fveto)
        veto = pymangle.Mangle(os.path.join(veto_dir, 'obs', fveto))
        w_veto = veto.weight(ra, dec)
        in_veto = in_veto | (w_veto > 0.)
    return in_veto


def BOSS_redshift(z):
    zmin, zmax = 0.4, 0.7
    mask = (zmin < z) & (z < zmax)
    return np.array(mask)


def BOSS_radial(z, sample='lowz-south', seed=0):
    ''' Downsample the redshifts to match the BOSS radial selection function.
    This assumes that the sample consists of the same type of galaxies (i.e. 
    constant HOD), but selection effects randomly remove some of them 
    Notes
    -----
    * nbar file from https://data.sdss.org/sas/bosswork/boss/lss/DR12v5/
    '''
    if sample == 'lowz-south':
        f_nbar = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'dat', 'nbar_DR12v5_LOWZ_South_om0p31_Pfkp10000.dat')
        zmin, zmax = 0.2, 0.37
    else:
        raise NotImplementedError

    # zcen,zlow,zhigh,nbar,wfkp,shell_vol,total weighted gals
    zcen, zlow, zhigh, nbar, wfkp, shell_vol, tot_gal = np.loadtxt(f_nbar,
                                                                   skiprows=2, unpack=True)
    zedges = np.concatenate([zlow, [zhigh[-1]]])

    ngal_z, _ = np.histogram(np.array(z), bins=zedges)

    # fraction to downsample
    # fdown_z = tot_gal/ngal_z.astype(float)

    # impose redshift limit
    zlim = (z > zmin) & (z < zmax)

    # i_z = np.digitize(z, zedges)
    # downsample = (np.random.rand(len(z)) < fdown_z[i_z])

    return zlim  # & downsample


def BOSS_area():
    f_poly = os.path.join('data', 'obs/mask_DR12v5_CMASSLOWZ_North.ply')
    boss_poly = pymangle.Mangle(f_poly)
    area = np.sum(boss_poly.areas * boss_poly.weights)  # deg^2
    return area


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


# hod functions

def thetahod_literature(paper):
    ''' best-fit HOD parameters from the literature. 

    Currently, HOD values from the following papers are available:
    * 'parejko2013_lowz'
    * 'manera2015_lowz_ngc'
    * 'manera2015_lowz_sgc'
    * 'redi2014_cmass'
    '''
    if paper == 'parejko2013_lowz':
        # lowz catalog from Parejko+2013 Table 3. Note that the
        # parameterization is slightly different so the numbers need to
        # be converted.
        p_hod = {
            'logMmin': 13.25,
            'sigma_logM': 0.43,  # 0.7 * sqrt(2) * log10(e)
            'logM0': 13.27,  # log10(kappa * Mmin)
            'logM1': 14.18,
            'alpha': 0.94
        }
    elif paper == 'manera2015_lowz_ngc':
        # best-fit HOD of the lowz catalog NGC from Table 2 of Manera et al.(2015)
        p_hod = {
            'logMmin': 13.20,
            'sigma_logM': 0.62,
            'logM0': 13.24,
            'logM1': 14.32,
            'alpha': 0.9
        }
    elif paper == 'manera2015_lowz_sgc':
        # best-fit HOD of the lowz catalog SGC from Table 2 of Manera et al.(2015)
        # Manera+(2015) actually uses a redshift dependent HOD. The HOD that's
        # currently implemented is primarily for the 0.2 < z < 0.35 population,
        # which has nbar~3x10^-4 h^3/Mpc^3
        p_hod = {
            'logMmin': 13.14,
            'sigma_logM': 0.55,
            'logM0': 13.43,
            'logM1': 14.58,
            'alpha': 0.93
        }
    elif paper == 'reid2014_cmass':
        # best-fit HOD from Reid et al. (2014) Table 4
        p_hod = {
            'logMmin': 13.03,
            'sigma_logM': 0.38,
            'logM0': 13.27,
            'logM1': 14.08,
            'alpha': 0.76
        }
    else:
        raise NotImplementedError

    return p_hod
