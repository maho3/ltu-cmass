"""
Functions for implementing BOSS-cmass forward models.
Many functions are from or inspired by: https://github.com/changhoonhahn/simbig/blob/main/src/simbig/forwardmodel.py
"""
# imports
import os
from os.path import join as pjoin
import numpy as np
import pymangle
import pandas as pd
from copy import deepcopy
import logging
import h5py

from astropy.io import fits
from astropy.coordinates import search_around_sky
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from ..utils import timing_decorator, cosmo_to_astropy


# cosmo functions


def xyz_to_sky(pos, vel=None, cosmo=None):
    """Converts cartesian coordinates to sky coordinates (ra, dec, z).
    Inspired by nbodykit.transform.CartesianToSky.
    """
    if vel is None:
        vel = np.zeros_like(pos)  # no peculiar velocity
    if cosmo is None:
        raise ValueError('cosmo must be provided.')

    pos, vel = map(deepcopy, [pos, vel])  # avoid modifying input
    pos, vel = map(np.atleast_2d, [pos, vel])  # ensure 2D arrays
    cosmo = cosmo_to_astropy(cosmo)

    pos /= cosmo.h  # convert from Mpc/h to Mpc
    pos *= u.Mpc  # label as Mpc
    vel *= u.km / u.s  # label as km/s

    # get ra, dec
    coord_cart = SkyCoord(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        representation_type='cartesian')
    coord_sphe = coord_cart.represent_as('spherical')
    ra = coord_sphe.lon.to(u.deg)
    dec = coord_sphe.lat.to(u.deg)

    # get redshift
    R = np.linalg.norm(pos, axis=-1)

    def z_from_comoving_distance(d):
        zgrid = np.logspace(-8, 1.5, 2048)
        zgrid = np.concatenate([[0.], zgrid])
        dgrid = cosmo.comoving_distance(zgrid)
        return interp1d(dgrid, zgrid)(d)

    # Convert comoving distance to redshift
    z = z_from_comoving_distance(R)

    vpec = (pos*vel).sum(axis=-1) / R
    z += vpec / c.to(u.km/u.s)*(1+z)

    return np.array([ra, dec, z]).T


def sky_to_xyz(rdz, cosmo):
    """Converts sky coordinates (ra, dec, z) to cartesian coordinates."""
    rdz = np.asarray(rdz)
    cosmo = cosmo_to_astropy(cosmo)

    ra, dec, z = rdz.T
    pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                   distance=cosmo.comoving_distance(z))
    pos = pos.cartesian.xyz
    pos *= cosmo.h  # convert from Mpc to Mpc/h

    return pos.value.T

# Geometry functions


def rotate_to_z(xyz, cosmo):
    """Returns a Rotation object which rotates a sightline (comoving position)
    to the z-axis.

    Args:
        xyz (array): (3,) array of center position of sky footprint.
            In Mpc/h.
        cosmo (array): Cosmological parameters
            [Omega_m, Omega_b, h, n_s, sigma8].

    Returns:
        rot (Rotation): Rotation object which rotates the sightline to z-axis.
        irot (Rotation): Inverse rotation object.
    """

    # calculate direction vector
    mvec = xyz / np.linalg.norm(xyz)

    # use a nearby point in +RA to affix x-axis
    rdz = xyz_to_sky(xyz, np.zeros(3), cosmo)[0]
    rdz += [0.001, 0, 0]  # add 0.001
    xyz1 = sky_to_xyz(rdz, cosmo)

    # calculate x-axis vector (to be rotated later to x-axis)
    xvec = xyz1 - xyz
    xvec /= np.linalg.norm(xvec)

    # rotate xyz to z
    rotz_axis = np.cross(mvec, [0, 0, 1])
    rotz_axis /= np.linalg.norm(rotz_axis)
    rotz_angle = np.arccos(np.dot(mvec, [0, 0, 1]))
    rotz = R.from_rotvec(rotz_angle*rotz_axis)

    # rotate xvec to x-axis
    xvec = rotz.apply(xvec)
    rotx_angle = -np.arctan2(xvec[1], xvec[0])
    rotx = R.from_rotvec(rotx_angle*np.array([0, 0, 1]))

    # combine rotations and measure inverse
    rot = rotx*rotz
    irot = rot.inv()

    return rot, irot


def random_rotate_translate(xyz, L, vel=None, seed=0):
    """Randomly rotate and translate a cube of points.

    Rotations are fixed to [0, 90, 180, 270] degrees on each axis,
    to satisfy periodic boundary conditions.

    Args:
    - xyz (np.ndarray): (N, 3) array of positions in the cube.
    - L (float): side length of the cube.
    - vel (np.ndarray, optional): (N, 3) array of velocities. 
    - seed (int): random seed for reproducibility. If 0, no transformation
        is applied.
    """

    assert np.all((xyz >= 0) & (xyz <= L)), "xyz must be in [0, L]"
    xyz, vel = map(deepcopy, [xyz, vel])

    if seed == 0:  # no transformation
        offset = np.zeros(3)
        rotation = R.identity()
    else:
        np.random.seed(seed)
        offset = np.random.rand(3)*L
        rotation = R.from_euler(
            'xyz', np.random.choice([0, 90, 180, 270], 3),
            degrees=True)

    # Rotate
    xyz -= L/2
    xyz = rotation.apply(xyz)
    xyz += L/2
    vel = rotation.apply(vel) if vel is not None else None

    # Translate
    xyz += offset
    xyz %= L

    return xyz, vel


# mask functions

def BOSS_angular(ra, dec, wdir='./data'):
    ''' Given RA and Dec, check whether the galaxies are within the angular
    mask of BOSS
    '''
    f_poly = os.path.join(wdir, 'obs', 'mask_DR12v5_CMASS_North.ply')
    mask = pymangle.Mangle(f_poly)

    w = mask.weight(ra, dec)
    mask = (w > np.random.rand(len(ra)))  # conform to angular completeness
    mask &= (w > 0.7)  # mask completeness < 0.7 (See arxiv:1509.06404)
    return mask


def BOSS_veto(ra, dec, verbose=False, wdir='./data'):
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

    for fveto in fvetos:
        if verbose:
            print(fveto)
        veto = pymangle.Mangle(os.path.join(wdir, 'obs', fveto))
        w_veto = veto.weight(ra, dec)
        in_veto = in_veto | (w_veto > 0.)
    return in_veto


def BOSS_redshift(z):
    zmin, zmax = 0.4, 0.7
    mask = (zmin < z) & (z < zmax)
    return np.array(mask)


def BOSS_fiber(ra, dec, sep=0.01722, mode=1):
    c = SkyCoord(ra=ra, dec=dec, unit=u.degree)
    seplimit = sep*u.degree
    idx1, idx2, _, _ = search_around_sky(c, c, seplimit)

    if mode == 1:
        iddrop = idx1[idx1 != idx2]
    elif mode == 2:
        iddrop = np.array(
            list(set(idx1[idx1 != idx2]).union(idx2[idx1 != idx2])),
            dtype=int)
    else:
        raise ValueError(f'Fiber collision type {mode} is not valid.')

    mask = np.ones(len(ra), dtype=bool)
    mask[iddrop] = False
    return mask


def BOSS_area(wdir='./data'):
    """Returns the area of the BOSS survey. Deprecated with addition of pypower."""
    f_poly = os.path.join(wdir, 'obs/mask_DR12v5_CMASSLOWZ_North.ply')
    boss_poly = pymangle.Mangle(f_poly)
    area = np.sum(boss_poly.areas * boss_poly.weights)  # deg^2
    return area


@timing_decorator
def gen_randoms(wdir='./data'):
    fname = pjoin(wdir, 'obs', 'random0_DR12v5_CMASS_North.fits')
    fields = ['RA', 'DEC', 'Z']
    with fits.open(fname) as hdul:
        randoms = np.array([hdul[1].data[x] for x in fields]).T
        randoms = pd.DataFrame(randoms, columns=fields)

    n_z = np.load(pjoin(wdir, 'obs', 'n-z_DR12v5_CMASS_North.npy'),
                  allow_pickle=True).item()
    be, hobs = n_z['be'], n_z['h']
    cutoffs = np.cumsum(hobs) / np.sum(hobs)
    w = np.diff(be[:2])[0]

    prng = np.random.uniform(size=len(randoms))
    randoms['Z'] = be[:-1][cutoffs.searchsorted(prng)]
    randoms['Z'] += w * np.random.uniform(size=len(randoms))

    # further selection functions
    mask = BOSS_angular(randoms['RA'], randoms['DEC'], wdir=wdir)
    randoms = randoms[mask]
    mask = BOSS_redshift(randoms['Z'])
    randoms = randoms[mask]
    mask = (~BOSS_veto(randoms['RA'], randoms['DEC'], verbose=True, wdir=wdir))
    randoms = randoms[mask]

    return randoms.values


def load_galaxies(source_dir, a, seed):
    filepath = pjoin(source_dir, 'galaxies', f'hod{seed:03}.h5')
    with h5py.File(filepath, 'r') as f:
        key = f'{a:.6f}'
        pos = f[key]['pos'][...]
        vel = f[key]['vel'][...]
        hostid = f[key]['hostid'][...]
    return pos, vel, hostid


def save_lightcone(outdir, ra, dec, z, galsnap=None, galidx=None,
                   weight=None, hod_seed=0, suffix=''):
    outfile = pjoin(outdir, f'hod{hod_seed:03}{suffix}.h5')
    logging.info(f'Saving lightcone to {outfile}')
    with h5py.File(outfile, 'w') as f:
        f.create_dataset('ra', data=ra)                # Right ascension [deg]
        f.create_dataset('dec', data=dec)              # Declination [deg]
        f.create_dataset('z', data=z)                  # Redshift
        if galsnap is not None:
            f.create_dataset('galsnap', data=galsnap)  # Snapshot index
        if galidx is not None:
            f.create_dataset('galidx', data=galidx)    # Galaxy index
        if weight is not None:
            f.create_dataset('weight', data=weight)    # Weight
