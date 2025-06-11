"""
Functions for implementing BOSS-cmass forward models.
Many functions are from or inspired by: https://github.com/changhoonhahn/simbig/blob/main/src/simbig/forwardmodel.py
"""
# imports
import os
from os.path import join
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

from ..utils import timing_decorator, cosmo_to_astropy, save_configuration_h5


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


def sky_to_unit_vectors(ra_deg, dec_deg):
    """Converts sky coordinates (ra_deg, dec_deg) to unit vectors in cartesian coordinates.
    
    Args:
        ra_deg (array): Right ascension in degrees.
        dec_deg (array): Declination in degrees.

    Returns:
        r_hat (array): Unit radial vector in cartesian coordinates.
        e_phi (array): Unit vector along increasing RA (constant Dec).
        e_theta (array): Unit vector along increasing Dec (constant RA).
    """

    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    # Unit radial vector r_hat
    r_hat = np.stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ], axis=-1)  # shape (N, 3)

    # Along increasing RA (constant Dec) — e_phi
    e_phi = np.stack([
        -np.sin(ra),
        np.cos(ra),
        np.zeros_like(ra)
    ], axis=-1)  # shape (N, 3)

    # Along increasing Dec (constant RA) — e_theta
    e_theta = np.stack([
        -np.sin(dec) * np.cos(ra),
        -np.sin(dec) * np.sin(ra),
        np.cos(dec)
    ], axis=-1)  # shape (N, 3)

    return r_hat, e_phi, e_theta

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

def BOSS_angular(ra, dec, wdir='./data', is_North=True):
    ''' Given RA and Dec, check whether the galaxies are within the angular
    mask of BOSS
    '''
    if is_North:
        f_poly = os.path.join(wdir, 'obs', 'mask_DR12v5_CMASS_North.ply')
    else:
        f_poly = os.path.join(wdir, 'obs', 'mask_DR12v5_CMASS_South.ply')
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
    """Fiber collision mask for BOSS galaxies as described in arXiv:2211.00723"""

    c = SkyCoord(ra=ra, dec=dec, unit=u.degree)
    seplimit = sep*u.degree
    m1, m2, _, _ = search_around_sky(c, c, seplimit)

    # remove self-matches
    notitself = m1 != m2
    m1 = m1[notitself]
    m2 = m2[notitself]

    if mode == 1:

        # pairs are double counted by search_around_sky. This selects the unique pairs
        _, ipair = np.unique(
            np.min(np.array([m1, m2]), axis=0), return_index=True)

        # only ~60% of galaxies within the angular scale are fiber collided
        # since 40% are in overlapping regions with substantially lower
        # fiber collision rates
        ncollid = int(0.6 * len(ipair))

        icollid = np.random.choice(ipair, size=ncollid, replace=False)

        mask = np.ones(len(ra)).astype(bool)
        mask[m1[icollid[:int(0.5*ncollid)]]] = False
        mask[m2[icollid[int(0.5*ncollid):]]] = False

    elif mode == 2:

        mask = np.ones(len(ra)).astype(bool)
        mask[m1] = False

    else:
        raise ValueError(f'Fiber collision type {mode} is not valid.')

    return mask


def BOSS_area(wdir='./data'):
    """Returns the area of the BOSS survey. Deprecated with addition of pypower."""
    f_poly = os.path.join(wdir, 'obs/mask_DR12v5_CMASSLOWZ_North.ply')
    boss_poly = pymangle.Mangle(f_poly)
    area = np.sum(boss_poly.areas * boss_poly.weights)  # deg^2
    return area


@timing_decorator
def gen_randoms(wdir='./data'):
    fname = join(wdir, 'obs', 'random0_DR12v5_CMASS_North.fits')
    fields = ['RA', 'DEC', 'Z']
    with fits.open(fname) as hdul:
        randoms = np.array([hdul[1].data[x] for x in fields]).T
        randoms = pd.DataFrame(randoms, columns=fields)

    n_z = np.load(join(wdir, 'obs', 'n-z_DR12v5_CMASS_North.npy'),
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
    filepath = join(source_dir, 'galaxies', f'hod{seed:05}.h5')
    with h5py.File(filepath, 'r') as f:
        key = f'{a:.6f}'
        if key not in f:
            raise ValueError(
                f'Snapshot a={key} not found in {filepath}. Ensure you are '
                'using the appropriate single-snapshot ngc_selection or the '
                'multi-snapshot ngc_lightcone.')
        pos = f[key]['pos'][...]  # comoving positions [Mpc/h]
        vel = f[key]['vel'][...]  # physical velocities [km/s]
        if 'hostid' in f[key]:
            hostid = f[key]['hostid'][...]
        else:  # not needed for simple survey selection
            hostid = None
    return pos, vel, hostid


def save_lightcone(outdir, ra, dec, z, galsnap=None, galidx=None,
                   weight=None, hod_seed=0, aug_seed=0, suffix='',
                   config=None, **kwargs):
    outfile = join(outdir, f'hod{hod_seed:05}_aug{aug_seed:05}{suffix}.h5')
    logging.info(f'Saving lightcone to {outfile}')
    with h5py.File(outfile, 'w') as f:
        if config is not None:
            save_configuration_h5(f, config, save_HOD=True)
        for k, v in kwargs.items():
            f.attrs[k] = v

        f.create_dataset('ra', data=ra)                # Right ascension [deg]
        f.create_dataset('dec', data=dec)              # Declination [deg]
        f.create_dataset('z', data=z)                  # Redshift
        if galsnap is not None:
            f.create_dataset('galsnap', data=galsnap)  # Snapshot index
        if galidx is not None:
            f.create_dataset('galidx', data=galidx)    # Galaxy index
        if weight is not None:
            f.create_dataset('weight', data=weight)    # Weight


def load_lightcone(indir, hod_seed=0, aug_seed=0, suffix=''):
    """Load lightcone data from file."""
    filename = f'hod{hod_seed:05}_aug{aug_seed:05}{suffix}.h5'
    filepath = join(indir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'File {filepath} does not exist.')
    with h5py.File(filepath, 'r') as f:
        ra = f['ra'][...]                # Right ascension [deg]
        dec = f['dec'][...]              # Declination [deg]
        z = f['z'][...]                  # Redshift
        galsnap = f['galsnap'][...]  # Snapshot index
        galidx = f['galidx'][...]    # Galaxy index
        weight = f['weight'][...] if 'weight' in f else None  # Weight
        attrs = dict(f.attrs)
    return ra, dec, z, galsnap, galidx, weight, attrs
