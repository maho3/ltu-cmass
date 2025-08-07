import numpy as np
import h5py
import argparse
import os
from mpi4py import MPI
from pypower import CatalogFFTPower, setup_logging
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import scott_bin_width
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial.transform import Rotation as R

# --- Utility Functions ---


def sky_to_xyz(rdz, cosmology):
    """Converts sky coordinates (ra, dec, z) to Cartesian coordinates."""
    ra, dec, z = np.asarray(rdz).T
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                      distance=cosmology.comoving_distance(z))
    pos = coords.cartesian.xyz.to_value(u.Mpc) * cosmology.h
    return pos.T


def sky_to_unit_vectors(ra, dec):
    """Converts RA and Dec to Cartesian unit vectors."""
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    r_hat = np.array([
        np.cos(dec_rad) * np.cos(ra_rad),
        np.cos(dec_rad) * np.sin(ra_rad),
        np.sin(dec_rad)
    ]).T
    e_phi = np.array([-np.sin(ra_rad), np.cos(ra_rad), np.zeros_like(ra_rad)]).T
    e_theta = np.array([
        -np.sin(dec_rad) * np.cos(ra_rad),
        -np.sin(dec_rad) * np.sin(ra_rad),
        np.cos(dec_rad)
    ]).T
    return r_hat, e_phi, e_theta


def get_nofz(z, fsky, cosmo):
    """
    Calculates n(z) given redshift values and sky coverage fraction.
    This function estimates the comoving number density of objects as a
    function of redshift.
    """
    # Use Scott's rule to determine optimal bin width for the histogram
    _, edges = scott_bin_width(z, return_bins=True)

    # Count number of objects in each redshift bin
    dig = np.searchsorted(edges, z, "right")
    N = np.bincount(dig, minlength=len(edges)+1)[1:-1]

    # Calculate comoving volume of each spherical shell corresponding to a bin
    R_hi = cosmo.comoving_distance(edges[1:]).to_value(u.Mpc) * cosmo.h
    R_lo = cosmo.comoving_distance(edges[:-1]).to_value(u.Mpc) * cosmo.h
    dV = (4./3.) * np.pi * (R_hi**3 - R_lo**3) * fsky

    # Avoid division by zero for empty bins
    dV[dV == 0] = np.inf

    # Create a spline interpolator for n(z)
    nofz = InterpolatedUnivariateSpline(
        0.5*(edges[1:] + edges[:-1]), N/dV, ext='const')
    return nofz


def _center_box(pos_data, pos_randoms, boxpad=1.0):
    """Shifts both data and randoms to be in a box starting at 0."""
    pos_combined = np.vstack([pos_data, pos_randoms])
    pos_min = np.min(pos_combined, axis=0)
    pos_max = np.max(pos_combined, axis=0)
    box_size_dims = (pos_max - pos_min) * boxpad
    new_box_size = np.max(box_size_dims)
    shift = pos_min - (new_box_size - (pos_max - pos_min)) / 2.0
    pos_data_centered = pos_data - shift
    pos_randoms_centered = pos_randoms - shift
    return pos_data_centered, pos_randoms_centered, new_box_size


def _noise_positions(pos, noise_radial, noise_transverse,
                     r_hat, e_phi, e_theta):
    """Applies observational noise to positions."""
    noise = np.random.randn(*pos.shape)
    pos += r_hat * noise[:, 0, None] * noise_radial
    pos += e_phi * noise[:, 1, None] * noise_transverse
    pos += e_theta * noise[:, 2, None] * noise_transverse
    return pos


def preprocess_lightcone_catalogs(
        data_ra, data_dec, data_z,
        randoms_ra, randoms_dec, randoms_z,
        noise_radial, noise_transverse, boxpad):
    """Loads, transforms, and prepares data and randoms catalogs."""
    # Convert to comoving coordinates
    pos_data = sky_to_xyz(
        np.vstack([data_ra, data_dec, data_z]).T, cosmo)
    pos_randoms = sky_to_xyz(
        np.vstack([randoms_ra, randoms_dec, randoms_z]).T, cosmo)

    # Add observational noise
    if noise_radial > 0 or noise_transverse > 0:
        print("Applying observational noise...")
        r_hat, e_phi, e_theta = sky_to_unit_vectors(data_ra, data_dec)
        pos_data = _noise_positions(
            pos_data, noise_radial, noise_transverse, r_hat, e_phi, e_theta)
        pos_randoms = _noise_positions(
            pos_randoms, noise_radial, noise_transverse, r_hat, e_phi, e_theta)

    # Rotate positions so that the mean line-of-sight is along the x-axis
    print("Rotating box...")
    mean_los = np.mean(pos_data, axis=0)
    target_los = np.array([1, 0, 0])
    rotation, _ = R.align_vectors([target_los], [mean_los])
    pos_data = rotation.apply(pos_data)
    pos_randoms = rotation.apply(pos_randoms)

    # Center the box and determine the new boxsize
    print("Centering box...")
    pos_data, pos_randoms, boxsize = _center_box(
        pos_data, pos_randoms, boxpad=boxpad)

    # Final type casting
    pos_data = pos_data.astype(np.float32)
    pos_randoms = pos_randoms.astype(np.float32)

    print(f"Final box size is {boxsize:.2f} Mpc/h")
    if np.any(pos_data < 0) or np.any(pos_data > boxsize):
        raise ValueError(
            "Error! Some data tracers are outside the computed box!")
    if np.any(pos_randoms < 0) or np.any(pos_randoms > boxsize):
        raise ValueError(
            "Error! Some random tracers are outside the computed box!")

    return pos_data, pos_randoms, boxsize


def compute_fkp_weights(z, fsky, cosmology, P0=20000):
    """Computes the FKP weights for a given redshift distribution."""
    nofz = get_nofz(z, fsky, cosmology)
    n = nofz(z)
    weights = 1.0 / (1.0 + n * P0)
    return weights


def main():
    """Main execution function."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run CatalogFFTPower on a lightcone catalog.")
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--randoms-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--use-fkp', action='store_true')
    parser.add_argument('--resampler', type=str, default='tsc')
    parser.add_argument('--boxpad', type=float, default=1.5)
    parser.add_argument('--noise-radial', type=float, default=0.0)
    parser.add_argument('--noise-transverse', type=float, default=0.0)
    args = parser.parse_args()

    interlacing = 2
    los = 'endpoint'
    position_type = 'xyz'
    ells = (0, 2, 4)
    kmin, kmax, dk = 0.0, 0.5, 0.00314
    kedges = np.arange(kmin, kmax, dk)

    pos_data, weights_data, pos_randoms, weights_randoms, boxsize = None, None, None, None, None

    if rank == 0:
        data_filename = args.data_file
        randoms_filename = args.randoms_file

        if not os.path.isfile(data_filename):
            raise FileNotFoundError(
                f"Data file not found: {data_filename}")
        with h5py.File(data_filename, 'r') as f:
            data_rdz = \
                np.concatenate([f['ra'][:], f['dec'][:], f['z'][:]], axis=1)

        if not os.path.isfile(randoms_filename):
            raise FileNotFoundError(
                f"Randoms file not found: {randoms_filename}")
        with h5py.File(randoms_filename, 'r') as f:
            randoms_rdz = \
                np.concatenate([f['ra'][:], f['dec'][:], f['z'][:]], axis=1)

        pos_data, pos_randoms, boxsize = preprocess_lightcone_catalogs(
            data_rdz, randoms_rdz,
            args.noise_radial, args.noise_transverse, args.boxpad
        )
        if args.use_fkp:
            weights_data = compute_fkp_weights(
                data_rdz[:, 2], fsky=0.25, cosmology=cosmo, P0=20000)
            weights_randoms = compute_fkp_weights(
                randoms_rdz[:, 2], fsky=0.25, cosmology=cosmo, P0=20000)
        else:
            weights_data = np.ones(len(pos_data), dtype=np.float32)
            weights_randoms = np.ones(len(pos_randoms), dtype=np.float32)

    result = CatalogFFTPower(
        data_positions1=pos_data,
        data_weights1=weights_data,
        randoms_positions1=pos_randoms,
        randoms_weights1=weights_randoms,
        boxsize=boxsize,
        edges=kedges,
        mpicomm=comm,
        mpiroot=0,
        resampler=args.resampler,
        interlacing=interlacing,
        cellsize=args.cellsize,
        boxpad=args.boxpad,
        los=los,
        position_type=position_type,
        ells=ells
    )

    if rank == 0:
        poles = result.poles
        k = poles.k
        Pk0 = poles(ell=0, complex=False)
        Pk2 = poles(ell=2, complex=False)
        Pk4 = poles(ell=4, complex=False)

        np.savez(args.output_file, k=k, Pk0=Pk0,
                 Pk2=Pk2, Pk4=Pk4, boxsize=boxsize)
        print(f"Power spectrum multipoles saved to {args.output_file}")


if __name__ == '__main__':
    main()
