

import os
import h5py
import argparse
import numpy as np
from mpi4py import MPI
from pypower import CatalogMesh, MeshFFTPower, setup_logging
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy.stats import scott_bin_width
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial.transform import Rotation as R

from .tools import noise_positions, save_group
from ..survey.tools import sky_to_xyz


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


def _center_box(data_pos, randoms_pos, boxpad=1.0):
    """Shifts both data and randoms to be in a box starting at 0."""
    pos_combined = np.vstack([data_pos, randoms_pos])
    pos_min = np.min(pos_combined, axis=0)
    pos_max = np.max(pos_combined, axis=0)
    box_size_dims = (pos_max - pos_min) * boxpad
    new_box_size = np.max(box_size_dims)
    shift = pos_min - (new_box_size - (pos_max - pos_min)) / 2.0
    data_pos_centered = data_pos - shift
    randoms_pos_centered = randoms_pos - shift
    return data_pos_centered, randoms_pos_centered, new_box_size


def preprocess_lightcone_catalogs(
        data_rdz, randoms_rdz,
        noise_radial, noise_transverse, boxpad):
    """Loads, transforms, and prepares data and randoms catalogs."""
    # Convert to comoving coordinates
    data_pos = sky_to_xyz(data_rdz, cosmo)
    randoms_pos = sky_to_xyz(randoms_rdz, cosmo)

    # Add observational noise
    if noise_radial > 0 or noise_transverse > 0:
        print("Applying observational noise...")
        data_pos = noise_positions(
            data_pos, data_rdz[:, 0], data_rdz[:, 1],
            noise_radial, noise_transverse)
        randoms_pos = noise_positions(
            randoms_pos, randoms_rdz[:, 0], randoms_rdz[:, 1],
            noise_radial, noise_transverse)

    # Rotate positions so that the mean line-of-sight is along the x-axis
    mean_los = np.mean(data_pos, axis=0)
    target_los = np.array([1, 0, 0])
    rotation, _ = R.align_vectors([target_los], [mean_los])
    data_pos = rotation.apply(data_pos)
    randoms_pos = rotation.apply(randoms_pos)

    # Center the box and determine the new boxsize
    data_pos, randoms_pos, boxsize = _center_box(
        data_pos, randoms_pos, boxpad=boxpad)

    # Final type casting
    data_pos = data_pos.astype(np.float32)
    randoms_pos = randoms_pos.astype(np.float32)

    if np.any(data_pos < 0) or np.any(data_pos > boxsize):
        raise ValueError(
            "Error! Some data tracers are outside the computed box!")
    if np.any(randoms_pos < 0) or np.any(randoms_pos > boxsize):
        raise ValueError(
            "Error! Some random tracers are outside the computed box!")

    return data_pos, randoms_pos, boxsize


def compute_fkp_weights(z, fsky, cosmology, P0=20000):
    """Computes the FKP weights for a given redshift distribution."""
    nofz = get_nofz(z, fsky, cosmology)
    n = nofz(z)
    weights = 1.0 / (1.0 + n * P0)
    return weights


def _get_fsky(cap):
    if cap == 'simbig':
        fsky = 0.0487
    elif cap == 'sgc':
        fsky = 0.0688
    elif cap == 'ngc':
        fsky = 0.1822
    elif cap == 'mtng':
        fsky = 0.1257
    else:
        raise NotImplementedError(f'Cap {cap} not implemented')
    return fsky


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
    parser.add_argument('--cap', required=True)
    parser.add_argument('--use-fkp', action='store_true')
    parser.add_argument('--high-res', action='store_true')
    parser.add_argument('--resampler', type=str, default='tsc')
    parser.add_argument('--boxpad', type=float, default=1.5)
    parser.add_argument('--noise-radial', type=float, default=0.0)
    parser.add_argument('--noise-transverse', type=float, default=0.0)
    args = parser.parse_args()

    interlacing = 2
    los = 'endpoint'
    position_type = 'xyz'
    ells = (0, 2, 4)

    data_pos, data_weights, randoms_pos, randoms_weights = \
        None, None, None, None

    if rank == 0:
        data_filename = args.data_file
        randoms_filename = args.randoms_file

        if not os.path.isfile(data_filename):
            raise FileNotFoundError(
                f"Data file not found: {data_filename}")
        with h5py.File(data_filename, 'r') as f:
            data_rdz = \
                np.stack([f['ra'][:], f['dec'][:], f['z'][:]], axis=1)

        if not os.path.isfile(randoms_filename):
            raise FileNotFoundError(
                f"Randoms file not found: {randoms_filename}")
        with h5py.File(randoms_filename, 'r') as f:
            randoms_rdz = \
                np.stack([f['ra'][:], f['dec'][:], f['z'][:]], axis=1)

        # TODO: Cache randoms?
        data_pos, randoms_pos, boxsize = preprocess_lightcone_catalogs(
            data_rdz, randoms_rdz,
            args.noise_radial, args.noise_transverse, args.boxpad
        )
        if args.use_fkp:
            fsky = _get_fsky(args.cap)
            data_weights = compute_fkp_weights(
                data_rdz[:, 2], fsky=fsky, cosmology=cosmo, P0=20000)
            randoms_weights = compute_fkp_weights(
                randoms_rdz[:, 2], fsky=fsky, cosmology=cosmo, P0=20000)
        else:
            data_weights = np.ones(len(data_pos), dtype=np.float32)
            randoms_weights = np.ones(len(randoms_pos), dtype=np.float32)

        data_pos, randoms_pos = data_pos.T, randoms_pos.T

    boxsize = comm.bcast(boxsize if rank == 0 else None, root=0)

    cellsize = 1000 / 128  # Voxel Size
    if args.high_res:
        cellsize /= 2.0

    kf = 2 * np.pi / boxsize
    # Ncells = boxsize // cellsize
    # knyq = np.pi * Ncells / boxsize    # not used, kmax is fixed at 0.5
    kedges = np.arange(0, 0.5, kf)

    # --- Step 1: Create the Mesh ---
    # This step handles painting the catalogs onto a 3D grid.
    mesh = CatalogMesh(
        data_positions=data_pos,
        data_weights=data_weights,
        randoms_positions=randoms_pos,
        randoms_weights=randoms_weights,
        boxsize=boxsize,
        mpicomm=comm,
        mpiroot=0,
        resampler=args.resampler,
        interlacing=interlacing,
        cellsize=cellsize,
        position_type=position_type
    )

    # --- Step 2: Calculate Power Spectrum from the Mesh ---
    # This step takes the prepared mesh, FFTs it, and computes the multipoles.
    result = MeshFFTPower(
        mesh,
        edges=kedges,
        ells=ells,
        los=los
    )

    if rank != 0:
        return

    # Store data
    poles = result.poles
    k = poles.k
    Pk0 = poles(ell=0, complex=False)
    Pk2 = poles(ell=2, complex=False)
    Pk4 = poles(ell=4, complex=False)
    Pk = np.stack([Pk0, Pk2, Pk4], axis=-1)
    out_data = {
        'Pk_k3D': k,
        'Pk': Pk,
    }

    # TODO: add Bk

    # Save metadata
    out_attrs = {}
    out_attrs['nbar'] = len(data_pos) / boxsize**3
    out_attrs['log10nbar'] = \
        np.log10(len(data_pos)) - 3 * np.log10(boxsize)
    out_attrs['high_res'] = args.high_res and args.resampler == 'ngp'
    out_attrs['noise_radial'] = args.noise_radial
    out_attrs['noise_transverse'] = args.noise_transverse

    # Save n(z)
    zbins = np.linspace(0.4, 0.7, 101)  # spacing in dz = 0.003
    out_data['nz'], out_data['nz_bins'] = \
        np.histogram(data_rdz[:, -1], bins=zbins)

    save_group(args.output_file, out_data, out_attrs)


if __name__ == '__main__':
    main()
