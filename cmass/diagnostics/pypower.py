
import h5py
import argparse
import numpy as np
from mpi4py import MPI
from pypower import CatalogMesh, MeshFFTPower, setup_logging
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy.stats import scott_bin_width
from scipy.interpolate import InterpolatedUnivariateSpline

from .tools import save_group
from .geometry import SURVEY_GEOMETRIES
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


def preprocess_lightcone_catalogs(
        data_rdz, randoms_rdz):
    """Loads, transforms, and prepares data and randoms catalogs."""
    # Convert to comoving coordinates
    data_pos = sky_to_xyz(data_rdz, cosmo)
    randoms_pos = sky_to_xyz(randoms_rdz, cosmo)

    # Final type casting
    data_pos = data_pos.astype(np.float32)
    randoms_pos = randoms_pos.astype(np.float32)

    return data_pos, randoms_pos


def compute_fkp_weights(z, fsky, cosmology, P0=10000, nofz=None):
    """Computes the FKP weights for a given redshift distribution."""
    if nofz is None:
        nofz = get_nofz(z, fsky, cosmology)
    n = nofz(z)
    weights = 1.0 / (1.0 + n * P0)
    return weights, nofz


def _get_box_geometry(cap):
    geom = SURVEY_GEOMETRIES.get(cap)
    fsky = geom['fsky']
    boxsize = geom['boxsize']
    boxcenter = geom['boxcenter']
    rotation = geom['rotation']
    return boxsize, boxcenter, rotation, fsky


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
    parser.add_argument('--noise-radial', type=float, default=0.0)
    parser.add_argument('--noise-transverse', type=float, default=0.0)
    args = parser.parse_args()

    interlacing = 2
    los = 'endpoint'
    position_type = 'xyz'
    ells = (0, 2, 4)
    P0 = 1e4

    data_pos, data_weights, randoms_pos, randoms_weights = \
        None, None, None, None

    data_rdz, randoms_rdz = None, None
    loaded = False
    if rank == 0:
        data_filename = args.data_file
        randoms_filename = args.randoms_file

        try:
            with h5py.File(data_filename, 'r') as f:
                data_rdz = \
                    np.stack([f['ra'][:], f['dec'][:], f['z'][:]], axis=1)
            with h5py.File(randoms_filename, 'r') as f:
                randoms_rdz = \
                    np.stack([f['ra'][:], f['dec'][:], f['z'][:]], axis=1)
        except Exception as e:
            error_msg = str(e)
        loaded = (data_rdz is not None) and (randoms_rdz is not None)

    # Catch and end if files are not loaded
    loaded = comm.bcast(loaded, root=0)
    if not loaded:
        if rank == 0:
            comm.Abort(1)
            raise FileNotFoundError(
                f"Data or randoms file not found: {data_filename} or "
                f"{randoms_filename}\n Exception: {error_msg}")
        return

    # Get box geometry
    boxsize, boxcenter, rotation, fsky = _get_box_geometry(args.cap)

    if rank == 0:
        # TODO: Cache randoms?
        data_pos, randoms_pos = preprocess_lightcone_catalogs(
            data_rdz, randoms_rdz
        )
        if args.use_fkp:
            data_weights, nofz = compute_fkp_weights(
                data_rdz[:, 2], fsky=fsky, cosmology=cosmo, P0=P0)
            randoms_weights, _ = compute_fkp_weights(
                randoms_rdz[:, 2], fsky=fsky, cosmology=cosmo, P0=P0, nofz=nofz)
        else:
            data_weights = np.ones(len(data_pos), dtype=np.float32)
            randoms_weights = np.ones(len(randoms_pos), dtype=np.float32)

        data_pos, randoms_pos = data_pos.T, randoms_pos.T

        # Rotate positions so they lie optimally along cardinal axes
        data_pos = rotation @ data_pos
        randoms_pos = rotation @ randoms_pos

        # Warn if any data_pos or randoms_pos are outside the box
        outside_box = np.sum(
            np.any(np.abs(data_pos.T - boxcenter) > boxsize / 2, axis=1))
        if outside_box > 0:
            comm.Abort(1)
            raise ValueError(
                f"Error: {outside_box} data positions are outside the box.")

    cellsize = 1000 / 128  # Voxel Size
    if args.high_res:
        cellsize /= 2.0

    kf = 2 * np.pi / boxsize
    # Ncells = boxsize // cellsize
    # knyq = np.pi * Ncells / boxsize    # not used, kmax is fixed
    kedges = np.arange(0, 0.41, kf)

    # --- Step 1: Create the Mesh ---
    # This step handles painting the catalogs onto a 3D grid.
    mesh = CatalogMesh(
        data_positions=data_pos,
        data_weights=data_weights,
        randoms_positions=randoms_pos,
        randoms_weights=randoms_weights,
        boxsize=boxsize,
        boxcenter=boxcenter,
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

    # # --- Step 3: Calculate the Bispectrum ---
    # field = mesh.to_mesh(field='fkp', dtype=np.float32, compensate=True)
    # TODO: Update this section to calculate the bispectrum (solve jax versioning)

    # Save metadata
    out_attrs = {}
    Ngalaxies = data_pos.shape[1]
    out_attrs['Ngalaxies'] = Ngalaxies
    out_attrs['boxsize'] = boxsize
    out_attrs['nbar'] = Ngalaxies / boxsize**3
    out_attrs['log10nbar'] = \
        np.log10(Ngalaxies) - 3 * np.log10(boxsize)
    out_attrs['high_res'] = args.high_res and args.resampler == 'tsc'

    # Save n(z)
    zbins = np.linspace(0.4, 0.7, 101)  # spacing in dz = 0.003
    out_data['nz'], out_data['nz_bins'] = \
        np.histogram(data_rdz[:, -1], bins=zbins)

    save_group(args.output_file, out_data, out_attrs)


if __name__ == '__main__':
    main()
