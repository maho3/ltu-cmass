"""This module provides functions to calculate summary statistics."""

import os
import copy
import numpy as np
import Pk_library as PKL
import MAS_library as MASL
import redshift_space_library as RSL
# import BFast  # installation issues on Delta
import PolyBin3D as pb
from ..utils import timing_decorator

# Fixed k-binning for all summary statistics.
# These are shared across all volumes so the data vector has consistent length
# and meaning for SBI, regardless of box size.
#
# K_MIN: set to ~k_F of the largest volume we use (3 Gpc/h → k_F ≈ 0.002),
#   rounded up to 0.01 to exclude the noisiest large-scale modes.
# DK_PK: dk=0.01 gives ~40 Pk bins (k_Nyq ≈ 0.40 is volume-independent),
#   comfortably O(100).
# DK_BK: dk=0.04 gives ~220 Bk triangles — chosen to keep the Bk data vector
#   O(100-200). Finer bins (dk=0.02) give 1540 triangles, which is too large.
# K_MAX_BK: 0.40 h/Mpc matches k_Nyquist for our standard mesh (N=128 per Gpc/h),
#   so we use all available signal without extrapolating.
K_MIN = 0.01    # h/Mpc, first bin edge
DK_PK = 0.01   # h/Mpc, bin width for P(k)  → ~40 bins
DK_BK = 0.04   # h/Mpc, bin width for B(k)  → ~220 triangles
K_MAX_BK = 0.40  # h/Mpc, hard kmax for B(k)


def MA(pos, L, N, MAS='CIC'):
    pos = np.ascontiguousarray(pos)
    pos = copy.deepcopy(pos)
    delta = np.zeros((N, N, N), dtype=np.float32)
    pos %= L
    MASL.MA(pos, delta, BoxSize=L, MAS=MAS)
    delta = delta.astype(np.float64)
    delta /= np.mean(delta)
    delta -= 1
    return delta


def MAz(pos, vel, L, N, cosmo, z, MAS='CIC', axis=0):
    pos, vel = map(np.ascontiguousarray, (pos, vel))
    pos = copy.deepcopy(pos)
    vel = copy.deepcopy(vel)
    RSL.pos_redshift_space(pos, vel, L, cosmo.H(z).value/cosmo.h, z, axis)
    return MA(pos, L, N, MAS)


def calcPk(delta, L, axis=0, MAS='CIC', threads=16):
    Pk = PKL.Pk(delta.astype(np.float32), L, axis, MAS, threads, verbose=False)
    k = Pk.k3D
    Nmodes = Pk.Nmodes3D
    Pk = Pk.Pk
    return k, Pk, Nmodes


def _fixed_pk_kedges(k_nyq):
    """The fixed (K_MIN, DK_PK) output grid, capped at a mesh's k_nyq.

    Same grid `rebin_pk` rebins pylians onto -- built here directly so
    pypower can bin straight onto it in one pass, instead of measuring on
    pypower's own (much finer, k_F-spaced) bins and rebinning after.
    """
    n_bins = (k_nyq - K_MIN) // DK_PK
    return np.append(K_MIN + DK_PK * np.arange(n_bins + 1),
                     k_nyq)  # last bin shorter


def calcPk_pypower(pos, L, N, axis=0, resampler='TSC', interlacing=2):
    """Periodic-box P(k) multipoles via pypower (CatalogMesh + MeshFFTPower).

    Paints the catalog with `interlacing` shifted copies to suppress the
    grid aliasing that pylians' single-pass compensated assignment leaves
    behind at k gtrsim 0.25 (see power_tests/REPORT.md). Bins directly onto
    the fixed (K_MIN, DK_PK) output grid (see `_fixed_pk_kedges`), so the
    result needs no further rebinning. Returns (k, Pk[:, :3], Nmodes) in the
    same layout as `calcPk`.
    """
    from pypower import CatalogMesh, MeshFFTPower
    los = ['x', 'y', 'z'][axis]
    mesh = CatalogMesh(
        data_positions=np.ascontiguousarray(pos, dtype=np.float32),
        boxsize=L, boxcenter=L / 2, nmesh=N,
        resampler=resampler.lower(), interlacing=interlacing,
        position_type='pos')
    kedges = _fixed_pk_kedges(np.pi * N / L)
    poles = MeshFFTPower(mesh, edges=kedges, ells=(0, 2, 4), los=los).poles
    Pk = np.stack([
        poles(ell=0, complex=False, remove_shotnoise=False),
        poles(ell=2, complex=False),
        poles(ell=4, complex=False)], axis=-1)
    return poles.k, Pk, poles.nmodes


def calcPk_pypower_field(field, L, axis=0, MAS='CIC'):
    """Periodic-box P(k) multipoles via pypower for an already-painted field
    (e.g. the nbody density field written by fastpm/borglpt/etc).

    No interlacing is applied here: the field was painted once upstream by
    the N-body code, so this only swaps the assignment-window
    deconvolution/FFT backend, not the anti-aliasing treatment. `field` is
    the mean-zero density contrast (as stored in nbody.h5); pypower's
    normalization for a bare RealField is derived from the field's own mean
    density (`mesh.csum() / volume`), so it must be passed as 1+delta, not
    delta -- passing delta directly makes the mean ~0 and blows up wnorm.
    Bins directly onto the fixed (K_MIN, DK_PK) output grid, as above.
    """
    from pypower import ArrayMesh, MeshFFTPower
    los = ['x', 'y', 'z'][axis]
    N = field.shape[0]
    mesh = ArrayMesh(
        np.ascontiguousarray(1 + field, dtype=np.float32),
        boxsize=L, type='real', nmesh=N)
    kedges = _fixed_pk_kedges(np.pi * N / L)
    poles = MeshFFTPower(mesh, edges=kedges, ells=(0, 2, 4), los=los,
                         compensations=MAS.lower()).poles
    Pk = np.stack([
        poles(ell=0, complex=False, remove_shotnoise=False),
        poles(ell=2, complex=False),
        poles(ell=4, complex=False)], axis=-1)
    return poles.k, Pk, poles.nmodes


def rebin_pk(k, Pk, Nmodes):
    """Rebin pylians P(k) onto a fixed physical grid using mode-weighted averaging.

    Pylians bins at width k_F = 2pi/L, which varies with box size. This maps
    those fine, volume-dependent bins onto a fixed grid (K_MIN, k_Nyq, DK_PK)
    so that the output data vector has consistent meaning and length across
    simulations of different volumes.

    Within each output bin, modes are averaged weighted by Nmodes — the number
    of Fourier modes in each fine bin. This is the statistically correct weight:
    bins with more modes have smaller variance and should contribute more to the
    average. It is equivalent to computing the straight average of all individual
    k-modes that fall in the coarse bin.

    Pk has shape (N_bins_fine, n_ell) where n_ell=3 (monopole, quadrupole,
    hexadecapole) as returned by pylians. Returns (k_centers, Pk_rebinned)
    where Pk_rebinned has shape (N_bins_coarse, n_ell). Output bins that
    contain no pylians modes are left as nan.
    """
    k_nyq = k.max()
    # Build fixed output bin edges and centers
    n_bins = (k_nyq - K_MIN) // DK_PK
    k_edges = np.append(K_MIN + DK_PK*np.arange(n_bins+1),
                        k_nyq)   # last bin shorter
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    n_ell = Pk.shape[1] if Pk.ndim > 1 else 1
    Pk_out = np.full((len(k_centers), n_ell), np.nan)
    for i, (lo, hi) in enumerate(zip(k_edges[:-1], k_edges[1:])):
        # Select all pylians bins whose centers fall in this output bin
        mask = (k >= lo) & (k < hi)
        if not mask.any():
            continue
        # Weight by number of Fourier modes: more modes = lower variance = higher weight
        w = Nmodes[mask]
        Pk_out[i] = np.average(Pk[mask], weights=w, axis=0)
    return k_centers, Pk_out


def calcQk_polybin(k, Pk, k123, Bk):
    # Reducing bispectrum with the power spectrum monopole
    # Qk = Bk123 / (Pk1 * Pk2 + Pk2 * Pk3 + Pk3 * Pk1)
    Pk1 = np.interp(k123[0], k, Pk[0])[None, :]
    Pk2 = np.interp(k123[1], k, Pk[0])[None, :]
    Pk3 = np.interp(k123[2], k, Pk[0])[None, :]
    Qk = Bk / (Pk1 * Pk2 + Pk2 * Pk3 + Pk3 * Pk1)
    return Qk


@timing_decorator
def calcBk_polybin(delta, L, axis=0, MAS='CIC', threads=16):
    assert axis == 2  # polybin measures along the z axis

    n_mesh = delta.shape[0]
    n_bins = (K_MAX_BK - K_MIN) // DK_BK
    k_bins = np.append(K_MIN + DK_BK*np.arange(n_bins+1),
                       K_MAX_BK)  # last bin shorter

    # set stuff up
    base = pb.PolyBin3D(
        sightline='global',  # TODO: check if this properly computes RSDs
        gridsize=[n_mesh]*3,
        boxsize=[L]*3,
        boxcenter=(0., 0., 0.),
        pixel_window=MAS.lower(),
        backend='fftw',
        nthreads=threads
    )
    pspec = pb.PSpec(
        base,
        k_bins,  # k-bin edges
        lmax=2,  # Legendre multipoles
        mask=None,  # real-space mask
        applySinv=None,  # filter to apply to data
    )
    bspec = pb.BSpec(
        base,
        k_bins,  # k-bin edges
        lmax=2,  # Legendre multipoles
        mask=None,  # real-space mask
        applySinv=None,  # filter to apply to data
        k_bins_squeeze=None,
        include_partial_triangles=False,
    )
    # compute
    pk_corr = pspec.Pk_ideal(delta, discreteness_correction=True)
    bk_corr = bspec.Bk_ideal(delta, discreteness_correction=True)

    # set up outputs
    k = pspec.get_ks()
    k123 = bspec.get_ks()
    pk = np.array([pk_corr[f'p{ell}'] for ell in [0, 2]])
    bk = np.array([bk_corr[f'b{ell}'] for ell in [0, 2]])
    qk = calcQk_polybin(k, pk, k123, bk)

    return k123, bk, qk, k, pk


def calcQk_bfast(Pk, Bk):
    Qk = Bk / (Pk[0] * Pk[1] + Pk[1] * Pk[2] + Pk[2] * Pk[0])
    return Qk


@timing_decorator
def calcBk_bfast(delta, L, axis=0, MAS='CIC', threads=16, cache_dir=None):
    if cache_dir is None:
        cache_dir = './'
    else:
        cache_dir += '/'
    os.makedirs(cache_dir, exist_ok=True)

    kF = 2*np.pi/L  # Fundamental frequency
    kmax = np.pi*delta.shape[0]/L  # Nyquist frequency
    Nbins = 23  # Set due to GPU memory limits for 3 Gpc/h box
    # kbinning which allows Nbins in this range (from Bfast code)
    fc = dk = kmax/kF/(Nbins+1/2)  # span kF to kmax

    result = BFast.Bk(
        delta, L, fc, dk, Nbins, triangle_type='All', MAS=MAS,
        fast=True, precision='float32', verbose=False,
        file_path=cache_dir, open_triangles=False
    )

    # remove when k1 >= k2 + k3, because open_triangles=False doesn't do it
    k123 = result[:, :3].T * kF  # k123
    mask = k123[0] < k123[1] + k123[2]

    k123 = k123[:, mask]
    pk = result[mask, 3:6].T
    bk = result[mask, 6:7].T
    counts = result[mask, 7]  # number of triangles in each bin (not used)
    qk = calcQk_bfast(pk, bk)

    # 1D k-bin centers for the power spectrum, extracted from the triangle legs
    k1d = np.unique(k123)
    return k123, bk, qk, k1d, pk


def get_redshift_space_pos(pos, vel, L, cosmo, z, axis=0):
    pos, vel = map(np.ascontiguousarray, (pos, vel))
    RSL.pos_redshift_space(pos, vel, L, cosmo.H(z).value/cosmo.h, z, axis)
    pos %= L
    return pos


def get_box_catalogue(pos, z, L, N):
    from summarizer.data import BoxCatalogue  # only import if needed

    return BoxCatalogue(
        galaxies_pos=pos,
        redshift=z,
        boxsize=L,
        n_mesh=N,
    )


def get_box_catalogue_rsd(pos, vel, z, L, cosmo, axis, N):
    pos = get_redshift_space_pos(
        pos=pos, vel=vel, z=z, cosmo=cosmo, axis=axis, L=L)
    return get_box_catalogue(pos, z, L, N)
