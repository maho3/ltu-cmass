import numpy as np
import Pk_library as PKL
import MAS_library as MASL
import redshift_space_library as RSL
import PolyBin3D as pb


def get_redshift_space_pos(pos, vel, L, h, z, axis=0):
    pos, vel = map(np.ascontiguousarray, (pos, vel))
    RSL.pos_redshift_space(pos, vel, L, h*100, z, axis)
    pos %= L
    return pos


def MA(pos, L, N, MAS='CIC'):
    pos = np.ascontiguousarray(pos)
    delta = np.zeros((N, N, N), dtype=np.float32)
    pos %= L
    MASL.MA(pos, delta, BoxSize=L, MAS=MAS)
    delta = delta.astype(np.float64)
    delta /= np.mean(delta)
    delta -= 1
    return delta


def MAz(pos, vel, L, N, cosmo, z, MAS='CIC', axis=0):
    pos, vel = map(np.ascontiguousarray, (pos, vel))
    RSL.pos_redshift_space(pos, vel, L, cosmo.H(z).value/cosmo.h, z, axis)
    return MA(pos, L, N, MAS)


def calcPk(delta, L, axis=0, MAS='CIC', threads=16):
    Pk = PKL.Pk(delta.astype(np.float32), L, axis, MAS, threads, verbose=False)
    k = Pk.k3D
    Pk = Pk.Pk
    return k, Pk


def calcBk(delta, L, axis=0, MAS='CIC', threads=16):
    # TODO: Use ili-summarizer here
    k_min = 1.05*2 * np.pi / L
    n_mesh = delta.shape[0]
    k_max = 0.95 * np.pi * n_mesh / L
    num_bins = 12
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), num_bins)

    base = pb.PolyBin3D(
        sightline='global',
        gridsize=n_mesh,
        boxsize=[L]*3,
        boxcenter=(0., 0., 0.),
        pixel_window=MAS.lower(),
        backend='jax'
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
    bk_corr = bspec.Bk_ideal(
        delta,
        discreteness_correction=True,
    )
    k123 = bspec.get_ks()
    weight = k123.prod(axis=0)
    return k123, np.array(
        [bk_corr[f'b{ell}'] * weight for ell in [0, 2]]
    )


def get_box_catalogue(pos, z, L, N):
    from summarizer.data import BoxCatalogue  # only import if needed

    return BoxCatalogue(
        galaxies_pos=pos,
        redshift=z,
        boxsize=L,
        n_mesh=N,
    )


def get_box_catalogue_rsd(pos, vel, z, L, h, axis, N):
    pos = get_redshift_space_pos(pos=pos, vel=vel, z=z, h=h, axis=axis, L=L,)
    return get_box_catalogue(pos, z, L, N)
