"""Tools to support halo and galaxy biasing."""
# TODO: This file is a mess. It needs to be cleaned up and organized.

import logging
import numpy as np
import astropy.units as apu
from astropy.coordinates import SkyCoord
import scipy
import tqdm
from sklearn.neighbors import KNeighborsRegressor
import jax.numpy as jnp
import jax.scipy.ndimage
import jax
from functools import partial
from pmwd import Configuration, Particles, scatter
from ...utils import timing_decorator
from ...nbody.tools import vfield_CIC


# General

@timing_decorator
def pad_3d(ppos, pvel, Lbox, Lpad):
    # pad a 3d particle cube with periodic boundary conditions

    def offset(*inds):
        # calculate the offset vector for a given region
        out = np.zeros(3)
        out[list(inds)] = Lbox
        return out

    def recursive_padding(ipos, idig, ivel, index=0):
        # recursively pad the cube
        if index >= 3:
            return []
        padded = []
        for i, dir in [(1, 1), (3, -1)]:
            mask = idig[:, index] == i
            ippad, ivpad = ipos[mask] + dir*offset(index), ivel[mask]
            padded += [(ippad, ivpad)]
            padded += recursive_padding(ippad, idig[mask], ivpad, index+1)
            padded += recursive_padding(ippad, idig[mask], ivpad, index+2)
        return padded

    regions = np.digitize(ppos, bins=[0, Lpad, Lbox - Lpad, Lbox])
    padlist = [(ppos, pvel)]
    padlist += recursive_padding(ppos, regions, pvel, index=0)
    padlist += recursive_padding(ppos, regions, pvel, index=1)
    padlist += recursive_padding(ppos, regions, pvel, index=2)
    padpos, padvel = zip(*padlist)
    padpos = np.concatenate(padpos)
    padvel = np.concatenate(padvel)

    logging.info(
        f'len(pad)/len(original): {len(padpos)}/{len(ppos)}'
        f' = {len(padpos)/len(ppos):.3f}')
    return padpos, padvel


# Deaglan's halo positioning
@timing_decorator
def sample_uniform(N: int, Nt: int, L: float, frac_sig_x: float, origin: np.ndarray):
    """
    Generate Nt points uniformly sampled from a box of side length L.
    The points are then radially perturbed by a fractional amount
    frac_sig_x, where the observer sits at x = (0, 0, 0).

    Args:
        - N (int): The number of grid points per side.
        - Nt (int): The number of tracers to generate.
        - L (float): The side-length of the box (Mpc/h).
        - frac_sig_x (float): The fractional uncertainty in the radial direction for noise.
        - origin (np.ndarray): The coordinates of the origin of the box (Mpc/h).

    Returns:
        - xtrue (np.ndarray): The true coordinates (Mpc/h) of the tracers.
        - xmeas (np.ndarray): The observed coordiantes (Mpc/h) of the tracers.
        - sigma_mu (float):  The uncertainty in the distance moduli of the tracers.
    """

    h = 1

    xtrue = np.random.uniform(low=0.0, high=N+1, size=Nt)
    ytrue = np.random.uniform(low=0.0, high=N+1, size=Nt)
    ztrue = np.random.uniform(low=0.0, high=N+1, size=Nt)

    # Convert to coordinates, and move relative to origin
    xtrue *= L / N  # Mpc/h
    ytrue *= L / N  # Mpc/h
    ztrue *= L / N  # Mpc/h

    xtrue += origin[0]
    ytrue += origin[1]
    ztrue += origin[2]

    # Convert to RA, Dec, Distance
    rtrue = np.sqrt(xtrue ** 2 + ytrue ** 2 + ztrue ** 2)   # Mpc/h
    c = SkyCoord(x=xtrue, y=ytrue, z=ztrue, representation_type='cartesian')
    RA = c.spherical.lon.degree
    Dec = c.spherical.lat.degree
    r_hat = np.array(SkyCoord(ra=RA*apu.deg, dec=Dec*apu.deg).cartesian.xyz)

    # Add noise to radial direction
    sigma_mu = 5. / np.log(10) * frac_sig_x
    mutrue = 5 * np.log10(rtrue * h * 1.e6 / 10)
    mumeas = mutrue + np.random.normal(size=len(mutrue)) * sigma_mu
    rmeas = 10 ** (mumeas / 5.) * 10 / h / 1.e6
    xmeas, ymeas, zmeas = rmeas[None, :] * r_hat

    xtrue = np.array([xtrue, ytrue, ztrue])
    xmeas = np.array([xmeas, ymeas, zmeas])

    return xtrue, xmeas, sigma_mu


def draw_linear(nsamp: int, alpha: float, beta: float, u0: float, u1: float) -> np.ndarray:
    """
    Draw a sample from the probability distribution:
    p(u) \propto alpha (u1 - u) + beta (u - u0)
    for u0 <= u <= u1 and p(u) = 0 otherwise.

    Args:
        - nsamp (int): Number of samples to draw.
        - alpha (float): The coefficient of (u1 - u) in p(u).
        - beta (float): The coefficient of (u - u0) in p(u).
        - u0 (float): The minimum allowed value of u.
        - u1 (float): The maximum allowed value of u.

    Return:
        - np.ndarray: The samples from p(u).
    """
    n = scipy.stats.uniform(0, 1).rvs(nsamp)
    if isinstance(alpha, np.ndarray):
        res = np.zeros(alpha.shape)
        m = alpha != beta
        res[m] = ((u1 - u0) * np.sqrt(n * (beta ** 2 - alpha ** 2) +
                  alpha ** 2) - u1 * alpha + u0 * beta)[m] / (beta - alpha)[m]
        res[~m] = (u0 + (u1 - u0) * n)[~m]
        return res
    else:
        if alpha != beta:
            return ((u1 - u0) * np.sqrt(n * (beta ** 2 - alpha ** 2) + alpha ** 2) - u1 * alpha + u0 * beta) / (beta - alpha)
        else:
            return u0 + (u1 - u0) * n


def periodic_index(index: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Apply periodic boundary conditions to an array of indices.

    Args:
        - index (np.ndarray): The indices to transform. Shape = (ndim, nvals). 
        - shape (tuple): The shape of the box used for periodic boundary conditions (N0, N1, ...)

    Returns:
        - new_index (np.ndarray): The values in index after applying periodic boundary conditions, such that for dimension i, the values are in the range [0, Ni)
    """
    assert index.shape[0] == len(shape)
    new_index = index.copy()
    for i in range(len(shape)):
        new_index[i, :] = np.mod(new_index[i, :], shape[i])
    return new_index


def get_new_index(index: np.ndarray, shape: tuple, subscript: tuple) -> np.ndarray:
    """
    If each entry of index corresponds to (0,0,0), find the index corresponding to the point given by subscript.

    Args:
        - index (np.ndarray): The indices to transform. Shape = (ndim, nvals). 
        - shape (tuple): The shape of the box used (N0, N1, ...).
        - subscript (tuple): The coordinate to find, relative to the values given in index.

    Returns:
        - new_index (np.ndarray): The new index values.

    """
    new_index = index.copy()
    for i in range(len(subscript)):
        new_index[i, :] += subscript[i]
    new_index = periodic_index(new_index, shape)
    return new_index


def sample_3d(phi: np.ndarray, Nt: int, L: float, frac_sig_x: float, origin: np.ndarray) -> np.ndarray:
    """
    Sample Nt points, assuming that the points are drawn from a Poisson process given by the field phi.
    phi gives the value of the field at the grid points, and we assume linear interpolation between points.
    The points are then radially perturbed by a fractional amount
    frac_sig_x, where the observer sits at x = (0, 0, 0).

    Args:
        - phi (np.ndarray): The field which defines the mean of the Poisson process.
        - Nt (int): The number of tracers to generate.
        - L (float): The side-length of the box (Mpc/h).
        - frac_sig_x (float): The fractional uncertainty in the radial direction for noise.
        - origin (np.ndarray): The coordinates of the origin of the box (Mpc/h).

    Returns:
        - xtrue (np.ndarray): The true coordinates (Mpc/h) of the tracers.
        - xmeas (np.ndarray): The observed coordiantes (Mpc/h) of the tracers.
        - sigma_mu (float):  The uncertainty in the distance moduli of the tracers.
    """

    N = phi.shape[0]
    h = 1

    # (1) Find which cell each point lives in
    mean = phi + \
        np.roll(phi, -1, axis=0) + \
        np.roll(phi, -1, axis=1) + \
        np.roll(phi, -1, axis=2) + \
        np.roll(phi, -1, axis=(0, 1)) + \
        np.roll(phi, -1, axis=(0, 2)) + \
        np.roll(phi, -1, axis=(1, 2)) + \
        np.roll(phi, -1, axis=(0, 1, 2))
    prob = mean.flatten() / mean.sum()
    i = np.arange(prob.shape[0])
    a1d = np.random.choice(i, Nt, p=prob)
    a3d = np.array(np.unravel_index(a1d, (N, N, N)))

    # (2) Find the x values
    shape = (N, N, N)
    alpha = np.zeros(Nt)
    for subscript in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]:
        idx = get_new_index(a3d, shape, subscript)
        alpha += phi[idx[0, :], idx[1, :], idx[2, :]]
    beta = np.zeros(Nt)
    for subscript in [(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
        idx = get_new_index(a3d, shape, subscript)
        beta += phi[idx[0, :], idx[1, :], idx[2, :]]
    u0 = a3d[0, :]
    u1 = a3d[0, :] + 1
    xtrue = draw_linear(Nt, alpha, beta, u0, u1)

    # (3) Find the y values
    shape = (N, N, N)
    alpha = np.zeros(Nt)
    for subscript in [(0, 0, 0), (0, 0, 1)]:
        idx = get_new_index(a3d, shape, subscript)
        alpha += phi[idx[0, :], idx[1, :], idx[2, :]] * (a3d[0, :] + 1 - xtrue)
    for subscript in [(1, 0, 0), (1, 0, 1)]:
        idx = get_new_index(a3d, shape, subscript)
        alpha += phi[idx[0, :], idx[1, :], idx[2, :]] * (xtrue - a3d[0, :])
    beta = np.zeros(Nt)
    for subscript in [(0, 1, 0), (0, 1, 1)]:
        idx = get_new_index(a3d, shape, subscript)
        beta += phi[idx[0, :], idx[1, :], idx[2, :]] * (a3d[0, :] + 1 - xtrue)
    for subscript in [(1, 1, 0), (1, 1, 1)]:
        idx = get_new_index(a3d, shape, subscript)
        beta += phi[idx[0, :], idx[1, :], idx[2, :]] * (xtrue - a3d[0, :])
    u0 = a3d[1, :]
    u1 = a3d[1, :] + 1
    ytrue = draw_linear(Nt, alpha, beta, u0, u1)

    # (4) Find the z values
    xd = (xtrue - a3d[0, :])  # x1-x0=1 so xd = x - x0
    yd = (ytrue - a3d[1, :])  # y1-y0=1 so yd = y - y0
    ia = get_new_index(a3d, shape, (0, 0, 0))
    ib = get_new_index(a3d, shape, (1, 0, 0))
    phi00 = phi[ia[0, :], ia[1, :], ia[2, :]] * (1 - xd) + \
        phi[ib[0, :], ib[1, :], ib[2, :]] * xd
    ia = get_new_index(a3d, shape, (0, 0, 1))
    ib = get_new_index(a3d, shape, (1, 0, 1))
    phi01 = phi[ia[0, :], ia[1, :], ia[2, :]] * (1 - xd) + \
        phi[ib[0, :], ib[1, :], ib[2, :]] * xd
    ia = get_new_index(a3d, shape, (0, 1, 0))
    ib = get_new_index(a3d, shape, (1, 1, 0))
    phi10 = phi[ia[0, :], ia[1, :], ia[2, :]] * (1 - xd) + \
        phi[ib[0, :], ib[1, :], ib[2, :]] * xd
    ia = get_new_index(a3d, shape, (0, 1, 1))
    ib = get_new_index(a3d, shape, (1, 1, 1))
    phi11 = phi[ia[0, :], ia[1, :], ia[2, :]] * (1 - xd) + \
        phi[ib[0, :], ib[1, :], ib[2, :]] * xd
    alpha = phi00 * (1 - yd) + phi10 * yd  # alpha = phi0
    beta = phi01 * (1 - yd) + phi11 * yd   # beta = phi1
    u0 = a3d[2, :]
    u1 = a3d[2, :] + 1
    ztrue = draw_linear(Nt, alpha, beta, u0, u1)

    # Convert to coordinates, and move relative to origin
    xtrue *= L / N  # Mpc/h
    ytrue *= L / N  # Mpc/h
    ztrue *= L / N  # Mpc/h

    xtrue += origin[0]
    ytrue += origin[1]
    ztrue += origin[2]

    # Convert to RA, Dec, Distance
    rtrue = np.sqrt(xtrue ** 2 + ytrue ** 2 + ztrue ** 2)   # Mpc/h
    c = SkyCoord(x=xtrue, y=ytrue, z=ztrue, representation_type='cartesian')
    RA = c.spherical.lon.degree
    Dec = c.spherical.lat.degree
    r_hat = np.array(SkyCoord(ra=RA*apu.deg, dec=Dec*apu.deg).cartesian.xyz)

    # Add noise to radial direction
    sigma_mu = 5. / np.log(10) * frac_sig_x
    mutrue = 5 * np.log10(rtrue * h * 1.e6 / 10)
    mumeas = mutrue + np.random.normal(size=len(mutrue)) * sigma_mu
    rmeas = 10 ** (mumeas / 5.) * 10 / h / 1.e6
    xmeas, ymeas, zmeas = rmeas[None, :] * r_hat

    xtrue = np.array([xtrue, ytrue, ztrue])
    xmeas = np.array([xmeas, ymeas, zmeas])

    return xtrue, xmeas, sigma_mu


# velocity fields

@timing_decorator
def sample_velocities_CIC(hpos, cfg, fvel=None, ppos=None, pvel=None):
    nbody = cfg.nbody

    if fvel is None:
        logging.info('Measuring CIC velocity field from particles...')
        if ppos is None or pvel is None:
            raise ValueError('No particles found for CIC interpolation.')
        fvel = vfield_CIC(ppos, pvel, cfg, interp=False)

    # Interpolate to halo positions
    hvel = [interp_field(fvel.T, hpos[i], nbody.L, np.zeros(3), order=1).T
            for i in range(len(hpos))]

    for i in range(len(hvel)):
        if np.any(np.isnan(hvel[i])):
            raise ValueError(
                'NaNs in halo velocities from CIC interpolation. '
                'Recommend reducing velocity mesh size, Nmesh.')

    return hvel


@timing_decorator
def sample_velocities_kNN(hpos, ppos, pvel):
    knn = KNeighborsRegressor(
        n_neighbors=5, leaf_size=1000,
        algorithm='ball_tree', weights='distance', n_jobs=-1)
    knn.fit(ppos, pvel)
    vtrues = [knn.predict(x) for x in tqdm.tqdm(hpos)]
    return vtrues


@timing_decorator
def sample_velocities_density(hpos, rho, L, Omega_m, smooth_R):
    vel = get_vgrid(rho, L, smooth_R, f=Omega_m**0.55)
    hvel = [interp_field(vel, hpos[i], L, np.zeros(3), order=1).T
            for i in range(len(hpos))]
    return hvel


def Fourier_ks(N_BOX: int, l: float) -> jnp.ndarray:
    """
    Compute the Fourier k grid for a given box

    Args:
        - N_BOX (int): The number of grid points per side of the box
        - l (float): The side-length of the box

    Returns:
        - jnp.ndarray: Array of k values for the Fourier grid

    """
    kx = 2*np.pi*np.fft.fftfreq(N_BOX, d=l)
    ky = 2*np.pi*np.fft.fftfreq(N_BOX, d=l)
    kz = 2*np.pi*np.fft.fftfreq(N_BOX, d=l)

    N_BOX_Z = (N_BOX//2 + 1)

    kx_vec = np.tile(kx[:, None, None], (1, N_BOX, N_BOX_Z))
    ky_vec = np.tile(ky[None, :, None], (N_BOX, 1, N_BOX_Z))
    kz_vec = np.tile(kz[None, None, :N_BOX_Z], (N_BOX, N_BOX, 1))

    k_norm = np.sqrt(kx_vec**2 + ky_vec**2 + kz_vec**2)
    k_norm[(k_norm < 1e-10)] = 1e-15

    return jnp.array([kx_vec, ky_vec, kz_vec]), jnp.array(k_norm)


@partial(jax.jit, static_argnames=['order'])
def jit_map_coordinates(image: jnp.ndarray, coords: jnp.ndarray, order: int) -> jnp.ndarray:
    """
    Jitted version of jax.scipy.ndimage.map_coordinates

    Args:
        - image (jnp.ndarray) - The input array
        - coords (jnp.ndarray) - The coordinates at which image is evaluated.
        - order (int): order of interpolation (0 <= order <= 5)
    Returns:
        - map_coordinates (jnp.ndarray) - The result of transforming the input. The shape of the output is derived from that of coordinates by dropping the first axis.

    """
    return jax.scipy.ndimage.map_coordinates(image, coords, order=order, mode='wrap')


def interp_field(input_array: jnp.ndarray, coords: jnp.ndarray, L: float, origin: jnp.ndarray, order: int, use_jitted: bool = False) -> jnp.ndarray:
    """
    Interpolate an array on a ND-cubic grid to new coordinates linearly

    Args:
        - input_array (jnp.ndarray): array to be interpolated
        - coords (jnp.ndarray shape=(npoint,ndim)): coordinates to evaluate at
        - L (float): length of box
        - origin (jnp.ndarray, shape=(ndim,)): position corresponding to index [0,0,...]
        - order (int): order of interpolation (0 <= order <= 5)

    Returns:
        - out_array (np.ndarray, SHAPE): field evaluated at coords of interest

    """

    N = input_array.shape[-1]

    # Change coords to index
    pos = (coords - origin) * N / L

    # NOTE: jax's 'wrap' is the same as scipy's 'grid-wrap'
    if len(input_array.shape) == coords.shape[1]:
        # Scalar
        if use_jitted:
            out_array = jit_map_coordinates(input_array, pos.T, order)
        else:
            out_array = jax.scipy.ndimage.map_coordinates(
                input_array, pos.T, order=order, mode='wrap')
    elif len(input_array.shape) == coords.shape[1] + 1:
        # Vector
        if use_jitted:
            out_array = jnp.array([jit_map_coordinates(
                input_array[i, ...], pos.T, order) for i in range(input_array.shape[0])])
        else:
            out_array = jnp.array([jax.scipy.ndimage.map_coordinates(
                input_array[i, ...], pos.T, order=order, mode='wrap') for i in range(input_array.shape[0])])
    else:
        raise NotImplementedError("Cannot interpolate arbitrary tensor")

    return out_array


@jax.jit
def project_radial(vec: jnp.ndarray, coords: jnp.ndarray, origin: jnp.ndarray) -> jnp.ndarray:
    """
    Project vectors along the radial direction, given by coords
    Args:
        - vec (jnp.ndarray, shape=(npoint,ndim)): array of vectors to be projected
        - coords (jnp.ndarray shape=(npoint,ndim)): coordinates to evaluate at
        - origin (jnp.ndarray, shape=(ndim,)): position corresponding to index [0,0,...]

    Returns:
        - vr (jnp.ndarray, shape=(npoint,)): vec projected along radial direction
    """
    x = coords - jnp.expand_dims(origin, axis=0)
    r = jnp.sqrt(jnp.sum(x**2, axis=1))
    x = x / jnp.expand_dims(r, axis=1)
    vr = jnp.sum(x * vec.T, axis=1)

    return vr


def get_vgrid(delta: jnp.ndarray, L_BOX: float, smooth_R: float, f: float) -> jnp.ndarray:
    """
    Convert an overdensity field to a velocity field

    Args:
        - delta (jnp.ndarray): The overdensity field
        - L_BOX (float): Box length along each axis (Mpc/h)
        - smooth_R (float): Smoothing scale (Mpc/h) to smooth density field before velocity computation
        - f (float): The logarithmic growth rate

    Returns:
        - jnp.ndarray: The corresponding velocity field (km/s)
    """

    # Some useful quantities
    N_SIDE = delta.shape[0]
    N_Z = N_SIDE // 2 + 1
    l = L_BOX / N_SIDE
    dV = l ** 3
    V = L_BOX ** 3

    # Precommpute quantities for FFT calculation
    k, k_norm = Fourier_ks(N_SIDE, l)
    mask = np.ones((N_SIDE, N_SIDE, N_Z))
    mask[N_Z:, :, 0] = 0.
    mask[N_Z:, :, -1] = 0.
    Fourier_mask = mask
    prior_mask = np.array([mask, mask])
    zero_ny_ind = [0, N_Z - 1]
    for a in zero_ny_ind:
        for b in zero_ny_ind:
            for c in zero_ny_ind:
                prior_mask[0, a, b, c] = 0.5 * prior_mask[0, a, b, c]
                prior_mask[1, a, b, c] = 0.
    update_index_real_0 = jnp.index_exp[0, N_Z:, :, 0]
    update_index_real_ny = jnp.index_exp[0, N_Z:, :, N_Z - 1]
    update_index_imag_0 = jnp.index_exp[1, N_Z:, :, 0]
    update_index_imag_ny = jnp.index_exp[1, N_Z:, :, N_Z - 1]
    flip_indices = -np.arange(N_SIDE)
    flip_indices[N_Z - 1] = -flip_indices[N_Z - 1]
    flip_indices = jnp.array(flip_indices.tolist())

    # FFT
    delta_k_complex = dV / V * jnp.fft.rfftn(delta)
    delta_k = jnp.array([delta_k_complex.real, delta_k_complex.imag])

    # Symmetrise
    delta_k = Fourier_mask[jnp.newaxis] * delta_k
    delta_k = delta_k.at[update_index_real_0].set(
        jnp.take(jnp.flip(delta_k[0, 1:(N_Z-1), :, 0],
                 axis=0), flip_indices, axis=1)
    )
    delta_k = delta_k.at[update_index_real_ny].set(
        jnp.take(jnp.flip(delta_k[0, 1:(N_Z-1), :,
                 N_Z - 1], axis=0), flip_indices, axis=1)
    )
    delta_k = delta_k.at[update_index_imag_0].set(
        -jnp.take(jnp.flip(delta_k[1, 1:(N_Z-1), :, 0],
                  axis=0), flip_indices, axis=1)
    )

    # Combine real and imaginary parts
    J = jnp.array(complex(0, 1))
    delta_k_complex = delta_k[0] + J * delta_k[1]

    # Filter field
    k_filter = jnp.exp(- 0.5 * (k_norm[:, :, :N_Z] * smooth_R)**2)
    kR = k_norm * smooth_R
    smooth_filter = np.exp(-0.5 * kR**2)
    v_kx = smooth_filter * J * 100 * k_filter * \
        f * delta_k_complex * k[0] / k_norm / k_norm
    v_ky = smooth_filter * J * 100 * k_filter * \
        f * delta_k_complex * k[1] / k_norm / k_norm
    v_kz = smooth_filter * J * 100 * k_filter * \
        f * delta_k_complex * k[2] / k_norm / k_norm

    # Convert back to real space
    vx = (jnp.fft.irfftn(v_kx) * V / dV)
    vy = (jnp.fft.irfftn(v_ky) * V / dV)
    vz = (jnp.fft.irfftn(v_kz) * V / dV)

    return jnp.array([vx, vy, vz])
