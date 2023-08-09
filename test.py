import numpy as np
import jax.numpy as jnp
import jax.scipy.ndimage
import jax
from functools import partial


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
