

# Quijote

from matplotlib.patches import Rectangle
import astropy.units as apu
from astropy.coordinates import SkyCoord
import scipy.stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import os
import sys
from struct import unpack


class FoF_catalog:
    def __init__(self, basedir, snapnum, long_ids=False, swap=False,
                 SFR=False, read_IDs=True, prefix='/groups_'):

        if long_ids:
            format = np.uint64
        else:
            format = np.uint32

        exts = ('000'+str(snapnum))[-3:]

        #################  READ TAB FILES #################
        fnb, skip, Final = 0, 0, False
        dt1 = np.dtype((np.float32, 3))
        dt2 = np.dtype((np.float32, 6))
        prefix = basedir + prefix + exts + "/group_tab_" + exts + "."
        while not (Final):
            f = open(prefix+str(fnb), 'rb')
            self.Ngroups = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.TotNgroups = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.Nids = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.TotNids = np.fromfile(f, dtype=np.uint64, count=1)[0]
            self.Nfiles = np.fromfile(f, dtype=np.uint32, count=1)[0]

            TNG, NG = self.TotNgroups, self.Ngroups
            if fnb == 0:
                self.GroupLen = np.empty(TNG, dtype=np.int32)
                self.GroupOffset = np.empty(TNG, dtype=np.int32)
                self.GroupMass = np.empty(TNG, dtype=np.float32)
                self.GroupPos = np.empty(TNG, dtype=dt1)
                self.GroupVel = np.empty(TNG, dtype=dt1)
                self.GroupTLen = np.empty(TNG, dtype=dt2)
                self.GroupTMass = np.empty(TNG, dtype=dt2)
                if SFR:
                    self.GroupSFR = np.empty(TNG, dtype=np.float32)

            if NG > 0:
                locs = slice(skip, skip+NG)
                self.GroupLen[locs] = np.fromfile(f, dtype=np.int32, count=NG)
                self.GroupOffset[locs] = np.fromfile(
                    f, dtype=np.int32, count=NG)
                self.GroupMass[locs] = np.fromfile(
                    f, dtype=np.float32, count=NG)
                self.GroupPos[locs] = np.fromfile(f, dtype=dt1, count=NG)
                self.GroupVel[locs] = np.fromfile(f, dtype=dt1, count=NG)
                self.GroupTLen[locs] = np.fromfile(f, dtype=dt2, count=NG)
                self.GroupTMass[locs] = np.fromfile(f, dtype=dt2, count=NG)
                if SFR:
                    self.GroupSFR[locs] = np.fromfile(
                        f, dtype=np.float32, count=NG)
                skip += NG

                if swap:
                    self.GroupLen.byteswap(True)
                    self.GroupOffset.byteswap(True)
                    self.GroupMass.byteswap(True)
                    self.GroupPos.byteswap(True)
                    self.GroupVel.byteswap(True)
                    self.GroupTLen.byteswap(True)
                    self.GroupTMass.byteswap(True)
                    if SFR:
                        self.GroupSFR.byteswap(True)

            curpos = f.tell()
            f.seek(0, os.SEEK_END)
            if curpos != f.tell():
                raise Exception(
                    "Warning: finished reading before EOF for tab file", fnb)
            f.close()
            fnb += 1
            if fnb == self.Nfiles:
                Final = True

        #################  READ IDS FILES #################
        if read_IDs:

            fnb, skip = 0, 0
            Final = False
            while not (Final):
                fname = basedir+"/groups_" + exts + \
                    "/group_ids_"+exts + "."+str(fnb)
                f = open(fname, 'rb')
                Ngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
                TotNgroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
                Nids = np.fromfile(f, dtype=np.uint32, count=1)[0]
                TotNids = np.fromfile(f, dtype=np.uint64, count=1)[0]
                Nfiles = np.fromfile(f, dtype=np.uint32, count=1)[0]
                Send_offset = np.fromfile(f, dtype=np.uint32, count=1)[0]
                if fnb == 0:
                    self.GroupIDs = np.zeros(dtype=format, shape=TotNids)
                if Ngroups > 0:
                    if long_ids:
                        IDs = np.fromfile(f, dtype=np.uint64, count=Nids)
                    else:
                        IDs = np.fromfile(f, dtype=np.uint32, count=Nids)
                    if swap:
                        IDs = IDs.byteswap(True)
                    self.GroupIDs[skip:skip+Nids] = IDs[:]
                    skip += Nids
                curpos = f.tell()
                f.seek(0, os.SEEK_END)
                if curpos != f.tell():
                    raise Exception(
                        "Warning: finished reading before EOF for IDs file", fnb)
                f.close()
                fnb += 1
                if fnb == Nfiles:
                    Final = True


def load_quijote_halos(snapdir):
    FoF = FoF_catalog(snapdir, 4, long_ids=False,
                      swap=False, SFR=False, read_IDs=False)

    # get the properties of the halos
    pos_h = FoF.GroupPos/1e3  # Halo positions in Mpc/h
    mass = FoF.GroupMass*1e10  # Halo masses in Msun/h
    vel_h = FoF.GroupVel*(1.0+0)  # Halo peculiar velocities in km/s
    Npart = FoF.GroupLen  # Number of CDM particles in the halo

    return pos_h, mass, vel_h, Npart


# Simon's galaxy biasing


class TruncatedPowerLaw:
    @staticmethod
    def _get_mean_ngal(rho, nmean, beta, epsilon_g, rho_g):
        d = 1 + rho
        x = np.power(np.abs(d / rho_g), -epsilon_g)
        ngal_mean = nmean * np.power(d, beta) * np.exp(-x)
        return ngal_mean

    @staticmethod
    def _loss(params, delta, count_field):
        nmean, beta, epsilon_g, rho_g = params
        ngal_mean = TruncatedPowerLaw._get_mean_ngal(
            delta, nmean, beta, epsilon_g, rho_g)
        loss = np.mean(ngal_mean - count_field * np.log(ngal_mean))
        return loss

    def fit(self, delta, count_field):
        initial_guess = np.array([1.1, 1.1, 1.1, 0.51])
        bounds = [(0, None)] * 4  # Bounds for positive values
        result = minimize(
            self._loss, initial_guess, args=(delta, count_field),
            method='Nelder-Mead',
            bounds=bounds,
            options={'disp': True}
        )
        popt = result.x
        print(f"Power law bias fit params: {popt}")
        return popt

    def predict(self, delta, popt):
        ngal_mean = TruncatedPowerLaw._get_mean_ngal(delta, *popt)
        return ngal_mean

    def sample(self, delta, popt):
        ngal_mean = self.predict(delta, popt)
        return np.random.poisson(ngal_mean)


# Deaglan's halo positioning


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
