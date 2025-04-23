
import os
from os.path import join
import h5py
import numpy as np
from omegaconf import OmegaConf


def get_cosmo(source_path):
    cfg = OmegaConf.load(join(source_path, 'config.yaml'))
    return np.array(cfg.nbody.cosmo)


def get_hod(diagfile):
    with h5py.File(diagfile, 'r') as f:
        hod_params = f.attrs['HOD_params'][:]
        hod_names = f.attrs['HOD_names'][:]
    return hod_params, hod_names


def closest_a(lst, a):
    lst = [float(el) for el in lst]
    lst = np.asarray(lst)
    idx = (np.abs(lst - a)).argmin()
    return lst[idx]


def load_Pk(diag_file, a):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            a = closest_a(list(f.keys()), a)
            a = f'{a:.6f}'
            for stat in ['Pk', 'zPk']:
                if stat in f[a]:
                    for i in range(3):  # monopole, quadrupole, hexadecapole
                        summ[stat+str(2*i)] = {
                            'k': f[a][stat+'_k3D'][:],
                            'value': f[a][stat][:, i],
                        }
    except (OSError, KeyError):
        return {}
    return summ


def load_lc_Pk(diag_file):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            stat = 'Pk'
            for i in range(3):  # monopole, quadrupole, hexadecapole
                summ[stat+str(2*i)] = {
                    'k': f[stat+'_k3D'][:],
                    'value': f[stat][:, i],
                }
    except (OSError, KeyError):
        return {}
    return summ


def preprocess_Pk(X, kmax, monopole=True, norm=None, kmin=0.):
    if (not monopole) and (norm is None):
        raise ValueError('norm must be provided when monopole is False')

    Xout = []
    for x in X:
        k, value = x['k'], x['value']
        # cut k
        value = value[(kmin <= k) & (k <= kmax)]
        Xout.append(value)
    Xout = np.array(Xout)

    if monopole:
        # log transform
        Xout = np.log(Xout)
    else:
        # compute monopole
        Xnorm = []
        for x in norm:
            k, value = x['k'], x['value']
            # cut k
            value = value[(kmin <= k) & (k <= kmax)]
            Xnorm.append(value)
        Xnorm = np.array(Xnorm)

        # normalize by the monopole
        Xout /= Xnorm

    # impute nans
    Xout = np.nan_to_num(Xout, nan=0.0)

    # flatten
    Xout = Xout.reshape(len(Xout), -1)

    return Xout


def load_Bk(diag_file, a):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            a = closest_a(list(f.keys()), a)
            a = f'{a:.6f}'
            for stat in ['Bk', 'Qk', 'zBk', 'zQk']:
                if stat in f[a]:
                    for i in range(1):  # just monopole
                        summ[stat+str(2*i)] = {
                            'k': f[a]['Bk_k123'][:],
                            'value': f[a][stat][i, :],
                        }
    except (OSError, KeyError):
        return {}
    return summ


def load_lc_Bk(diag_file):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            for stat in ['Bk', 'Qk']:
                if stat in f:
                    for i in range(1):  # just monopole
                        summ[stat+str(2*i)] = {
                            'k': f['Bk_k123'][:],
                            'value': f[stat][i, :]
                        }
    except (OSError, KeyError):
        return {}
    return summ


def preprocess_Bk(X, kmax, kmin=0., log=False, equilateral_only=False):

    Xout = []
    for x in X:
        k, value = x['k'], x['value']

        # cut kmax and kmin
        m = ~ np.any((k < kmin) | (kmax < k), axis=0)
        value = value[m]
        k = k[:, m]

        if equilateral_only:
            # equal wavevector components and error tolerance for safety
            k1, k2, k3 = k
            m = np.isclose(k1, k2) & np.isclose(k2, k3)
            value = value[m]
        else:
            # check if k1 >= k2 + k3 for generic triangle configurations
            k1, k2, k3 = k
            m = (k1 < k2 + k3)
            value = value[m]

        if log:
            value = np.log10(value)
        Xout.append(value)
    Xout = np.array(Xout)

    # impute nans
    Xout = np.nan_to_num(Xout, nan=0.0)

    # flatten
    Xout = Xout.reshape(len(Xout), -1)

    return Xout
