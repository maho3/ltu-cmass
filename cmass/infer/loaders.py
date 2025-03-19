
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
    return hod_params


def load_Pk(diag_file, a):
    a = f'{a:.6f}'
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
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


def preprocess_Pk(X, kmax, monopole=True, norm=None):
    if (not monopole) and (norm is None):
        raise ValueError('norm must be provided when monopole is False')

    Xout = []
    for x in X:
        k, value = x['k'], x['value']
        # cut k
        value = value[k <= kmax]
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
            value = value[k <= kmax]
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
    a = f'{a:.6f}'
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
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
                            'value': f[stat][i, :] / np.prod(f['Bk_k123'][:], axis=0),
                        }
    except (OSError, KeyError):
        return {}
    return summ


def preprocess_Bk(X, kmax, log=False):

    Xout = []
    for x in X:
        k, value = x['k'], x['value']
        # cut k
        value = value[~ np.any(k > kmax, axis=0)]
        if log:
            value = np.log10(value)
        Xout.append(value)
    Xout = np.array(Xout)

    # impute nans
    Xout = np.nan_to_num(Xout, nan=0.0)

    # flatten
    Xout = Xout.reshape(len(Xout), -1)

    return Xout
