
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
                    summ[stat] = {
                        'k': f[a][stat+'_k3D'][:],
                        'value': f[a][stat][:],
                    }
    except (OSError, KeyError):
        return {}
    return summ


def preprocess_Pk(X, kmax, poles=[0]):
    Xout = []
    for x in X:
        k, value = x['k'], x['value']
        # only use desired poles
        value = value[..., poles]
        # cut k
        value = value[k < kmax, ...]
        Xout.append(value)
    Xout = np.array(Xout)

    # log transform
    Xout = np.log(Xout + 1e-5)

    # impute nans
    Xout = np.nan_to_num(Xout, nan=0.0)

    # flatten
    Xout = Xout.reshape(len(Xout), -1)

    return Xout
