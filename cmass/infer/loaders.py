
import os
from os.path import join
import h5py
import numpy as np
from omegaconf import OmegaConf
from cmass.bias.hod import lookup_hod_model


def get_cosmo(source_path):
    cfg = OmegaConf.load(join(source_path, 'config.yaml'))
    return np.array(cfg.nbody.cosmo)


def get_hod_params(diagfile):
    with h5py.File(diagfile, 'r') as f:
        hod_params = f.attrs['HOD_params'][:]
    return hod_params


get_hod = get_hod_params  # for backwards compatibility


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
            a = closest_a(f.keys(), a)
            a = f'{a:.6f}'
            # load the summaries
            for stat in ['Pk', 'zPk']:
                if stat in f[a]:
                    for i in range(3):  # monopole, quadrupole, hexadecapole
                        summ[stat+str(2*i)] = {
                            'k': f[a][stat+'_k3D'][:],
                            'value': f[a][stat][:, i],
                            'log10nbar': f[a].attrs['log10nbar'],
                            'a_loaded': a}
    except (OSError, KeyError):
        return {}
    return summ


def load_lc_Pk(diag_file):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            # load the summaries
            stat = 'Pk'
            for i in range(3):  # monopole, quadrupole, hexadecapole
                summ[stat+str(2*i)] = {
                    'k': f[stat+'_k3D'][:],
                    'value': f[stat][:, i],
                    'log10nbar': f.attrs['log10nbar']}
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
            a = closest_a(f.keys(), a)
            a = f'{a:.6f}'
            # load the summaries
            for stat in ['Bk', 'Qk', 'zBk', 'zQk']:
                if stat in f[a]:
                    for i in range(1):  # just monopole
                        summ[stat+str(2*i)] = {
                            'k': f[a]['Bk_k123'][:],
                            'value': f[a][stat][i, :],
                            'log10nbar': f[a].attrs['log10nbar'],
                            'a_loaded': a}
    except (OSError, KeyError):
        return {}
    return summ


def load_lc_Bk(diag_file):
    if not os.path.exists(diag_file):
        return {}
    summ = {}
    try:
        with h5py.File(diag_file, 'r') as f:
            # load the summaries
            for stat in ['Bk', 'Qk']:
                if stat in f:
                    for i in range(1):  # just monopole
                        summ[stat+str(2*i)] = {
                            'k': f['Bk_k123'][:],
                            'value': f[stat][i, :],
                            'log10nbar': f.attrs['log10nbar']}
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


def _construct_hod_prior(configfile):
    cfg = OmegaConf.load(configfile)
    hodcfg = cfg.bias.hod
    hodmodel = lookup_hod_model(
        model=hodcfg.model if hasattr(hodcfg, "model") else None,
        assem_bias=hodcfg.assem_bias if hasattr(
            hodcfg, "assem_bias") else False,
        vel_assem_bias=hodcfg.vel_assem_bias if hasattr(
            hodcfg, "vel_assem_bias") else False,
        zpivot=hodcfg.zpivot if hasattr(
            hodcfg, "zpivot") else None
    )
    names, lower, upper, sigma, distribution = (
        hodmodel.parameters, hodmodel.lower_bound,
        hodmodel.upper_bound, hodmodel.sigma, hodmodel.distribution
    )
    # correct for unknowns
    distribution = ['uniform'] * \
        len(names) if distribution is None else distribution
    sigma = [0.] * len(names) if sigma is None else sigma
    hodprior = np.array(
        list(zip(names, distribution, lower, upper, sigma)),
        dtype=object
    )
    return hodprior


def _load_single_simulation_summaries(sourcepath, tracer, a=None, only_cosmo=False):
    # specify paths to diagnostics
    diagpath = join(sourcepath, 'diag')
    if tracer == 'galaxy':
        diagpath = join(diagpath, 'galaxies')  # oops
    elif 'lightcone' in tracer:
        diagpath = join(diagpath, f'{tracer}')
    if not os.path.isdir(diagpath):
        return [], []

    # for each diagnostics file
    filelist = ['halos.h5'] if tracer == 'halo' else os.listdir(diagpath)
    summlist, paramlist = [], []
    for f in filelist:
        diagfile = join(diagpath, f)

        # load summaries  # TODO: load other summaries
        summ = {}
        if 'lightcone' in tracer:
            summ.update(load_lc_Pk(diagfile))
            summ.update(load_lc_Bk(diagfile))
        else:
            summ.update(load_Pk(diagfile, a))
            summ.update(load_Bk(diagfile, a))
        if len(summ) == 0:
            continue  # skip empty files

        # load cosmo/hod parameters
        params = get_cosmo(sourcepath)
        if (tracer != 'halo') & (not only_cosmo):  # add HOD params
            hodparams = get_hod_params(diagfile)
            params = np.concatenate([params, hodparams], axis=0)

        # append to lists
        summlist.append(summ)
        paramlist.append(params)

    return summlist, paramlist
