
import os
from os.path import join
import h5py
import numpy as np
import logging
from omegaconf import OmegaConf
from cmass.bias.tools.hod import lookup_hod_model


def get_cosmo(source_path):
    cfg = OmegaConf.load(join(source_path, 'config.yaml'))
    return np.array(cfg.nbody.cosmo)


def get_hod_params(diagfile):
    with h5py.File(diagfile, 'r') as f:
        hod_params = f.attrs['HOD_params'][:]
    return hod_params


def get_noise_params(diagfile):
    # This is a hack, because I accidentally saved noise properties in groups.
    # This has been fixed in recent PR
    # Kept here for compatibility (TODO: deprecate)
    with h5py.File(diagfile, 'r') as f:
        if 'noise_radial' in f.attrs:
            noise_params = [f.attrs['noise_radial'],
                            f.attrs['noise_transverse']]
        else:
            g = f[list(f.keys())[0]]
            noise_params = [g.attrs['noise_radial'],
                            g.attrs['noise_transverse']]
    return np.array(noise_params)


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
                            'nbar': f[a].attrs['nbar'],
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
                    'nbar': f.attrs['nbar'],
                    'log10nbar': f.attrs['log10nbar'],
                    'nz': f['nz'][:],
                    'nz_bins': f['nz_bins'][:],
                }
    except (OSError, KeyError):
        return {}
    return summ


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
                    for i in range(2):
                        if i >= f[a][stat].shape[0]:
                            continue
                        summ[stat+str(2*i)] = {
                            'k': f[a]['Bk_k123'][:],
                            'value': f[a][stat][i, :],
                            'nbar': f[a].attrs['nbar'],
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
                            'nbar': f.attrs['nbar'],
                            'log10nbar': f.attrs['log10nbar'],
                            'nz': f['nz'][:],
                            'nz_bins': f['nz_bins'][:]
                        }
    except (OSError, KeyError):
        return {}
    return summ


def _is_in_kminmax(k, kmin, kmax):
    k = np.atleast_2d(k)
    return np.all((kmin <= k) & (k <= kmax), axis=0)


def _filter_Pk(X, kmin, kmax):
    # filter kmin and kmax
    return np.array(
        [x['value'][_is_in_kminmax(x['k'], kmin, kmax)] for x in X])


def _get_nbar(data):
    return np.array([x['nbar'] for x in data]).reshape(-1, 1)


def _get_log10nbar(data):
    return np.repeat(
        np.array([x['log10nbar'] for x in data]).reshape(-1, 1),
        10, axis=-1
    )  # repeat for more visibility


def _get_log10nz(data):
    """
    Extracts n(z) values from each data entry and bins them into 3 coarse bins
    with edges at [0.4, 0.5, 0.6, 0.7]. Returns a 2D array of shape (len(data), 3).
    """
    num_bins = 3
    bin_edges = np.linspace(0.4, 0.7, num_bins + 1)
    binned_nz = np.empty((len(data), num_bins))
    for i, entry in enumerate(data):
        nz_values = entry['nz']
        nz_bin_edges = entry['nz_bins']
        nz_bin_centers = 0.5 * (nz_bin_edges[:-1] + nz_bin_edges[1:])
        # Assign each bin center to a coarse bin
        coarse_bin_indices = np.digitize(nz_bin_centers, bin_edges) - 1
        # Sum n(z) values in each coarse bin
        for j in range(num_bins):
            binned_nz[i, j] = np.sum(nz_values[coarse_bin_indices == j])
    # repeat for more visibility
    num_repeat = 5
    binned_nz = np.repeat(binned_nz, num_repeat, axis=-1)
    binned_nz_logged = np.where(binned_nz == 0, -1, np.log10(binned_nz))
    return binned_nz_logged  # shape: (num_entries, num_bins*num_repeat)


def signed_log(x, base=10):
    # Compute the signed logarithm of x (for negative values)
    return np.sign(x) * np.log1p(np.abs(x)) / np.log(base)


def preprocess_Pk(data, kmax, monopole=True, norm=None, kmin=0., correct_shot=False):
    # process Pk: filtering for k's, normalizing, and flattening
    if not monopole and norm is None:
        raise ValueError('norm must be provided when monopole is False')

    X = _filter_Pk(data, kmin, kmax)

    if monopole:
        if correct_shot:
            X -= 1./_get_nbar(data)  # subtract shot noise
        X = signed_log(X)
    else:
        Xnorm = _filter_Pk(norm, kmin, kmax)
        X /= Xnorm

    return np.nan_to_num(X, nan=0.0).reshape(len(X), -1)


def _is_valid_triangle(k):
    # Return mask of triangles satisfying triangle inequality: k1 < k2 + k3
    k1, k2, k3 = k
    return (k1 < k2 + k3) & (k2 < k1 + k3) & (k3 < k1 + k2)


def _is_squeezed(k):
    # Return mask of squeezed triangles: k1 < k2 + k3 and k1 < 0.5 * (k2 + k3)
    k1, k2, k3 = k
    return np.isclose(k1, k2) & (k3 < k2)


def _is_isoceles(k):
    # Return mask of isoceles triangles
    k1, k2, k3 = k
    return np.isclose(k1, k2) | np.isclose(k2, k3) | np.isclose(k1, k3)


def _is_equilateral(k):
    # Return mask of equilateral triangles
    k1, k2, k3 = k
    return np.isclose(k1, k2) & np.isclose(k2, k3)


def _is_subsampled(k):
    # Return mask where only every 5th k is True
    return np.arange(len(k[0])) % 5 == 0


def _filter_Bk(X, kmin, kmax, equilateral=False, squeezed=False,
               subsampled=False, isoceles=False):
    if sum([equilateral, squeezed, subsampled]) > 1:
        raise ValueError(
            "Only one of equilateral, squeezed, or subsampled can be True.")
    if equilateral:
        return np.array(
            [x['value'][_is_in_kminmax(x['k'], kmin, kmax) & _is_equilateral(x['k'])]
             for x in X])
    elif squeezed:
        return np.array(
            [x['value'][_is_in_kminmax(x['k'], kmin, kmax) & _is_squeezed(x['k'])]
             for x in X])
    elif subsampled:
        return np.array(
            [x['value'][_is_in_kminmax(x['k'], kmin, kmax) & _is_subsampled(x['k'])]
             for x in X])
    elif isoceles:
        return np.array(
            [x['value'][_is_in_kminmax(x['k'], kmin, kmax) & _is_isoceles(x['k'])]
             for x in X])
    else:
        return np.array(
            [x['value'][_is_in_kminmax(x['k'], kmin, kmax) & _is_valid_triangle(x['k'])]
             for x in X])


def preprocess_Bk(data, kmax, kmin=0., log=False,
                  equilateral_only=False, squeezed_only=False,
                  subsampled_only=False, isoceles_only=False,
                  correct_shot=False):
    # process Bk: filtering for k's, normalizing, and flattening
    X = _filter_Bk(data, kmin, kmax, equilateral_only,
                   squeezed_only, subsampled_only, isoceles_only)

    if correct_shot:  # Eq. 44 arxiv:1610.06585
        pass  # not implemented because I'm not sure its right

    if log:
        X = signed_log(X)

    return np.nan_to_num(X, nan=0.0).reshape(len(X), -1)


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


def _construct_noise_prior(sourcepath, tracer):
    """
    This goes into a diagnostic file, reads the noise_dist from the attributes,
    and returns a noise prior. This is kind of a hack, because our current 
    diagnostics don't save the prior params, just the distribution.

    TODO: fix this hack
    """
    diagpath = join(sourcepath, 'diag')
    if tracer == 'galaxy':
        diagpath = join(diagpath, 'galaxies')
    elif 'lightcone' in tracer:
        diagpath = join(diagpath, f'{tracer}')
    filelist = ['halos.h5'] if tracer == 'halo' else os.listdir(diagpath)
    with h5py.File(join(diagpath, filelist[0]), 'r') as f:
        noisedist = f.attrs['noise_dist'] if 'noise_dist' in f.attrs else None
    # This is a hack, because I accidentally saved noise properties in groups.
    # This has been fixed in recent PR
    # Kept here for compatibility (TODO: deprecate)
    with h5py.File(join(diagpath, filelist[0]), 'r') as f:
        noisedist = None
        if 'noise_dist' in f.attrs:
            noisedist = f.attrs['noise_dist']
        else:
            g = f[list(f.keys())[0]]
            if 'noise_dist' in g.attrs:
                noisedist = g.attrs['noise_dist']
    if noisedist is None:
        raise ValueError(
            f'No noise distribution found in {join(diagpath, filelist[0])}.')

    # this is a terrible hack to find the config files
    codepath = os.path.abspath(__file__)
    confpath = join(os.path.dirname(os.path.dirname(codepath)), 'conf', 'noise')
    filepath = join(confpath, noisedist.lower()+'.yaml')
    try:
        noiseprior = OmegaConf.load(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Noise prior configuration file not found: {filepath}')
    return noiseprior


def _load_single_simulation_summaries(sourcepath, tracer, a=None,
                                      include_hod=True, include_noise=False):
    # specify paths to diagnostics
    diagpath = join(sourcepath, 'diag')
    if tracer == 'galaxy':
        diagpath = join(diagpath, 'galaxies')
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
        if (tracer != 'halo') & (include_hod):  # add HOD params
            hodparams = get_hod_params(diagfile)
            params = np.concatenate([params, hodparams], axis=0)

        if include_noise:  # add noise params
            try:
                noise_params = get_noise_params(diagfile)
                params = np.concatenate([params, noise_params], axis=0)
            except Exception as e:
                logging.error(f'Issue with {diagfile}')
                raise e

        # append to lists
        summlist.append(summ)
        paramlist.append(params)

    return summlist, paramlist
