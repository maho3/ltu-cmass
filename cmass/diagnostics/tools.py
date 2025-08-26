"""Tools and routiines for diagnostics."""

import os
import numpy as np
import logging
import h5py
import datetime
from ..survey.tools import sky_to_unit_vectors
from ..utils import save_configuration_h5


def _delete(group, to_delete):
    for key in group.keys():
        if isinstance(group[key], h5py.Group):  # if its another group
            _delete(group[key], to_delete)
        elif key in to_delete:  # if its a dataset
            del group[key]


def _check(group, to_check):
    # checks that all to_check keys are in group or its subgroups
    saved_keys = list(group.keys())

    # check if empty
    if len(saved_keys) == 0:
        return False

    # check recursively
    if isinstance(group[saved_keys[0]], h5py.Group):
        computed = True
        for key in saved_keys:
            if isinstance(group[key], h5py.Group):  # if its another group
                computed &= _check(group[key], to_check)
        return computed

    # check if all keys are in group
    for key in to_check:
        if key not in saved_keys:
            return False
    return True


def _get_snapshot_alist(filename, focus_z=None):
    # load data file and get keys
    with h5py.File(filename, 'r') as f:
        alist = list(f.keys())

    # Filter alist to only include the closest to a specified redshift
    if focus_z is not None:
        i = np.argmin(np.abs(np.array(alist, dtype=float) - 1./(1 + focus_z)))
        alist = [alist[i]]
    return alist


def check_existing(file, summaries, from_scratch=False, rsd=False):
    if not os.path.isfile(file):
        return summaries

    # Check if summaries are already saved, and may remove them if from_scratch
    to_compute = []

    for s in summaries:
        # which keys to check for
        if s == 'Pk':
            to_check = ['Pk_k3D', 'Pk']
        elif s == 'Bk':
            to_check = ['Bk_k123', 'Bk', 'Qk', 'bPk_k3D', 'bPk']
        else:
            raise NotImplementedError(f'Summary {s} not yet implemented')

        if rsd:  # check for redshift space
            to_check += [f'z{s}' for s in to_check]

        # check if keys exist in the file
        with h5py.File(file, 'r') as f:
            computed = _check(f, to_check)

        # if already computed
        if computed and (not from_scratch):
            logging.info(f'{s} summaries already computed. Skipping...')
            continue

        # if not computed or from_scratch
        if not computed:
            logging.info(f'{s} summaries not fully computed. '
                         f'Running {s} from scratch...')
        else:
            logging.info(f'{s} already computed, but from_scratch=True. '
                         f'Running {s} from scratch...')
        with h5py.File(file, 'a') as f:
            _delete(f, to_check)
        to_compute.append(s)
    return to_compute


def get_mesh_resolution(L, high_res=False, use_ngp=False):
    # set mesh resolution and mass assignment scheme
    N = int((L/1000)*128)  # 128 cells per 1000 Mpc/h
    if high_res:
        N *= 2  # double resolution
    MAS = 'NGP' if use_ngp else 'TSC'
    return N, MAS


def noise_positions(pos, ra, dec,
                    noise_radial, noise_transverse):
    """Applies observational noise to positions."""
    r_hat, e_phi, e_theta = sky_to_unit_vectors(ra, dec)
    noise = np.random.randn(*pos.shape)
    pos += r_hat * noise[:, 0, None] * noise_radial
    pos += e_phi * noise[:, 1, None] * noise_transverse
    pos += e_theta * noise[:, 2, None] * noise_transverse
    return pos


def save_group(file, data, attrs=None, a=None, config=None,
               save_HOD=False, save_noise=False):
    logging.info(f'Saving {len(data)} datasets to {file}')
    with h5py.File(file, 'a') as f:
        if a is not None:
            group = f.require_group(a)
        else:
            group = f
        if attrs is not None:
            for key, value in attrs.items():
                group.attrs[key] = value
            group.attrs['timestamp'] = datetime.datetime.now().isoformat()
        for key, value in data.items():
            if key in group:
                del group[key]
            group.create_dataset(key, data=value)

        if config is not None:
            save_configuration_h5(
                f, config, save_HOD=save_HOD, save_noise=save_noise)


# Summarizer functions

def get_binning(summary, L, N, threads, rsd=False):
    ells = [0,] if not rsd else [0, 2, 4]
    if summary == 'Pk':
        return {
            'k_edges': np.linspace(0, 1., 31),
            'n_mesh': N,
            'los': 'z',
            'compensations': 'ngp',
            'ells': ells,
        }
    if summary == 'Bk':
        k_min = 1.05*2 * np.pi / L
        n_mesh = 64
        k_max = 0.95 * np.pi * n_mesh / L
        num_bins = 15
        return {
            'k_bins': np.logspace(np.log10(k_min), np.log10(k_max), num_bins),
            'n_mesh': n_mesh,
            'lmax': 2,
            'ells': ells,
        }
    if summary == 'TwoPCF':
        num_bins = 60
        return {
            'r_bins': np.logspace(-2, np.log10(150.), num_bins),
            'mu_bins': np.linspace(-1., 1., 201),
            'ells': ells,
            'n_threads': threads,
            'los': 'z',
        }
    if summary == 'WST':
        num_bins = 60
        return {
            'J_3d': 3,
            'L_3d': 3,
            'integral_powers': [0.8,],
            'sigma': 0.8,
            'n_mesh': N,
        }
    if summary == 'DensitySplit':
        num_bins = 60
        return {
            'r_bins': np.logspace(-1, np.log10(150.), num_bins),
            'mu_bins': np.linspace(-1., 1., 201),
            'n_quantiles': 5,
            'smoothing_radius': 10.0,
            'ells': ells,
            'n_threads': threads,
        }
    if summary == 'KNN':
        num_bins = 60
        return {
            'r_bins': np.logspace(-2, np.log10(30.), num_bins),
            'k': [1, 3, 7, 11],
            'n_threads': threads,
        }
    else:
        raise NotImplementedError(f'{summary} not implemented')


def store_summary(
    catalog, group, summary_name,
    box_size, num_bins, num_threads, use_rsd=False
):
    # get summary binning
    binning_config = get_binning(
        summary_name, box_size, num_bins, num_threads, rsd=use_rsd)

    logging.info(f'Computing Summary: {summary_name}')

    # compute summary
    import summarizer  # only import if needed. TODO: get working
    summary_function = getattr(summarizer, summary_name)(**binning_config)
    summary_data = summary_function(catalog)

    # store summary
    summary_dataset = summary_function.to_dataset(summary_data)
    for coord_name, coord_value in summary_dataset.coords.items():
        dataset_key = f"{'z' if use_rsd else ''}{summary_name}_{coord_name}"
        group.create_dataset(dataset_key, data=coord_value.values)
    summary_key = summary_name if not use_rsd else f'z{summary_name}'
    group.create_dataset(summary_key, data=summary_dataset.values)
