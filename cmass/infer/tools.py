

import torch
import io
import os
import pickle
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import DictConfig, OmegaConf
import optuna
from typing import List
import numpy as np


# Bispectrum triangle-configuration tags, stripped when resolving a summary's
# k-cut family (e.g. zEqQk0 -> zQk).
_BK_TAGS = ('Eq', 'Sq', 'Ss', 'Is')

# Summaries which carry no k-dependence, and so take no k-cut.
_KLESS_SUMMARIES = ('nbar', 'nz')

# Names involved in the eta_vb_centrals/noise_radial degeneracy reparam
# (infer.reparam_degeneracy) -- see scripts/plot_degeneracy_reparam.py for
# the diagnostic that motivated this. eta_vb_centrals lives in the HOD block
# of theta, noise_radial in the noise block; reparam replaces them in place
# with polar coords (degen_r, degen_phi) parallel/perpendicular to the
# degeneracy, so the theta layout/length is unchanged.
DEGEN_NAME_A = 'eta_vb_centrals'
DEGEN_NAME_B = 'noise_radial'
DEGEN_NEW_NAME_R = 'degen_r'
DEGEN_NEW_NAME_PHI = 'degen_phi'

# Placeholder priors for (degen_r, degen_phi), assumed uniform for now.
# TODO: derive the actual induced prior from the eta_vb_centrals/noise_radial
# priors instead of assuming uniform.
# degen_phi = atan2(Bn, An) with An, Bn both >= 0, so [0, 90] deg is exact.
# An, Bn lie in [0, 1], so degen_r <= sqrt(2). The (r, phi) box is a tight
# bound on, but not equal to, the image of the unit square.
DEGEN_R_PRIOR_BOUNDS = (0.0, float(np.sqrt(2)))
DEGEN_PHI_PRIOR_BOUNDS = (0.0, 90.0)

# Written by preprocess next to hodprior.csv: the physical
# eta_vb_centrals/noise_radial bounds used to normalize the reparam, needed
# to invert it (hodprior.csv has eta_vb_centrals renamed to degen_r).
REPARAM_BOUNDS_FILE = 'reparam_bounds.yaml'


def saved_reparam_degeneracy(exp_path):
    """infer.reparam_degeneracy as exp_path was preprocessed, read from its
    saved config.yaml rather than the current run's cfg."""
    saved = OmegaConf.load(os.path.join(exp_path, 'config.yaml')).infer
    return bool(saved.get('reparam_degeneracy', False))


def check_reparam_degeneracy(exp_path, cfg):
    """Fail if cfg's infer.reparam_degeneracy disagrees with exp_path's."""
    saved = saved_reparam_degeneracy(exp_path)
    if saved != bool(cfg.infer.get('reparam_degeneracy', False)):
        raise ValueError(
            f'{exp_path} was preprocessed with infer.reparam_degeneracy='
            f'{saved}; set it to match.')


def reparam_degeneracy_bounds(hodprior, noiseprior):
    """Prior bounds used to normalize eta_vb_centrals/noise_radial before the
    polar (r, phi) reparameterization -- the two live on very different
    scales (noise_radial's range is ~6x wider), so A^2+B^2=const is only a
    clean circle once each axis is scaled by its own prior range."""
    hod_names = hodprior[:, 0].astype(str)
    idx = np.where(hod_names == DEGEN_NAME_A)[0]
    if len(idx) == 0:
        raise ValueError(
            f'{DEGEN_NAME_A} not found in hodprior; reparam_degeneracy '
            'requires it to be part of the inferred HOD parameters.')
    lo_a, hi_a = hodprior[idx[0], 2:4].astype(float)
    lo_b, hi_b = float(noiseprior.params.a), float(noiseprior.params.b)
    return (lo_a, hi_a), (lo_b, hi_b)


def apply_degeneracy_reparam(theta, names, bounds_a, bounds_b):
    """Replace the eta_vb_centrals and noise_radial columns of theta with a
    polar (r, phi) reparameterization of their prior-range-normalized
    values: r=sqrt(An^2+Bn^2) (perpendicular to the degeneracy, tightly
    constrained by the data) and phi=atan2(Bn,An) in degrees (parallel to
    the degeneracy, the direction the data leaves mostly unconstrained).
    Column order/length is unchanged; only the two named columns are
    overwritten, and their names are updated to match."""
    theta = np.array(theta, dtype=float)
    iA, iB = names.index(DEGEN_NAME_A), names.index(DEGEN_NAME_B)
    loA, hiA = bounds_a
    loB, hiB = bounds_b
    An = (theta[:, iA] - loA) / (hiA - loA)
    Bn = (theta[:, iB] - loB) / (hiB - loB)
    theta[:, iA] = np.sqrt(An**2 + Bn**2)
    theta[:, iB] = np.degrees(np.arctan2(Bn, An))
    new_names = list(names)
    new_names[iA] = DEGEN_NEW_NAME_R
    new_names[iB] = DEGEN_NEW_NAME_PHI
    return theta, new_names


def _is_mapping(kmax):
    return isinstance(kmax, (dict, DictConfig))


def _kcut_keys(summ):
    """Candidate mapping keys for a summary, in decreasing specificity.

    e.g. zEqQk0 -> ['zEqQk0', 'zEqQk', 'zQk', 'default']
    """
    keys = [summ]
    family = summ.rstrip('0123456789')
    if family and family != summ:
        keys.append(family)
    for tag in _BK_TAGS:
        if tag in family:
            keys.append(family.replace(tag, '', 1))
            break
    keys.append('default')
    return keys


def resolve_kmax(kmax, summ):
    """Resolve the kmax cut for a single summary.

    kmax is either a scalar (applied to every summary) or a mapping keyed by
    summary family (Pk, zQk, ...), exact summary name (zPk4), or 'default'.
    """
    if not _is_mapping(kmax):
        return kmax
    for key in _kcut_keys(summ):
        if key in kmax:
            return kmax[key]
    raise KeyError(
        f'No kmax specified for summary {summ!r} in {dict(kmax)}. Provide a '
        f'key matching one of {_kcut_keys(summ)[:-1]}, or a "default" key.')


def kcut_dirname(kmin, kmax):
    """Directory name encoding a k-cut.

    Scalar kmax reproduces the legacy name verbatim (kmin-0.0_kmax-0.4).
    Mapping kmax is encoded as
    kmin-0.0_kmax-def=0.2__zPk=0.6__zPk4=0.3 ('default' short as 'def').
    """
    if _is_mapping(kmax):
        kmax = '__'.join(
            f'{"def" if k == "default" else k}={kmax[k]}'
            for k in sorted(kmax))
    return f'kmin-{kmin}_kmax-{kmax}'


def iter_kcuts(exp):
    """Iterate over the (kmin, kmax) cuts of an experiment."""
    kmin_list = exp.kmin if 'kmin' in exp else [0.]
    kmax_list = exp.kmax if 'kmax' in exp else [0.4]
    for kmin in kmin_list:
        for kmax in kmax_list:
            yield kmin, kmax


def study_name_from_path(exp_path):
    """Recover the optuna study name (the summary combination) from a path
    of the form .../<tracer>/<summary>/<kcut>."""
    return os.path.basename(os.path.dirname(exp_path.rstrip('/')))


def split_experiments(exp_cfg):
    new_exps = []
    for exp in exp_cfg:
        for kmin, kmax in iter_kcuts(exp):
            new_exp = exp.copy()
            new_exp.kmin = [kmin]
            new_exp.kmax = [kmax]
            new_exps.append(new_exp)
    return new_exps


def prepare_loader(x, theta, device='cpu', **kwargs):
    x = torch.Tensor(x).to(device)
    theta = torch.Tensor(theta).to(device)
    dataset = TensorDataset(x, theta)
    loader = DataLoader(dataset, **kwargs)
    return loader


class CPU_Unpickler(pickle.Unpickler):
    # Unpickles a torch model saved on GPU to CPU
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def load_posterior(modelpath, device):
    # Load a posterior from a model file
    with open(modelpath, 'rb') as f:
        ensemble = CPU_Unpickler(f).load()
    ensemble = ensemble.to(device)
    for p in ensemble.posteriors:
        p.to(device)
    return ensemble


def select_top_trials(study: optuna.study.Study, n_nets: int) -> List[optuna.trial.FrozenTrial]:
    """
    Select the top N nets from an optuna study.
    """
    trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    if len(trials) == 0:
        raise ValueError('No completed trials found in the study.')

    trials = sorted(trials, key=lambda t: t.value, reverse=True)
    return trials[:n_nets]


def log2_avg(A, s=0):
    A = np.asarray(A)
    if len(A) <= s:
        return A
    idx = s + (1 << np.arange((len(A) - s).bit_length())) - 1
    idx = np.r_[np.arange(s), idx] if s > 0 else idx
    return np.add.reduceat(A, idx) / np.diff(np.append(idx, len(A)))
