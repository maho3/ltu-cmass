

import torch
import io
import os
import pickle
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import DictConfig
import optuna
from typing import List
import numpy as np


# Bispectrum triangle-configuration tags, stripped when resolving a summary's
# k-cut family (e.g. zEqQk0 -> zQk).
_BK_TAGS = ('Eq', 'Sq', 'Ss', 'Is')

# Summaries which carry no k-dependence, and so take no k-cut.
_KLESS_SUMMARIES = ('nbar', 'nz')


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
