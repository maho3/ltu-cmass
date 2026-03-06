

import torch
import io
import pickle
from torch.utils.data import TensorDataset, DataLoader
import optuna
from typing import List


def split_experiments(exp_cfg):
    new_exps = []
    for exp in exp_cfg:
        kmin_list = exp.kmin if 'kmin' in exp else [0.]
        kmax_list = exp.kmax if 'kmax' in exp else [0.4]
        for kmin in kmin_list:
            for kmax in kmax_list:
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
