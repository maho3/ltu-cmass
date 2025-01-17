

import torch
import io
import pickle


def split_experiments(exp_cfg):
    new_exps = []
    for exp in exp_cfg:
        for kmax in exp.kmax:
            new_exp = exp.copy()
            new_exp.kmax = [kmax]
            new_exps.append(new_exp)
    return new_exps


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
