import warnings
warnings.filterwarnings('ignore')
import matplotlib.colors as mcolors
import os
import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
import logging
import torch
import torch.nn as nn
import time
import numpy as np
import sys

from ili.embedding import FCN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)


def run_inference(x_train, theta_train, cfg, out_dir):
    print(x_train.min(), theta_train.max())
    start = time.time()

    # select the network configuration
    mcfg = cfg.net
    logging.info(f'Using network architecture: {mcfg}')

    # define a prior
    prior = ili.utils.Uniform(low=[0.1, 0.02, 0.5, 0.8, 0.6], high=[0.5, 0.08, 0.9, 1.2, 1], device=device)

    # define an embedding network
    if mcfg.fcn_depth == 0:
        embedding = nn.Identity()
    else:
        embedding = FCN(
            n_hidden=[mcfg.fcn_width]*mcfg.fcn_depth,
            act_fn='ReLU'
        )

    # instantiate your neural networks to be used as an ensemble
    if cfg.backend == 'lampe':
        net_loader = ili.utils.load_nde_lampe
        extra_kwargs = {}
    elif cfg.backend == 'sbi':
        net_loader = ili.utils.load_nde_sbi
        extra_kwargs = {'engine': cfg.engine}
    else:
        raise NotImplementedError
    kwargs = {k: v for k, v in mcfg.items() if k in [
        'model', 'hidden_features', 'num_transforms', 'num_components']}
    nets = [net_loader(**kwargs, **extra_kwargs, embedding_net=embedding)]

    # define training arguments
    bs, lr = cfg.batch_size, cfg.learning_rate
    bs = mcfg.batch_size if bs is None else bs
    lr = mcfg.learning_rate if lr is None else lr
    print(cfg)
    print(mcfg)
    train_args = {
        'learning_rate': lr,
        'stop_after_epochs': cfg.stop_after_epochs,
        'validation_fraction': cfg.val_frac,
        'lr_decay_factor': cfg.lr_decay_factor,
        'lr_patience': cfg.lr_patience,
    }

    # setup data loaders
    # x_train, x_val,theta_train,theta_val = train_test_split(x_train, theta_train, test_size=0.1, random_state=0)
    loader = NumpyLoader(x=x_train, theta=theta_train)

    # initialize the trainer
    runner = InferenceRunner.load(
        backend=cfg.backend,
        engine=cfg.engine,
        prior=prior,
        nets=nets,
        device=cfg.device,
        train_args=train_args,
        out_dir=None
    )

    # train the model
    posterior, histories = runner(loader=loader)

    return posterior, histories


def main(net_idx):

    path = '/anvil/scratch/x-abairagi/cmass-ili/quijote/nbody/models/galaxy/Pk0+Patches/'
    train_cfg = OmegaConf.load('/home/x-abairagi/ltu-cmass/cmass/conf/infer/default.yaml')
    train_cfg.save_dir = path+'nets/'
    train_cfg.net_index = net_idx
    nets_dict = OmegaConf.load('/home/x-abairagi/ltu-cmass/cmass/conf/net/tuning.yaml')
    train_cfg.net = nets_dict[train_cfg.net_index]

    # nzs = np.load('/home/x-csui/workspace/cmass/cmass_data/dataset/nz_hod_inference/nzs.npy')
    # params = np.load('/home/x-csui/workspace/cmass/cmass_data/dataset/nz_hod_inference/hod_params.npy')
    #train test split
    # nzs_train, nzs_test,params_train,params_test = train_test_split(nzs, params, test_size=0.1, random_state=0)
    
    x_train = np.load(path+'x_train.npy') 
    # mean=x_train.mean(axis=0)
    # std=x_train.std(axis=0)
    # x_train=(x_train-mean)/std
    params_train = np.load(path+'theta_train.npy')[:,:5]
     
    pos, his = run_inference(x_train, params_train, train_cfg, train_cfg.save_dir)
    torch.save(pos, os.path.join(train_cfg.save_dir, f'posterior_{net_idx}.pth'))


if __name__ == '__main__':
    model_idx = int(sys.argv[1])
    main(model_idx)
