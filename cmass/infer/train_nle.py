"""
Trains a Neural Likelihood Estimator (NLE) — a diagonal-Gaussian emulator
that learns p(x | theta) with independent per-dimension variance.

This is a dev tool for studying how input parameters affect the data vector.
It reads the same preprocessed data produced by cmass.infer.preprocess and
writes nle.pt and loss.png into the same net directory as the NPE's
posterior.pkl, without overwriting it.

Usage (identical CLI syntax to train.py):
    python -m cmass.infer.train_nle suite=... sim=... lhid=...

Reloading in a notebook:
    import torch
    from cmass.infer.train_nle import DiagonalGaussianNLE
    ckpt = torch.load('nle.pt', weights_only=False)
    model = DiagonalGaussianNLE(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    # predicted mean and std:
    #   mean, std = model.predict(theta_tensor)
    # log p(x | theta):
    #   model.log_prob(x_tensor, theta_tensor)
    # sample x given theta:
    #   model.sample(200, theta_tensor)
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from .tools import split_experiments
from ..utils import timing_decorator, clean_up
from ..nbody.tools import parse_nbody_config


log = logging.getLogger(__name__)

# ── Hardcoded hyperparameters ────────────────────────────────────────────────
HIDDEN_FEATURES = 128
HIDDEN_DEPTH = 3
N_EPOCHS = 500
BATCH_SIZE = 256
LR = 1e-3


class DiagonalGaussianNLE(nn.Module):
    """MLP that predicts mean and diagonal log-variance of p(x | theta).

    Args:
        D_x: dimensionality of the data vector x.
        D_theta: dimensionality of the parameter vector theta.
        hidden_features: width of each hidden layer.
        hidden_depth: number of hidden layers.
    """

    def __init__(self, D_x, D_theta, hidden_features=128, hidden_depth=3):
        super().__init__()
        layers = []
        in_dim = D_theta
        for _ in range(hidden_depth):
            layers += [nn.Linear(in_dim, hidden_features), nn.ReLU()]
            in_dim = hidden_features
        self.backbone = nn.Sequential(*layers)
        self.head_mean = nn.Linear(in_dim, D_x)
        self.head_logvar = nn.Linear(in_dim, D_x)

    def forward(self, theta):
        """Return (mean, log_var) each of shape (..., D_x)."""
        h = self.backbone(theta)
        return self.head_mean(h), self.head_logvar(h)

    def predict(self, theta):
        """Return (mean, std) as tensors."""
        mean, logvar = self.forward(theta)
        return mean, torch.exp(0.5 * logvar)

    def log_prob(self, x, theta):
        """Diagonal-Gaussian log p(x | theta), shape (...,)."""
        mean, logvar = self.forward(theta)
        return -0.5 * (
            mean.shape[-1] * np.log(2 * np.pi)
            + logvar.sum(dim=-1)
            + ((x - mean) ** 2 / torch.exp(logvar)).sum(dim=-1)
        )

    def sample(self, n, theta):
        """Sample n data vectors for each row in theta.

        Args:
            n: number of samples.
            theta: (batch, D_theta) or (D_theta,).

        Returns:
            Tensor of shape (n, batch, D_x) or (n, D_x).
        """
        mean, logvar = self.forward(theta)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn((n,) + mean.shape, device=mean.device)
        return mean.unsqueeze(0) + eps * std.unsqueeze(0)


def load_nle(path, device='cpu'):
    """Load a trained DiagonalGaussianNLE from a checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DiagonalGaussianNLE(**ckpt['model_kwargs']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def _train(model, x_train, theta_train, x_val, theta_val, device='cpu'):
    """Train the NLE and return per-epoch train/val losses."""
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(theta_train, dtype=torch.float32))
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(theta_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_losses, val_losses = [], []

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss, n = 0.0, 0
        for xb, tb in train_loader:
            xb, tb = xb.to(device), tb.to(device)
            optimizer.zero_grad()
            loss = -model.log_prob(xb, tb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            n += len(xb)
        train_losses.append(epoch_loss / n)

        model.eval()
        epoch_loss, n = 0.0, 0
        with torch.no_grad():
            for xb, tb in val_loader:
                xb, tb = xb.to(device), tb.to(device)
                loss = -model.log_prob(xb, tb).mean()
                epoch_loss += loss.item() * len(xb)
                n += len(xb)
        val_losses.append(epoch_loss / n)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            log.info(
                f'Epoch {epoch+1:4d}/{N_EPOCHS}  '
                f'train_loss={train_losses[-1]:.4f}  '
                f'val_loss={val_losses[-1]:.4f}')

    return train_losses, val_losses


def _save(model, D_x, D_theta, out_path):
    """Save model checkpoint with enough info to reload."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_kwargs': {
            'D_x': D_x,
            'D_theta': D_theta,
            'hidden_features': HIDDEN_FEATURES,
            'hidden_depth': HIDDEN_DEPTH,
        },
    }, out_path)
    log.info(f'Saved NLE checkpoint to {out_path}')


def _plot_loss(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(train_losses, label='train', lw=1)
    ax.plot(val_losses, label='validation', lw=1)
    ax.set(xlabel='Epoch', ylabel='Negative log-likelihood')
    ax.legend()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    log.info(f'Saved loss plot to {out_path}')


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:
    cfg = parse_nbody_config(cfg)
    model_dir = join(cfg.meta.wdir, cfg.nbody.suite, cfg.sim, 'models')
    if cfg.infer.save_dir is not None:
        model_dir = cfg.infer.save_dir
    if cfg.infer.exp_index is not None:
        cfg.infer.experiments = split_experiments(cfg.infer.experiments)
        cfg.infer.experiments = [cfg.infer.experiments[cfg.infer.exp_index]]

    log.info('Running NLE training with config:\n' + OmegaConf.to_yaml(cfg))

    device = cfg.infer.device
    tracer = cfg.infer.tracer

    for exp in cfg.infer.experiments:
        assert len(exp.summary) > 0, 'No summaries provided'
        name = '+'.join(exp.summary)
        save_path = join(model_dir, tracer, name)

        kmin_list = exp.kmin if 'kmin' in exp else [0.]
        kmax_list = exp.kmax if 'kmax' in exp else [0.4]

        for kmin in kmin_list:
            for kmax in kmax_list:
                log.info(
                    f'NLE training for {name} with {kmin} <= k <= {kmax}')
                exp_path = join(save_path, f'kmin-{kmin}_kmax-{kmax}')

                try:
                    x_train = np.load(join(exp_path, 'x_train.npy'))
                    theta_train = np.load(join(exp_path, 'theta_train.npy'))
                    x_val = np.load(join(exp_path, 'x_val.npy'))
                    theta_val = np.load(join(exp_path, 'theta_val.npy'))
                except FileNotFoundError:
                    log.error(
                        f'Missing preprocessed data in {exp_path}. '
                        'Run cmass.infer.preprocess first.')
                    continue

                log.info(
                    f'x_train {x_train.shape}, theta_train {theta_train.shape}')
                log.info(
                    f'x_val {x_val.shape}, theta_val {theta_val.shape}')

                D_x = x_train.shape[1]
                D_theta = theta_train.shape[1]

                out_dir = join(
                    exp_path, 'nets', f'net-{cfg.infer.net_index}')
                os.makedirs(out_dir, exist_ok=True)

                model = DiagonalGaussianNLE(
                    D_x, D_theta,
                    hidden_features=HIDDEN_FEATURES,
                    hidden_depth=HIDDEN_DEPTH,
                ).to(device)
                train_losses, val_losses = _train(
                    model, x_train, theta_train, x_val, theta_val,
                    device=device)

                _save(model, D_x, D_theta, join(out_dir, 'nle.pt'))
                _plot_loss(
                    train_losses, val_losses, join(out_dir, 'nle_loss.png'))


if __name__ == "__main__":
    main()
