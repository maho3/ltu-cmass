"""
Reparameterizations of theta, applied at preprocess time.

A reparameterization overwrites some named columns of theta in place (the
layout and length of theta are unchanged) with coordinates the posterior is
easier to learn in. Each provides forward (physical -> new), inverse
(new -> physical), the renaming, and uniform prior bounds for the new
coordinates. The physical bounds it was built with are saved next to the
preprocessed data (REPARAM_BOUNDS_FILE) so it can be inverted later.
"""

import os
import numpy as np
from omegaconf import OmegaConf

REPARAM_BOUNDS_FILE = 'reparam_bounds.yaml'


class PolarReparam:
    """Polar coordinates of two prior-range-normalized parameters (a, b):
    r = sqrt(an^2 + bn^2) and phi = atan2(bn, an) in degrees, with
    xn = (x - lo) / (hi - lo). Suited to a degeneracy along
    an^2 + bn^2 = const: r runs across it, phi along it."""

    def __init__(self, a, b, r, phi):
        self.a, self.b, self.r, self.phi = a, b, r, phi

    @property
    def prior_bounds(self):
        # an, bn in [0, 1], so phi in [0, 90] exactly and r <= sqrt(2). The
        # box bounds, but is larger than, the image of the unit square.
        # TODO: derive the induced prior instead of assuming uniform.
        return {self.r: (0.0, float(np.sqrt(2))), self.phi: (0.0, 90.0)}

    def rename(self, names):
        new = {self.a: self.r, self.b: self.phi}
        return [new.get(n, n) for n in names]

    def unrename(self, names):
        old = {self.r: self.a, self.phi: self.b}
        return [old.get(n, n) for n in names]

    def forward(self, theta, names, bounds):
        theta = np.array(theta, dtype=float)
        ia, ib = names.index(self.a), names.index(self.b)
        (loa, hia), (lob, hib) = bounds[self.a], bounds[self.b]
        an = (theta[:, ia] - loa) / (hia - loa)
        bn = (theta[:, ib] - lob) / (hib - lob)
        theta[:, ia] = np.hypot(an, bn)
        theta[:, ib] = np.degrees(np.arctan2(bn, an))
        return theta, self.rename(names)

    def inverse(self, theta, names, bounds):
        """Returns theta, names, and a mask of rows that land inside the
        physical prior box (the (r, phi) prior box is larger than it)."""
        theta = np.array(theta, dtype=float)
        ir, iphi = names.index(self.r), names.index(self.phi)
        (loa, hia), (lob, hib) = bounds[self.a], bounds[self.b]
        r, phi = theta[:, ir], np.radians(theta[:, iphi])
        an, bn = r * np.cos(phi), r * np.sin(phi)
        ok = (an >= 0) & (an <= 1) & (bn >= 0) & (bn <= 1)
        theta[:, ir] = loa + (hia - loa) * an
        theta[:, iphi] = lob + (hib - lob) * bn
        return theta, self.unrename(names), ok

    def rename_hodprior(self, hodprior):
        """hodprior with any row for a or b renamed to its new coordinate,
        under the assumed uniform prior."""
        hodprior = hodprior.copy()
        names = hodprior[:, 0].astype(str)
        for old, new in ((self.a, self.r), (self.b, self.phi)):
            for i in np.flatnonzero(names == old):
                hodprior[i] = [new, 'uniform', *self.prior_bounds[new],
                               None, None]
        return hodprior


# infer.reparam_degeneracy: eta_vb_centrals (HOD block) and noise_radial
# (noise block) roughly follow an^2 + bn^2 = const.
DEGENERACY = PolarReparam(
    'eta_vb_centrals', 'noise_radial', 'degen_r', 'degen_phi')


def degeneracy_bounds(hodprior, noiseprior):
    """Physical prior bounds of DEGENERACY's (a, b), which normalize them
    before the polar transform. They live on very different scales, so
    an^2 + bn^2 = const is only a circle once each is scaled by its own
    prior range."""
    names = hodprior[:, 0].astype(str)
    idx = np.flatnonzero(names == DEGENERACY.a)
    if len(idx) == 0:
        raise ValueError(
            f'{DEGENERACY.a} not found in hodprior; reparam_degeneracy '
            'requires it to be part of the inferred HOD parameters.')
    return {DEGENERACY.a: tuple(hodprior[idx[0], 2:4].astype(float)),
            DEGENERACY.b: (float(noiseprior.params.a),
                           float(noiseprior.params.b))}


def save_bounds(exp_path, bounds):
    OmegaConf.save(
        OmegaConf.create({k: list(map(float, v)) for k, v in bounds.items()}),
        os.path.join(exp_path, REPARAM_BOUNDS_FILE))


def load_bounds(exp_path):
    """Saved bounds, or None if exp_path has no REPARAM_BOUNDS_FILE."""
    path = os.path.join(exp_path, REPARAM_BOUNDS_FILE)
    if not os.path.exists(path):
        return None
    return {k: tuple(map(float, v)) for k, v in OmegaConf.load(path).items()}


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
