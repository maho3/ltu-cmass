import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree
import argparse
import logging

from os.path import join as pjoin
from tools.freecode import load_quijote_halos, TruncatedPowerLaw, sample_3d
from tools.utils import get_global_config, get_logger, timing_decorator

logger = logging.getLogger(__name__)


def load_hhalos(snapdir):
    pos_h, mass, vel_h, Npart = load_quijote_halos(snapdir)
    posm = np.concatenate([pos_h, np.log10(mass)[:, None]], axis=1)
    h, edges = np.histogramdd(
        posm,
        (128, 128, 128, 10),
        range=[(0, 1e3)]*3+[(12.8, 15.8)]
    )
    return h, edges


def load_borg(source_dir):
    rho = np.load(pjoin(source_dir, 'rho.npy'))
    ppos = np.load(pjoin(source_dir, 'ppos.npy'))
    pvel = np.load(pjoin(source_dir, 'pvel.npy'))
    return rho, ppos, pvel


@timing_decorator
def fit_bias_params(load_dir):
    logging.info('Loading 1 Gpc sims...')
    rho1g = np.load(pjoin(load_dir, 'df_m_128_z=0.npy'))
    hhalos1g, edges = load_hhalos(load_dir)

    logging.info('Fitting power law...')
    law = TruncatedPowerLaw()
    popt = np.zeros((10, 4))
    for i in range(10):
        popt[i] = law.fit(rho1g.flatten(), hhalos1g[..., i].flatten())
    return popt, edges[-1]


@timing_decorator
def main():
    # Load global configuration
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    args = parser.parse_args()

    logging.info(f'Running with lhid={args.lhid}...')

    load_dir = pjoin(glbcfg['wdir'],
                     f'quijote/density_field/latin_hypercube/{args.lhid}')
    popt, medges = fit_bias_params(load_dir)

    logging.info('Loading 3 Gpc sims...')
    source_dir = pjoin(
        glbcfg['wdir'], 'borg-quijote/latin_hypercube_HR-L3000-N384',
        f'{args.lhid}')
    rho, ppos, pvel = load_borg(source_dir)

    logging.info('Building KDE tree...')
    tree = KDTree(ppos)  # todo: account for periodic boundary conditions

    logging.info('Sampling power law...')
    law = TruncatedPowerLaw()
    hsamp = np.stack([law.sample(rho, popt[i]) for i in range(10)], axis=-1)

    logging.info('Sampling halos in Poisson field...')
    xtrues = []
    for i in range(10):
        xtrue, _, _ = sample_3d(
            hsamp[..., i],
            np.sum(hsamp[..., i]).astype(int),
            3000, 0, np.zeros(3))
        xtrues.append(xtrue.T)

    logging.info('Calculating velocities...')
    k = 5
    vtrues = []
    for i in range(10):
        print(i)
        _, nns = tree.query(xtrues[i], k)
        vnns = pvel[nns.reshape(-1)].reshape(-1, k, 3)
        vtrues.append(np.mean(vnns, axis=1))

    logging.info('Sampling masses...')
    mtrues = []
    for i in range(len(medges)-1):
        im = np.random.uniform(*medges[i:i+2], size=len(xtrues[i]))
        mtrues.append(im)

    logging.info('Combine...')
    xtrues = np.concatenate(xtrues, axis=0)
    vtrues = np.concatenate(vtrues, axis=0)
    mtrues = np.concatenate(mtrues, axis=0)

    logging.info('Saving...')
    np.save(pjoin(source_dir, 'halo_pos.npy'), xtrues)
    np.save(pjoin(source_dir, 'halo_vel.npy'), vtrues)
    np.save(pjoin(source_dir, 'halo_mass.npy'), mtrues)

    logging.info('Done!')


if __name__ == '__main__':
    main()
