"""
Sample halos from the density field using a bias model. Assumes
a continuous Poissonian distribution of halos by interpolating
between the grid points of the density field

Requires:
    - scipy

Input:
    - rho: density field
    - ppos: particle positions
    - pvel: particle velocities
    - popt: bias parameters
    - medges: mass bin edges

Output:
    - xtrues: halo positions
    - vtrues: halo velocities
    - mtrues: halo masses
"""

import numpy as np
import tqdm
from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsRegressor
import argparse
import logging
from os.path import join as pjoin

from tools.shared_code import TruncatedPowerLaw, sample_3d
from tools.utils import get_global_config, get_logger, timing_decorator

logger = logging.getLogger(__name__)


def load_bias_params(bias_path, lhid):
    # load the bias parameters for using the 1 Gpc Quijote sims
    popt = np.load(pjoin(bias_path, f'{lhid}.npy'))
    medges = np.load(pjoin(bias_path, 'medges.npy'))
    return popt, medges


@timing_decorator
def load_borg(source_dir):
    rho = np.load(pjoin(source_dir, 'rho.npy'))
    ppos = np.load(pjoin(source_dir, 'ppos.npy'))
    pvel = np.load(pjoin(source_dir, 'pvel.npy'))
    return rho, ppos, pvel


@timing_decorator
def pad(ppos, pvel, Lbox, Lpad):
    # pad the 3d particle cube with periodic boundary conditions

    def offset(*inds):
        # calculate the offset vector for a given region
        out = np.zeros(3)
        out[list(inds)] = Lbox
        return out

    def recursive_padding(ipos, idig, ivel, index=0):
        # recursively pad the cube
        if index >= 3:
            return []
        padded = []
        for i, dir in [(1, 1), (3, -1)]:
            mask = idig[:, index] == i
            ippad, ivpad = ipos[mask] + dir*offset(index), ivel[mask]
            padded += [(ippad, ivpad)]
            padded += recursive_padding(ippad, idig[mask], ivpad, index+1)
            padded += recursive_padding(ippad, idig[mask], ivpad, index+2)
        return padded

    regions = np.digitize(ppos, bins=[0, Lpad, Lbox - Lpad, Lbox])
    padlist = [(ppos, pvel)]
    padlist += recursive_padding(ppos, regions, pvel, index=0)
    padlist += recursive_padding(ppos, regions, pvel, index=1)
    padlist += recursive_padding(ppos, regions, pvel, index=2)
    padpos, padvel = zip(*padlist)
    padpos = np.concatenate(padpos)
    padvel = np.concatenate(padvel)

    logging.info(
        f'len(pad)/len(original): {len(padpos)}/{len(ppos)}'
        f' = {len(padpos)/len(ppos):.3f}')
    return padpos, padvel


@timing_decorator
def sample_counts(rho, popt):
    # sample the halo counts from the bias model
    law = TruncatedPowerLaw()
    return np.stack([law._get_mean_ngal(rho, *popt[i]) for i in range(10)],
                    axis=-1)
    # return np.stack([law.sample(rho, popt[i]) for i in range(10)], axis=-1)


@timing_decorator
def sample_positions(hsamp):
    # sample the positions from the halo counts
    xtrues = []
    for i in range(10):
        xtrue, _, _ = sample_3d(
            hsamp[..., i],
            np.sum(hsamp[..., i]).astype(int),
            3000, 0, np.zeros(3))
        xtrues.append(xtrue.T)
    return xtrues


@timing_decorator
def sample_velocities(xtrues, ppos, pvel):
    knn = KNeighborsRegressor(
        n_neighbors=5, leaf_size=1000,
        algorithm='ball_tree', weights='distance', n_jobs=-1)
    knn.fit(ppos, pvel)
    vtrues = [knn.predict(x) for x in tqdm.tqdm(xtrues)]
    return vtrues


@timing_decorator
def sample_masses(Nsamp, medges):
    # sample the masses from the mass bins
    mtrues = []
    for i in range(len(medges)-1):
        im = np.random.uniform(*medges[i:i+2], size=Nsamp[i])
        mtrues.append(im)
    return mtrues


def main():
    # Load global configuration
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--simtype', type=str, default='borg-quijote')
    args = parser.parse_args()

    logging.info(f'Running with lhid={args.lhid}...')
    logging.info('Loading bias parameters...')
    bias_path = pjoin(glbcfg['wdir'], 'quijote/bias_fit/LH_n=128')
    popt, medges = load_bias_params(bias_path, args.lhid)

    logging.info('Loading 3 Gpc sims...')
    source_dir = pjoin(
        glbcfg['wdir'], f'{args.simtype}/latin_hypercube_HR-L3000-N384',
        f'{args.lhid}')
    rho, ppos, pvel = load_borg(source_dir)

    logging.info('Padding...')
    ppos, pvel = pad(ppos, pvel, Lbox=3000, Lpad=10)

    logging.info('Sampling power law...')
    hsamp = sample_counts(rho, popt)

    logging.info('Sampling halo positions as a Poisson field...')
    xtrues = sample_positions(hsamp)

    logging.info('Calculating velocities...')
    vtrues = sample_velocities(xtrues, ppos, pvel)

    logging.info('Sampling masses...')
    mtrues = sample_masses([len(x) for x in xtrues], medges)

    logging.info('Combine...')
    xtrues = np.concatenate(xtrues, axis=0)
    vtrues = np.concatenate(vtrues, axis=0)
    mtrues = np.concatenate(mtrues, axis=0)

    logging.info('Saving cube...')
    np.save(pjoin(source_dir, 'halo_pos.npy'), xtrues)
    np.save(pjoin(source_dir, 'halo_vel.npy'), vtrues)
    np.save(pjoin(source_dir, 'halo_mass.npy'), mtrues)

    logging.info('Done!')


if __name__ == '__main__':
    main()
