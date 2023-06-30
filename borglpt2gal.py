import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree
import argparse

from os.path import join as pjoin
from tools.utils import *



parser = argparse.ArgumentParser()
parser.add_argument('--cind', type=int, required=True)
args = parser.parse_args()

lhid = args.cind
print(f'Running with lhid={lhid}...')


wdir = '/home/mattho/data'
source_dir = pjoin(wdir, f'cmass-ili/borg-quijote/latin_hypercube_HR-L3000-N384/{lhid}')


def load_rho(snid):
    rho = np.load(pjoin(wdir, 
                        f'quijote/density_field/latin_hypercube/{snid}',
                        'df_m_128_z=0.npy'))
    return rho
def load_hhalos(snid):
    snapdir = f'/home/mattho/data/quijote/Halos/latin_hypercube/{snid}'
    pos_h, mass, vel_h, Npart = load_quijote_halos(snapdir)
    posm = np.concatenate([pos_h, np.log10(mass)[:,None]], axis=1)
    h, edges = np.histogramdd(
        posm, 
        (128,128,128,10),
        range=[(0,1e3)]*3+[(12.8,15.8)]
    )
    return h, edges
def load_borg(snid):
    snapdir = pjoin(wdir, f'cmass-ili/borg-quijote/latin_hypercube_HR-L3000-N384/{snid}')
    rho = np.load(pjoin(snapdir, 'rho.npy'))
    ppos = np.load(pjoin(snapdir, 'ppos.npy'))
    pvel = np.load(pjoin(snapdir, 'pvel.npy'))
    return rho, ppos, pvel

print('Loading 1 Gpc sims...')
rho1g = load_rho(lhid)
hhalos1g, edges = load_hhalos(lhid)


print('Fitting Power Law...')
law = TruncatedPowerLaw()
def f(ind):
    popt = law.fit(rho1g.flatten(), hhalos1g[...,ind].flatten())
    return popt
popt = np.array(list(map(f, range(10))))

del rho1g
del hhalos1g


print('Loading 3 Gpc sims...')
rho, ppos, pvel = load_borg(lhid)
tree = KDTree(ppos)  # todo: account for periodic boundary conditions


print('Building KDE tree...')
tree = KDTree(ppos)


print('Sampling power law...')
hsamp = np.stack([law.sample(rho, popt[i]) for i in range(10)], axis=-1)


print('Sampling halos in Poisson field...')
xtrues = []
for i in range(10):
    xtrue, _, _ = sample_3d(
        hsamp[...,i], 
        np.sum(hsamp[...,i]).astype(int), 
        3000, 0, np.zeros(3))
    xtrues.append(xtrue.T)

    
print('Calculating velocities...')
k=5
vtrues = []
for i in range(10):
    print(i)
    _, nns = tree.query(xtrues[i], k)
    vnns = pvel[nns.reshape(-1)].reshape(-1,k,3)
    vtrues.append(np.mean(vnns, axis=1))


print('Sampling masses...')
mtrues = []
for i in range(len(edges)-1):
    im = np.random.uniform(*edges[-1][i:i+2], size=len(xtrues[i]))
    mtrues.append(im)

print('Combine...')
xtrues = np.concatenate(xtrues, axis=0)
vtrues = np.concatenate(vtrues, axis=0)
mtrues = np.concatenate(mtrues, axis=0)
    

print('Saving...')
np.save(pjoin(source_dir, f'halo_pos.npy'), xtrues)
np.save(pjoin(source_dir, f'halo_vel.npy'), vtrues)
np.save(pjoin(source_dir, f'halo_mass.npy'), mtrues)

print('Done!')