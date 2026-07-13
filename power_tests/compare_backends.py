"""Compare pylians vs pypower P(k) multipoles on periodic Quijote halo boxes.

Measures P0/P2/P4 in real and redshift space (RSD axis=2) for several
mesh/backend configs, with matched k_F binning, plus wall-clock timings.

Run from repo root:
    PYTHONPATH=. python power_tests/compare_backends.py --lhid 663
"""
import argparse
import os
import time
import h5py
import numpy as np

from cmass.utils.utils import load_params, cosmo_to_astropy
from cmass.diagnostics.calculations import (
    MA, calcPk, get_redshift_space_pos)
from pypower import MeshFFTPower, CatalogMesh

L = 1000.0
Z = 0.5
A = '0.666667'
DATADIR = '/work/hdd/bdne/maho3/cmass-ili/quijote/nbody/L1000-N128'
KEDGES = np.arange(0, 0.41, 2 * np.pi / L)
NREP = 3

CONFIGS = [
    # (name, backend, N, interlacing)
    ('pylians_n128', 'pylians', 128, None),
    ('pylians_n256', 'pylians', 256, None),
    ('pypower_n128_i0', 'pypower', 128, 0),
    ('pypower_n128_i2', 'pypower', 128, 2),
    ('pypower_n256_i2', 'pypower', 256, 2),
    ('pypower_n512_i2', 'pypower', 512, 2),   # truth
    ('pylians_n512', 'pylians', 512, None),   # truth cross-check
]


def run_pylians(pos, N, threads):
    """Paint + P(k) with the summ.py pathway. Returns (k, Pk[Nk,3], Nmodes)."""
    delta = MA(pos, L, N, MAS='TSC').astype(np.float32)
    k, Pk, Nmodes = calcPk(delta, L, axis=2, MAS='TSC', threads=threads)
    mask = k < KEDGES[-1]
    return k[mask], Pk[mask, :3], Nmodes[mask]


def run_pypower(pos, N, interlacing):
    """Paint + P(k) with pypower in periodic-box mode, los='z'.

    Monopole kept WITHOUT shot-noise subtraction to match pylians convention.
    """
    mesh = CatalogMesh(
        data_positions=pos, boxsize=L, boxcenter=L / 2, nmesh=N,
        resampler='tsc', interlacing=interlacing, position_type='pos')
    res = MeshFFTPower(mesh, edges=KEDGES, ells=(0, 2, 4), los='z')
    poles = res.poles
    k = poles.k
    Pk = np.stack([
        poles(ell=0, complex=False, remove_shotnoise=False),
        poles(ell=2, complex=False),
        poles(ell=4, complex=False)], axis=-1)
    Nmodes = poles.nmodes
    good = ~np.isnan(k)
    return k[good], Pk[good], Nmodes[good]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lhid', type=int, required=True)
    ap.add_argument('--outdir', default='./data/scratch/power_tests')
    ap.add_argument('--threads', type=int, default=16)
    args = ap.parse_args()

    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))

    with h5py.File(f'{DATADIR}/{args.lhid}/halos.h5', 'r') as f:
        hpos = f[A]['pos'][:].astype(np.float32)
        hvel = f[A]['vel'][:].astype(np.float32)
    cosmo = cosmo_to_astropy(
        load_params(args.lhid, 'params/latin_hypercube_params.txt'))
    print(f'lhid={args.lhid}: {len(hpos)} halos')

    zpos = get_redshift_space_pos(
        hpos.copy(), hvel.copy(), L, cosmo, Z, axis=2).astype(np.float32)

    out = {'lhid': args.lhid, 'Nhalos': len(hpos)}
    for space, pos in [('real', hpos), ('zspace', zpos)]:
        for name, backend, N, interlacing in CONFIGS:
            times = []
            for rep in range(NREP):
                t0 = time.perf_counter()
                if backend == 'pylians':
                    k, Pk, Nmodes = run_pylians(pos, N, args.threads)
                else:
                    k, Pk, Nmodes = run_pypower(pos, N, interlacing)
                times.append(time.perf_counter() - t0)
            key = f'{space}_{name}'
            out[f'{key}_k'] = k
            out[f'{key}_Pk'] = Pk
            out[f'{key}_Nmodes'] = Nmodes
            out[f'{key}_times'] = np.array(times)
            print(f'{key}: {min(times):.2f}s (best of {NREP})')

    os.makedirs(args.outdir, exist_ok=True)
    fname = os.path.join(args.outdir, f'compare_lhid{args.lhid}.npz')
    np.savez(fname, **out)
    print(f'saved {fname}')


if __name__ == '__main__':
    main()
