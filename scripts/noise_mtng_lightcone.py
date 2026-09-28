"""
Apply observational noise to the MTNG lightcone, one catalog per noisegrid row.

Reads the un-noised base catalog (hod00000_aug00000.h5), displaces galaxies by
a separable radial/transverse Gaussian kernel in comoving space, re-projects to
(ra, dec, z), and re-applies the survey footprint cuts. Row i is written to
aug{i+1} so that aug00000 always stays the noiseless catalog.

Downstream, cmass.diagnostics.summ reads these pre-noised catalogs directly;
diag.noise_seed only selects the output filename and records (sigma_r, sigma_t)
as attrs, so the noise must be applied here.

    python scripts/noise_mtng_lightcone.py --wdir $wdir --lhid 0 \
        --noise-file $wdir/noise_priors/noisegrid.csv --rows 31,32,33 \
        --write-bias-cfg
"""
import argparse
import logging
import os
import shutil
from os.path import join

import h5py
import numpy as np
from omegaconf import OmegaConf

from cmass.diagnostics.tools import noise_positions
from cmass.survey.tools import sky_to_xyz, xyz_to_sky
from cmass.utils import get_source_path, load_params

CAP = 'mtng'
L, N = 3000, 384
COSMOFILE = './params/mtng_cosmologies.txt'

# survey footprint, mirroring cmass/conf/survey/mtng.yaml
RA_RANGE = (0., 90.)
DEC_RANGE = (0., 90.)
Z_RANGE = (0.4, 0.7)

# the HOD/bias provenance of the MTNG lightcone, which is not produced by our
# own bias pipeline. summ needs this section to resolve parse_hod().
BIAS_CFG = {
    'bias': {
        'halo': {
            'model': 'CHARM', 'base_suite': 'calib_1gpch_z0.5',
            'L': 1000, 'N': 128, 'vel': 'CIC',
        },
        'hod': {
            'model': 'zheng07zinterp', 'assem_bias': True,
            'vel_assem_bias': True, 'custom_prior': 'mtng',
            'from_samples': True, 'mdef': '200c', 'use_conc': True,
            'default_params': 'reid2014_cmass', 'seed': 0,
            'zpivot': [0.4, 0.5, 0.7],
            'theta': {
                'logMmin_z0': 12.745153427124023,
                'logMmin_z1': 13.213532447814941,
                'logMmin_z2': 13.475794792175293,
                'sigma_logM': 0.38,
                'logM0_z0': 13.27, 'logM0_z1': 13.27, 'logM0_z2': 13.27,
                'logM1_z0': 14.08, 'logM1_z1': 14.08, 'logM1_z2': 14.08,
                'alpha': 0.76,
                'mean_occupation_centrals_assembias_param1': 0.0,
                'mean_occupation_satellites_assembias_param1': 0.0,
                'eta_vb_centrals': 0.0,
                'eta_vb_satellites': 1.0,
                'conc_gal_bias_satellites': 1.0,
            },
            'noise_uniform': False,
        },
    }
}


def write_bias_cfg(source_path):
    cfgfile = join(source_path, 'config.yaml')
    backup = cfgfile + '.orig'
    if not os.path.exists(backup):
        shutil.copy2(cfgfile, backup)
    cfg = OmegaConf.load(cfgfile)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(BIAS_CFG))
    with open(cfgfile, 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info(f'Wrote bias section to {cfgfile} (backup {backup})')


def footprint_mask(rdz):
    ra, dec, z = rdz.T
    return ((ra >= RA_RANGE[0]) & (ra <= RA_RANGE[1]) &
            (dec >= DEC_RANGE[0]) & (dec <= DEC_RANGE[1]) &
            (z >= Z_RANGE[0]) & (z <= Z_RANGE[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wdir', required=True)
    parser.add_argument('--lhid', type=int, default=0)
    parser.add_argument('--noise-file', required=True)
    parser.add_argument('--rows', required=True,
                        help='comma-separated noisegrid rows, or "all"')
    parser.add_argument('--write-bias-cfg', action='store_true')
    parser.add_argument('--seed-base', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true',
                        help='report kept counts without writing catalogs')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='[%(asctime)s-%(levelname)s] %(message)s',
        datefmt='%H:%M:%S')

    source_path = get_source_path(args.wdir, CAP, 'nbody', L, N, args.lhid)
    lcdir = join(source_path, f'{CAP}_lightcone')

    if args.write_bias_cfg and not args.dry_run:
        write_bias_cfg(source_path)

    noise_grid = np.loadtxt(args.noise_file, delimiter=',')
    if args.rows == 'all':
        rows = list(range(len(noise_grid)))
    else:
        rows = [int(r) for r in args.rows.split(',') if r != '']

    cosmo = load_params(args.lhid, COSMOFILE)

    basefile = join(lcdir, f'hod{0:05}_aug{0:05}.h5')
    with h5py.File(basefile, 'r') as f:
        base = {k: f[k][:] for k in ['ra', 'dec', 'z', 'galidx', 'galsnap']}
    rdz = np.stack([base['ra'], base['dec'], base['z']], axis=1)
    logging.info(f'Loaded {len(rdz)} MTNG galaxies (Om={cosmo[0]:.4f})')

    # a single forward projection, reused for every row
    xyz = sky_to_xyz(rdz, cosmo)

    for row in rows:
        sig_r, sig_t = (float(v) for v in noise_grid[row])
        np.random.seed(args.seed_base + row)

        if sig_r == 0 and sig_t == 0:
            rdz_n = rdz.copy()
        else:
            pos = noise_positions(
                xyz.copy(), base['ra'], base['dec'],
                noise_radial=sig_r, noise_transverse=sig_t)
            rdz_n = xyz_to_sky(pos, cosmo=cosmo)

        keep = footprint_mask(rdz_n)
        outfile = join(lcdir, f'hod{0:05}_aug{row + 1:05}.h5')
        logging.info(f'row {row}: sig_r={sig_r:.3f} sig_t={sig_t:.3f} '
                     f'kept {keep.sum()} -> {outfile}')
        if args.dry_run:
            continue

        with h5py.File(outfile, 'w') as f:
            f.create_dataset('ra', data=rdz_n[keep, 0])
            f.create_dataset('dec', data=rdz_n[keep, 1])
            f.create_dataset('z', data=rdz_n[keep, 2])
            f.create_dataset('galidx', data=base['galidx'][keep])
            f.create_dataset('galsnap', data=base['galsnap'][keep])


if __name__ == '__main__':
    main()
