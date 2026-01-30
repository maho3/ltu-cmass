import numpy as np
import os
import yaml
import logging
from omegaconf import DictConfig, OmegaConf

from .hodtools import HODEngine
from ..lightcone import lc
from ..bias.apply_hod import load_snapshot#, randoms_engine
from ..bias.tools.hod import parse_hod
from ..nbody.tools import parse_nbody_config
from .hodlightcone import split_galsnap_galidx, check_saturation
from .tools import save_lightcone, in_simbig_selection

use_randoms = False
lhid = 0 #663
config_path = f'/anvil/scratch/x-mho1/cmass-ili/mtnglike/fastpm_recnoise/L3000-N384/{lhid}/'
source_path = f'/anvil/scratch/x-mho1/cmass-ili/mtnglike/fastpm/L3000-N384/{lhid}/'
save_dir = f'/anvil/scratch/x-dbartlett/cmass/mtnglike/fastpm/L3000-N384_tests/{lhid}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# Load config file
fname = os.path.join(config_path, 'config.yaml')
with open(fname, 'r') as f:
    config = yaml.safe_load(f)
print(f'Using config file: {fname}')

# Change bias part of config file
fname = 'cmass/conf/bias/zdep.yaml'
with open(fname, 'r') as f:
    bias_config = yaml.safe_load(f)
config['bias'] = bias_config

# Add in the config.yaml part
fname = 'cmass/conf/config.yaml'
with open(fname, 'r') as f:
    meta_config = yaml.safe_load(f)
for key, value in meta_config.items():
    if key not in config:
        config[key] = value

for key, value in config.items():
    print(f'\n{key}: {value}')


# Parse the config
cfg = OmegaConf.create(config)
cfg = parse_nbody_config(cfg)
cfg = parse_hod(cfg)

# Save with original hod_seed
if cfg.bias.hod.seed == 0:
    hod_seed = cfg.bias.hod.seed
else:
    # (parse_hod modifies it to lhid*1e6 + hod_seed)
    hod_seed = int(cfg.bias.hod.seed - cfg.nbody.lhid * 1e6)

aug_seed = int(config['survey']['aug_seed'])
geometry = config['survey']['geometry']
BoxSize=float(config['nbody']['L'])
Omega_m = float(config['nbody']['cosmo'][0])
Ngrid = int(config['nbody']['N'])
snap_times = sorted(config['nbody']['asave'])[::-1]  # decreasing order
zmin, zmax = config['survey']['z_range']
zmin = float(zmin)
zmax = float(zmax)
snap_times = [float(a) for a in snap_times if (zmin < (1/a - 1) < zmax)]
noise_uniform = True
print(f'Using snapshots at redshifts: {snap_times}')

# Get path to lightcone module (where n(z) is saved)
nz_dir = os.path.dirname(lc.__file__)

# Load mask
wdir = "/anvil/scratch/x-mho1/cmass-ili"
config['survey']['boss_dir'] = os.path.join(wdir, 'obs')
if geometry == 'ngc':
    maskobs = lc.Mask(boss_dir=config['survey']['boss_dir'],
                        veto=True, is_north=True)
    remap_case = 1
    zmid = 0.45
elif geometry == 'sgc':
    maskobs = lc.Mask(boss_dir=config['survey']['boss_dir'],
                        veto=True, is_north=False)
    remap_case = 3
    zmid = 0.55
elif geometry == 'mtng':
    maskobs = None
    remap_case = 0
    zmid = 0.55
elif geometry == 'simbig':
    maskobs = lc.Mask(boss_dir=config['survey']['boss_dir'],
                        veto=True, is_north=False)
    remap_case = 4
    zmid = 0.55
else:
    raise ValueError(
        'Invalid geometry {geometry}. Choose from NGC, SGC, or MTNG.')


for rsd in [True, False]:

    logging.info(f'Running lightcone with RSD={rsd}...')
    # Setup lightcone
    kwargs = dict(
        boss_dir=nz_dir if config['survey']['fix_nz'] else None,
        mask=maskobs,
        BoxSize=BoxSize,
        Omega_m=Omega_m,
        zmin=zmin,
        zmax=zmax,
        # zmid=zmid,  # to set the offset of the simulation box
        snap_times=snap_times,
        verbose=True,
        augment=aug_seed,
        remap_case=remap_case,
        seed=42,
        is_north=geometry == 'ngc'
    )
    lightcone = lc.Lightcone(**kwargs)

    # Setup HOD model function
    if not use_randoms:
        hod_fct = HODEngine(cfg, snap_times, source_path)
    else:
        raise NotImplementedError(
            'Randoms are not implemented yet in this script. '
            'Please use the HODEngine with a snapshot source path.')
    lightcone.set_hod(hod_fct)

    # Load snapshots
    for snap_idx, a in enumerate(snap_times):
        logging.info(f'Loading snapshot at a={a:.6f}...')
        if not use_randoms:
            hpos, hvel, _, _ = load_snapshot(source_path, a)
        else:
            nbar_randoms = 3e-5  # number density of CMASS
            Nrandoms = int(nbar_randoms * BoxSize**3)
            hpos = np.random.rand(Nrandoms, 3) * BoxSize
            hvel = np.zeros_like(hpos)

        if not rsd:
            hvel = np.zeros_like(hvel)

        # Uniformly noise the halos positons in the voxel
        if noise_uniform:
            Delta = BoxSize / Ngrid
            logging.info(
                f'Applying uniform position noise for voxel size {Delta} Mpc/h')
            hpos += np.random.uniform(-Delta/2, Delta/2, size=hpos.shape)
            hpos = np.mod(hpos, BoxSize)  # wrap around the box

        lightcone.add_snap(snap_idx, hpos, hvel)

    # Run lightcone
    ra, dec, z, galid = lightcone.finalize()

    # Conform to [0, 2pi] and [-pi/2, pi/2]
    ra = np.mod(ra, 360)
    dec = np.mod(dec + 90, 180) - 90

    galsnap, galidx = split_galsnap_galidx(galid)

    # If SIMBIG, apply selection
    if geometry == 'simbig' and not cfg.survey.nomask:
        logging.info('Applying SIMBIG selection...')
        m = in_simbig_selection(ra, dec, z)
        ra, dec, z = ra[m], dec[m], z[m]
        galsnap, galidx = galsnap[m], galidx[m]

    # If MTNG, apply MTNG selection
    if geometry == 'mtng':
        m = (ra >= 0) & (ra < 90) & (dec >= 0) & (dec < 90)
        ra, dec, z = ra[m], dec[m], z[m]
        galsnap, galidx = galsnap[m], galidx[m]

    # Check if n(z) is saturated
    # if cfg.survey.nomask:
    #     saturated = False
    # else:
    saturated = check_saturation(z, nz_dir, zmin, zmax, geometry)

    # Save the lightcone
    save_lightcone(
        save_dir,
        ra=ra, dec=dec, z=z,
        galsnap=galsnap,
        galidx=galidx,
        hod_seed=hod_seed,
        aug_seed=str(aug_seed) + '_rsd' if rsd else str(aug_seed) + '_no_rsd',
        saturated=saturated,
        config=cfg
    )