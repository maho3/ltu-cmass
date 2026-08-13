"""
Stitches together halo snapshots to create an extrapolated lightcone, applies
a redshift-dependent HOD model, and applies survey masks and selection effects.
Allows for NGC, SGC, and MTNG-like lightcones.

Input:
    - halos.h5
        - a: scale factors
        - pos: halo positions
        - vel: halo velocities
        - mass: halo masses

Output:
    - lightcone/hod{hod_seed}_aug{augmentation_seed}.h5
        - ra: right ascension
        - dec: declination
        - z: redshift
        - galsnap: snapshot index
        - galidx: galaxy index

NOTE:
    - This script has superseeded lightcone.py and selection.py which stitched
    together NGC lightcones from snapshots of galaxies. This script now jointly
    applies HOD and stitches the lightcone, allowing for z-dependent HODs.
    - This script allows for NGC, SGC, and MTNG-like lightcones.
    - This only works for snapshot mode, wherein lightcone evolution is
    mimicked by stitching snapshots together. For the non-snapshot mode
    alternative, use 'ngc_selection.py'.
    - The fiber collisions are applied in-sync with resampling.
"""

import os
import numpy as np
import logging
from os.path import join
import hydra
from omegaconf import DictConfig, OmegaConf
from ..utils import get_source_path, timing_decorator, clean_up, save_cfg
from ..nbody.tools import parse_nbody_config
from .tools import save_lightcone, in_simbig_selection
from .hodtools import HODEngine, randoms_engine
from ..bias.apply_hod import load_snapshot
from ..bias.tools.hod import parse_hod, parse_noise

try:
    from ..lightcone import lc
except ImportError:
    raise ImportError(
        'Lightcone extrapolation not compiled. Please `make` the '
        'lightcone package in cmass/lightcone')


def split_galsnap_galidx(gid):
    return np.divmod(gid, 2**((gid.itemsize-1)*8))


def stitch_lightcone(lightcone, source_path, snap_times, BoxSize, Ngrid,
                     noise_uniform, use_randoms=False, randoms_nbar=3.0e-3):
    # Load snapshots
    for snap_idx, a in enumerate(snap_times):
        logging.info(f'Loading snapshot at a={a:.6f}...')
        if not use_randoms:
            hpos, hvel, _, _ = load_snapshot(source_path, a)
        else:
            # Uniform cube, later downsampled to the target n(z) if fix_nz.
            # randoms_nbar must exceed the peak target density in every
            # z-bin or the downsampling cannot saturate; see the
            # survey.randoms_nbar comment in conf/survey/default.yaml.
            Nrandoms = int(randoms_nbar * BoxSize**3)
            logging.info(f'Drawing {Nrandoms} uniform randoms '
                         f'(nbar={randoms_nbar:.3e} (h/Mpc)^3)')
            hpos = np.random.rand(Nrandoms, 3) * BoxSize
            hvel = np.zeros_like(hpos)

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

    # Split galid into galsnap and galidx
    galsnap, galidx = split_galsnap_galidx(galid)
    return ra, dec, z, galsnap, galidx


def _nz_cap(geometry):
    """Cap name in the n(z) target filename, or None if we have no target."""
    return {'ngc': 'North', 'sgc': 'South', 'mtng': 'MTNG'}.get(geometry)


def prepare_nz_target(nz_dir, out_dir, geometry, zmin, zmax, factor):
    """Write the n(z) target the C++ lightcone will actually read.

    Two things are handled here.

    1. `factor` scales the observed counts. fix_nz alone downsamples to
       exactly the observed n(z); randoms want that same *shape* at higher
       density, so factor=10 gives 10x the observed n(z). It may be a scalar
       or a per-bin vector -- mtng needs the latter, because its angular cut
       is applied *after* the C++ downsampling (see main), so the target has
       to be pre-inflated by 1/survival to land on the requested factor.

    2. `Lightcone` picks its target file purely from `is_north`
       (lightcone.cpp: North if is_north else South) -- there is no MTNG
       branch, so an mtng run would otherwise silently downsample to the
       South n(z). We therefore write the counts we actually want under
       *both* the cap's own name and the name the C++ will read.

    Returns the directory to pass as `boss_dir` (and to check_saturation).
    """
    cap = _nz_cap(geometry)
    if cap is None:
        raise ValueError(f'No n(z) target available for geometry {geometry}')

    src = join(nz_dir,
               f'nz_DR12v5_CMASS_{cap}_zmin{zmin:.4f}_zmax{zmax:.4f}.dat')
    counts = np.loadtxt(src, usecols=(0,))
    factor = np.asarray(factor, dtype=float)
    if factor.ndim and len(factor) != len(counts):
        raise ValueError(
            f'nz_factor has {len(factor)} entries but the {cap} target has '
            f'{len(counts)} bins')
    scaled = np.rint(counts * factor).astype(int)

    os.makedirs(out_dir, exist_ok=True)
    # 'North' if is_north else 'South' -- must match hodlightcone's is_north
    read_as = 'North' if geometry == 'ngc' else 'South'
    for name in {cap, read_as}:
        dst = join(out_dir,
                   f'nz_DR12v5_CMASS_{name}_zmin{zmin:.4f}_zmax{zmax:.4f}.dat')
        np.savetxt(dst, scaled, fmt='%d')
    fstr = (f'x{float(factor):g}' if not factor.ndim
            else f'x[{factor.min():.2f}-{factor.max():.2f}] per-bin')
    logging.info(
        f'Wrote n(z) target for {geometry} ({fstr}, {scaled.sum()} '
        f'objects over {len(scaled)} bins) to {out_dir}')
    return out_dir


def check_saturation(z, nz_dir, zmin, zmax, geometry):
    cap = _nz_cap(geometry)
    if cap is None:
        return False  # SIMBIG hasn't been calculated yet

    filepath = join(
        nz_dir, f'nz_DR12v5_CMASS_{cap}_zmin{zmin:.4f}_zmax{zmax:.4f}.dat')
    nzobs = np.loadtxt(filepath, usecols=(0,))
    zbins = np.linspace(zmin, zmax, len(nzobs)+1)

    # Check if n(z) is saturated (within 1-sigma of observed n(z))
    nz = np.histogram(z, bins=zbins)[0]
    saturated = np.all(nz >= nzobs - np.sqrt(nzobs))
    return saturated


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(
        cfg, ['meta', 'sim', 'multisnapshot', 'nbody', 'bias',
              'survey', 'noise'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    cfg = parse_hod(cfg)
    # parse noise (seeded by lhid and hod seed)  (TODO: make take less lines)
    noise_seed = int(cfg.nbody.lhid*1e4 + cfg.bias.hod.seed)
    cfg.noise.radial, cfg.noise.transverse = \
        parse_noise(seed=noise_seed,
                    dist=cfg.noise.dist,
                    params=cfg.noise.params)

    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))
    source_path = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, cfg.sim,
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid
    )
    # Save with original hod_seed
    if cfg.bias.hod.seed == 0:
        hod_seed = cfg.bias.hod.seed
    else:
        # (parse_hod modifies it to lhid*1e4 + hod_seed)
        hod_seed = int(cfg.bias.hod.seed - cfg.nbody.lhid * 1e4)
    aug_seed = cfg.survey.aug_seed  # for rotating and shuffling

    geometry = cfg.survey.geometry  # whether to use NGC, SGC, or MTNG mask
    geometry = geometry.lower()

    # check if noise_uniform, then the sim is FastPM
    if cfg.bias.hod.noise_uniform and ('fastpm' not in cfg.sim):
        raise ValueError(
            'noise_uniform is only supported for CHARM simulations. '
            'Please either set cfg.bias.hod.noise_uniform=False, use a CHARM '
            'sim, or disable this warning.')

    # throw a big warning if we're generating randoms
    if cfg.survey.randoms:
        logging.warning(
            'Generating uniform randoms. This is not for generating '
            'training data, but for generating randoms for the lightcone. '
        )

    # Load mask
    if geometry == 'ngc':
        maskobs = lc.Mask(boss_dir=cfg.survey.boss_dir,
                          veto=True, is_north=True)
        remap_case = 1
        zmid = 0.45
    elif geometry == 'sgc':
        maskobs = lc.Mask(boss_dir=cfg.survey.boss_dir,
                          veto=True, is_north=False)
        remap_case = 3
        zmid = 0.55
    elif geometry == 'mtng':
        maskobs = None
        remap_case = 0
        zmid = 0.55
    elif geometry == 'simbig':
        maskobs = lc.Mask(boss_dir=cfg.survey.boss_dir,
                          veto=True, is_north=False)
        remap_case = 4
        zmid = 0.55
    else:
        raise ValueError(
            'Invalid geometry {geometry}. Choose from NGC, SGC, or MTNG.')

    # Get path to lightcone module (where n(z) is saved)
    nz_dir = os.path.dirname(lc.__file__)

    # Setup lightcone constructor
    if 'z_range' in cfg.survey:
        zmin, zmax = cfg.survey.z_range
    else:
        zmin, zmax = 0.4, 0.7

    # If no mask mode, do not mask the lightcone (for testing only!)
    if cfg.survey.nomask:
        logging.warning(
            'No mask mode is enabled. This will not apply any survey mask '
            'or selection effects. Use with caution, only for testing purposes.'
        )
        maskobs = None
        zmin, zmax = 0.0, 1.1  # midpoint is the same

    # If fix_nz, resolve the n(z) target. nz_factor>1 scales the observed
    # counts (randoms want the observed shape at higher density), and
    # prepare_nz_target also works around the C++ only reading North/South.
    nz_factor = getattr(cfg.survey, 'nz_factor', 1)
    if cfg.survey.fix_nz:
        nz_factor = OmegaConf.to_object(nz_factor) \
            if OmegaConf.is_config(nz_factor) else nz_factor
        tag = (f'x{float(nz_factor):g}' if np.isscalar(nz_factor)
               else f'perbin{np.mean(nz_factor):.2f}')
        target_dir = prepare_nz_target(
            nz_dir,
            join(cfg.meta.wdir, 'scratch', 'nz_targets',
                 f'{geometry}_{tag}'),
            geometry, zmin, zmax, nz_factor)
    else:
        target_dir = nz_dir  # only used by check_saturation

    # Setup lightcone
    snap_times = sorted(cfg.nbody.asave)[::-1]  # decreasing order
    snap_times = [a for a in snap_times if (zmin < (1/a - 1) < zmax)]
    lightcone = lc.Lightcone(
        boss_dir=target_dir if cfg.survey.fix_nz else None,
        mask=maskobs,
        BoxSize=cfg.nbody.L,
        Omega_m=cfg.nbody.cosmo[0],
        zmin=zmin,
        zmax=zmax,
        zmid=zmid,  # to set the offset of the simulation box
        snap_times=snap_times,
        verbose=False,
        augment=aug_seed,
        remap_case=remap_case,
        sigmaradial=cfg.noise.radial,
        sigmatransverse=cfg.noise.transverse,
        seed=42,
        is_north=geometry == 'ngc'
    )

    # Setup HOD model function
    if not cfg.survey.randoms:
        hod_fct = HODEngine(cfg, snap_times, source_path)
    else:
        hod_fct = randoms_engine  # for constructing randoms
    lightcone.set_hod(hod_fct)

    logging.info(f'Stitching snapshots a={snap_times}')
    ra, dec, z, galsnap, galidx = stitch_lightcone(
        lightcone, source_path, snap_times,
        cfg.nbody.L, cfg.nbody.N, cfg.bias.hod.noise_uniform,
        use_randoms=cfg.survey.randoms,
        randoms_nbar=float(getattr(cfg.survey, 'randoms_nbar', 3.0e-3)))

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
    if cfg.survey.nomask:
        saturated = False
    else:
        saturated = check_saturation(z, target_dir, zmin, zmax, geometry)

    # Save
    if geometry == 'ngc':
        outdir = join(source_path, 'ngc_lightcone')
    elif geometry == 'sgc':
        outdir = join(source_path, 'sgc_lightcone')
    elif geometry == 'mtng':
        outdir = join(source_path, 'mtng_lightcone')
    elif geometry == 'simbig':
        outdir = join(source_path, 'simbig_lightcone')
    os.makedirs(outdir, exist_ok=True)
    save_lightcone(
        outdir,
        ra=ra, dec=dec, z=z,
        galsnap=galsnap,
        galidx=galidx,
        hod_seed=hod_seed,
        aug_seed=aug_seed,
        saturated=saturated,
        config=cfg,
        noise_radial=cfg.noise.radial,
        noise_transverse=cfg.noise.transverse
    )
    save_cfg(source_path, cfg, field='survey')


if __name__ == "__main__":
    main()
