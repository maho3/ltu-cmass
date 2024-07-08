"""
Simulate density field using pmwd. This script is similar to
cmass.nbody.pmwd, with the major difference
* It saves multiple snapshots of the density field and velocity field.

Note: THIS SCRIPT IS OUTDATED AND DOESN'T WORK.

Requires:
    - pmwd

Params:
    - nbody.suite: suite name

    - nbody.L: box size (in Mpc/h)
    - nbody.N: number of grid points per dimension in density field
    - nbody.lhid: index of the cosmological parameters in the
        latin_hypercube_params_bonus.txt file
    - nbody.matchIC: whether to match ICs to file (0 no, 1 yes, 2 quijote)
    - nbody.save_particles: whether to save particle positions and velocities


    - nbody.zi: initial redshift
    - nbody.zf: final redshift
    - nbody.supersampling: particle resolution factor relative to density field

    - nbody.B: force grid resolution factor relative to density field
    - nbody.N_steps: number of steps in the nbody simulation


Output:
    - rho: density field
    - ppos: particle positions
    - pvel: particle velocities
"""


import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa, must go before jax
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'  # noqa, must go before jax
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # noqa, must go before jax

from pmwd import (Configuration, Cosmology, boltzmann, linear_modes,
                  lpt, pmnbody)
# TODO: use these instead of different snapshots
from pmwd import nbody_init, nbody_step
import jax.numpy as jnp
import logging
import numpy as np
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf
from ..utils import get_source_path, timing_decorator
from .tools import (
    parse_nbody_config, gen_white_noise, load_white_noise,
    save_nbody, rho_and_vfield)


def configure_pmwd(cfg):
    nbody = cfg.nbody
    cosmo = nbody.cosmo

    N = nbody.N*nbody.supersampling
    ptcl_spacing = nbody.L/N
    ptcl_grid_shape = (N,)*3

    # make a separate configuration for each snapshot

    pmconfs, pmcosmos = [], []
    for i in range(nbody.nsnap):
        if i == 0:
            zi = nbody.zi
            nstep = nbody.nstep_i
        else:
            zi = nbody.zlist[i-1]
            nstep = nbody.nstep_snap
        zf = nbody.zlist[i]
        ai = 1 / (1 + zi)
        af = 1 / (1 + zf)
        conf = Configuration(ptcl_spacing, ptcl_grid_shape,
                             a_start=ai, a_stop=af,
                             a_nbody_maxstep=(af-ai)/nstep,
                             mesh_shape=cfg.nbody.B)
        pmcosmo = Cosmology.from_sigma8(
            conf, sigma8=cosmo[4], n_s=cosmo[3], Omega_m=cosmo[0],
            Omega_b=cosmo[1], h=cosmo[2])
        pmcosmo = boltzmann(pmcosmo, conf)
        pmconfs.append(conf)
        pmcosmos.append(pmcosmo)

    return pmconfs, pmcosmos


def get_ICs(cfg):
    nbody = cfg.nbody
    N = nbody.N*nbody.supersampling
    if nbody.matchIC:
        path_to_ic = f'wn/N{N}/wn_{nbody.lhid}.dat'
        if nbody.quijote:
            path_to_ic = pjoin(cfg.meta.wdir, 'quijote', path_to_ic)
        else:
            path_to_ic = pjoin(cfg.meta.wdir, path_to_ic)
        return load_white_noise(path_to_ic, N, quijote=nbody.quijote)
    else:
        return gen_white_noise(N, seed=nbody.lhid)


@timing_decorator
def run_density(wn, pmconfs, pmcosmos, cfg):
    nbody = cfg.nbody

    # run initial displacement
    logging.info('Running initial displacement...')
    ic = linear_modes(wn, pmcosmos[0], pmconfs[0])
    ptcl, obsvbl = lpt(ic, pmcosmos[0], pmconfs[0])

    # run nbody simulation
    rhos, fvels, poss, vels = [], [], [], []
    for i in range(nbody.nsnap):
        logging.info(f'Running snapshot {i+1}/{nbody.nsnap}...')
        ptcl, obsvbl = pmnbody(ptcl, obsvbl, pmcosmos[0], pmconfs[i])

        pos = np.array(ptcl.pos())
        vel = np.array(ptcl.vel)

        vel *= 100  # km/s

        rho, fvel = rho_and_vfield(
            pos, vel,
            Ngrid=nbody.N,
            BoxSize=nbody.L,
            MAS='CIC',
            omega_m=nbody.cosmo[0],
            h=nbody.cosmo[2],
        )

        # Save
        rhos.append(rho)
        fvels.append(fvel)
        if cfg.nbody.save_particles:
            poss.append(pos)
            vels.append(vel)
        else:  # save memory
            poss.append(None)
            vels.append(None)

    return rhos, fvels, poss, vels


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'nbody'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(
        "Logging directory: " +
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    # Setup
    pmconfs, pmcosmos = configure_pmwd(cfg)

    # Get ICs
    wn = get_ICs(cfg)

    # Run
    rhos, fvels, poss, vels = run_density(wn, pmconfs, pmcosmos, cfg)

    # Save
    outdir = get_source_path(cfg, "pmwd", check=False)
    for i in range(cfg.nbody.nsnap):
        save_nbody(outdir, rhos[i], fvels[i], poss[i], vels[i], i,
                   cfg.nbody.save_particles, cfg.nbody.save_velocities)
    with open(pjoin(outdir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info("Done!")


if __name__ == '__main__':
    main()
