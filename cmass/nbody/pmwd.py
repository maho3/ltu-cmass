"""
Simulate density field using pmwd.

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
                  lpt, nbody, scatter)
import jax.numpy as jnp
import logging
import numpy as np
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from ..utils import get_source_path, timing_decorator, load_params
from .tools import (
    gen_white_noise, load_white_noise, save_nbody, rho_and_vfield)


def parse_config(cfg):
    with open_dict(cfg):
        nbody = cfg.nbody
        nbody.ai = 1 / (1 + nbody.zi)  # initial scale factor
        nbody.af = 1 / (1 + nbody.zf)  # final scale factor
        nbody.quijote = nbody.matchIC == 2  # whether to match ICs to Quijote
        nbody.matchIC = nbody.matchIC > 0  # whether to match ICs to file

        # load cosmology
        nbody.cosmo = load_params(nbody.lhid, cfg.meta.cosmofile)

    if cfg.nbody.quijote:
        logging.info('Matching ICs to Quijote')
        assert cfg.nbody.L == 1000  # enforce same size of quijote

    return cfg


def configure_pmwd(cfg):
    nbody = cfg.nbody
    cosmo = nbody.cosmo

    N = nbody.N*nbody.supersampling
    ptcl_spacing = nbody.L/N
    ptcl_grid_shape = (N,)*3
    pmconf = Configuration(ptcl_spacing, ptcl_grid_shape,
                           a_start=nbody.ai, a_stop=nbody.af,
                           a_nbody_maxstep=(nbody.af-nbody.ai)/nbody.N_steps,
                           mesh_shape=cfg.nbody.B)
    pmcosmo = Cosmology.from_sigma8(
        pmconf, sigma8=cosmo[4], n_s=cosmo[3], Omega_m=cosmo[0],
        Omega_b=cosmo[1], h=cosmo[2])
    pmcosmo = boltzmann(pmcosmo, pmconf)
    return pmconf, pmcosmo


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
def run_density(wn, pmconf, pmcosmo, cfg):
    ic = linear_modes(wn, pmcosmo, pmconf)
    ptcl, obsvbl = lpt(ic, pmcosmo, pmconf)

    ptcl, obsvbl = nbody(ptcl, obsvbl, pmcosmo, pmconf)

    pos = np.array(ptcl.pos())
    vel = np.array(ptcl.vel)

    vel *= 100  # km/s
    return pos, vel


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'nbody'])

    # Build run config
    cfg = parse_config(cfg)
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(
        "Logging directory: " +
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    # Setup
    pmconf, pmcosmo = configure_pmwd(cfg)

    # Get ICs
    wn = get_ICs(cfg)

    # Run
    pos, vel = run_density(wn, pmconf, pmcosmo, cfg)

    # Calculate velocity field
    rho, fvel = rho_and_vfield(
        pos, vel, cfg.nbody.L, cfg.nbody.N, 'CIC',
        omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2])

    # Convert from comoving -> peculiar velocities
    fvel *= (1 + cfg.nbody.zf)

    # Save
    outdir = get_source_path(cfg, "pmwd", check=False)
    save_nbody(outdir, rho, fvel, pos, vel,
               cfg.nbody.save_particles, cfg.nbody.save_velocities)
    with open(pjoin(outdir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info("Done!")


if __name__ == '__main__':
    main()
