"""
Simulate density field using pmwd Particle Mesh.

Input:
    - wn: initial white noise field

Output:
    - nbody.h5
        - rho: density contrast field
        - fvel: velocity field
        - pos: particle positions [optional]
        - vel: particle velocities [optional]

NOTE:
    - [dev] If making this lightcone, consider using:
            from pmwd import nbody_init, nbody_step
"""


import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa, must go before jax
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'  # noqa, must go before jax
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # noqa, must go before jax

from pmwd import (
    Configuration, Cosmology, boltzmann, linear_modes, lpt, nbody)
from pmwd.pm_util import fftinv
import logging
import numpy as np
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf
from ..utils import get_source_path, timing_decorator
from .tools import (
    parse_nbody_config, get_ICs, save_nbody, rho_and_vfield)


def configure_pmwd(
    N, L, supersampling, B, ai, af, N_steps, cosmo
):
    N = N*supersampling
    ptcl_spacing = L/N
    ptcl_grid_shape = (N,)*3
    pmconf = Configuration(
        ptcl_spacing, ptcl_grid_shape,
        a_start=ai, a_stop=af,
        a_nbody_maxstep=(af-ai)/N_steps,
        mesh_shape=B)
    pmcosmo = Cosmology.from_sigma8(
        pmconf, sigma8=cosmo[4], n_s=cosmo[3], Omega_m=cosmo[0],
        Omega_b=cosmo[1], h=cosmo[2])
    pmcosmo = boltzmann(pmcosmo, pmconf)
    return pmconf, pmcosmo


@timing_decorator
def run_transfer(wn, pmconf, pmcosmo):
    rho = linear_modes(wn, pmcosmo, pmconf)
    rho = fftinv(
        rho, shape=pmconf.ptcl_grid_shape,
        norm=pmconf.ptcl_spacing)
    rho /= 100  # needed to align with borg, idk why
    return rho


@timing_decorator
def run_density(wn, pmconf, pmcosmo):
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
    cfg = parse_nbody_config(cfg)
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(
        "Logging directory: " +
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    # Output directory
    outdir = get_source_path(cfg, "pmwd", check=False)
    os.makedirs(outdir, exist_ok=True)

    # Get ICs
    wn = get_ICs(cfg)

    # Run transfer function
    if cfg.nbody.save_transfer:
        pmconf, pmcosmo = configure_pmwd(
            N=cfg.nbody.N, L=cfg.nbody.L,
            supersampling=cfg.nbody.supersampling,
            B=cfg.nbody.B,
            ai=1./(1+99),  # z=99
            af=1,  # ignored
            N_steps=cfg.nbody.N_steps, cosmo=cfg.nbody.cosmo)
        rho_transfer = run_transfer(wn, pmconf, pmcosmo)
        np.save(pjoin(outdir, 'rho_transfer.npy'), rho_transfer)
        del rho_transfer

    # Run density
    pmconf, pmcosmo = configure_pmwd(
        N=cfg.nbody.N, L=cfg.nbody.L,
        supersampling=cfg.nbody.supersampling,
        B=cfg.nbody.B, ai=cfg.nbody.ai, af=cfg.nbody.af,
        N_steps=cfg.nbody.N_steps, cosmo=cfg.nbody.cosmo)
    pos, vel = run_density(wn, pmconf, pmcosmo)

    # Calculate velocity field
    rho, fvel = rho_and_vfield(
        pos, vel, cfg.nbody.L, cfg.nbody.N, 'CIC',
        omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2])

    if not cfg.nbody.save_particles:
        pos, vel = None, None

    # Convert to overdensity field
    rho /= np.mean(rho)
    rho -= 1

    # Convert from comoving -> physical velocities
    fvel *= (1 + cfg.nbody.zf)

    # Save
    save_nbody(outdir, rho, fvel, pos, vel)
    with open(pjoin(outdir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info("Done!")


if __name__ == '__main__':
    main()
