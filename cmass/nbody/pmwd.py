import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '16'  # noqa, must go before jax
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'  # noqa, must go before jax

from pmwd import (
    Configuration,
    Cosmology,
    boltzmann,
    linear_modes,
    lpt,
    nbody,
    scatter,
)
import jax.numpy as jnp
import logging
import numpy as np
from os.path import join as pjoin
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from ..utils import (get_source_path, timing_decorator, load_params)
from .tools import gen_white_noise, load_white_noise, save_nbody


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
    # L, N, N_steps, supersampling, ai, af, cosmo):
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
        return gen_white_noise(N)


@timing_decorator
def run_density(wn, pmconf, pmcosmo, cfg):
    ic = linear_modes(wn, pmcosmo, pmconf)
    del wn
    ptcl, obsvbl = lpt(ic, pmcosmo, pmconf)
    del ic
    ptcl, obsvbl = nbody(ptcl, obsvbl, pmcosmo, pmconf)

    pos = np.array(ptcl.pos())
    vel = ptcl.vel

    # Compute density
    scale = cfg.nbody.supersampling * cfg.nbody.B
    rho = scatter(ptcl, pmconf,
                  mesh=jnp.zeros(3*(cfg.nbody.N,)),
                  cell_size=pmconf.cell_size*scale)
    rho /= scale**3  # undo supersampling

    rho -= 1  # make it zero mean
    vel *= 100  # km/s
    return rho, pos, vel


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

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
    rho, pos, vel = run_density(wn, pmconf, pmcosmo, cfg)

    # Save
    outdir = get_source_path(
        cfg.meta.wdir, "pmwd", cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid,
        check=False)
    save_nbody(outdir, rho, pos, vel, cfg.nbody.save_particles)

    with open(pjoin(outdir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)


if __name__ == '__main__':

    main()

    logging.info('Done')
