"""
Simulate density field using BORG PM models.
NOTE: This works with the private BORG version, available to Aquila members.

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

    - nbody.transfer: transfer function 'CLASS' or 'EH'
    - nbody.order: LPT order (1 or 2)


Output:
    - rho: density field
    - ppos: particle positions
    - pvel: particle velocities
"""

import os
os.environ["PYBORG_QUIET"] = "yes"  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # noqa, must go before jax
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'  # noqa, must go before jax

from os.path import join as pjoin
import numpy as np
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import aquila_borg as borg
from ..utils import get_source_path, timing_decorator, load_params
from .tools import gen_white_noise, load_white_noise, save_nbody, vfield_CIC
from .tools_borg import build_cosmology


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


def get_ICs(cfg):
    nbody = cfg.nbody
    N = nbody.N
    if nbody.matchIC:
        path_to_ic = f'wn/N{N}/wn_{nbody.lhid}.dat'
        if nbody.quijote:
            path_to_ic = pjoin(cfg.meta.wdir, 'quijote', path_to_ic)
        else:
            path_to_ic = pjoin(cfg.meta.wdir, path_to_ic)
        return - load_white_noise(path_to_ic, N, quijote=nbody.quijote)
    else:
        return gen_white_noise(N, seed=nbody.lhid)


@timing_decorator
def run_density(wn, cpar, cfg):
    nbody = cfg.nbody

    # Compute As
    sigma8_true = np.copy(cpar.sigma8)
    cpar.sigma8 = 0
    cpar.A_s = 2.3e-9
    k_max, k_per_decade = 10, 100
    extra_class = {}
    extra_class['YHe'] = '0.24'
    cosmo = borg.cosmo.ClassCosmo(cpar, k_per_decade, k_max, extra=extra_class)
    cosmo.computeSigma8()
    cos = cosmo.getCosmology()
    cpar.A_s = (sigma8_true/cos['sigma_8'])**2*cpar.A_s
    cpar.sigma8 = sigma8_true

    # initialize box and chain
    box = borg.forward.BoxModel()
    box.L = 3*(nbody.L,)
    box.N = 3*(nbody.N,)

    chain = borg.forward.ChainForwardModel(box)
    if nbody.transfer == 'CLASS':
        chain @= borg.forward.model_lib.M_PRIMORDIAL_AS(box)
        transfer_class = borg.forward.model_lib.M_TRANSFER_CLASS(
            box, opts=dict(a_transfer=1.0))
        transfer_class.setModelParams({"extra_class_arguments": extra_class})
        chain @= transfer_class
    elif nbody.transfer == 'EH':
        chain @= borg.forward.model_lib.M_PRIMORDIAL(
            box, opts=dict(a_final=1.0))
        chain @= borg.forward.model_lib.M_TRANSFER_EHU(
            box, opts=dict(reverse_sign=True))
    else:
        raise NotImplementedError

    pm = borg.forward.model_lib.M_PM_CIC(
        box,
        opts=dict(a_initial=1.0, a_final=nbody.af,
                  do_rsd=False,
                  supersampling=nbody.supersampling,
                  part_factor=1.01,
                  forcesampling=nbody.B,
                  pm_start_z=nbody.zi,
                  pm_nsteps=nbody.N_steps,
                  tcola=nbody.COLA)
    )
    chain @= pm
    chain.setAdjointRequired(False)

    chain.setCosmoParams(cpar)

    # forward model
    logging.info('Running forward...')
    chain.forwardModel_v2(wn)

    Npart = pm.getNumberOfParticles()
    rho = np.empty(chain.getOutputBoxModel().N)
    pos = np.empty(shape=(Npart, 3))
    vel = np.empty(shape=(Npart, 3))
    chain.getDensityFinal(rho)
    pm.getParticlePositions(pos)
    pm.getParticleVelocities(vel)

    vel *= 100  # km/s

    return rho, pos, vel


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
    cpar = build_cosmology(*cfg.nbody.cosmo)

    # Get ICs
    wn = get_ICs(cfg)

    # Run
    rho, pos, vel = run_density(wn, cpar, cfg)

    # Calculate velocity field
    fvel = None
    if cfg.nbody.save_velocities:
        fvel = vfield_CIC(pos, vel, cfg)
        # convert from comoving -> peculiar velocities
        fvel *= (1 + cfg.nbody.zf)

    # Save
    outdir = get_source_path(cfg, "borgpm", check=False)
    save_nbody(outdir, rho, fvel, pos, vel,
               cfg.nbody.save_particles, cfg.nbody.save_velocities)
    with open(pjoin(outdir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info("Done!")


if __name__ == '__main__':
    main()
