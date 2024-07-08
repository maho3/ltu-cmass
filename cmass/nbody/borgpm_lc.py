"""
Simulate density field using BORG PM models. This script is similar to
cmass.nbody.borgpm, with two major differences:
* It doesn't manage MPI processes. TODO: Implement this!
* It saves multiple snapshots of the density field and velocity field.

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

from os.path import join as pjoin
import numpy as np
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import aquila_borg as borg
from ..utils import get_source_path, timing_decorator, load_params
from .tools import (
    get_ICs, save_nbody, rho_and_vfield)
from .tools_borg import (
    build_cosmology, transfer_EH, transfer_CLASS, run_transfer,
    BorgNotifier)


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


@timing_decorator
def run_density(wn, cpar, cfg, outdir=None):
    nbody = cfg.nbody
    N = nbody.N*nbody.supersampling

    # initialize box and chain
    box = borg.forward.BoxModel()
    box.L = 3*(nbody.L,)
    box.N = 3*(N,)

    chain = borg.forward.ChainForwardModel(box)
    if nbody.transfer == 'CLASS':
        chain = transfer_CLASS(chain, box, cpar)
    elif nbody.transfer == 'EH':
        chain = transfer_EH(chain, box)
    else:
        raise NotImplementedError(
            f'Transfer function "{nbody.transfer}" not implemented.')

    pm = borg.forward.model_lib.M_PM_CIC(
        box,
        opts=dict(
            a_initial=1.0,  # ignored, reset by transfer fn
            a_final=nbody.af,
            do_rsd=False,
            supersampling=1,
            part_factor=1.01,
            forcesampling=nbody.B,
            pm_start_z=nbody.zi,
            pm_nsteps=nbody.N_steps,
            tcola=nbody.COLA
        )
    )

    if hasattr(nbody, 'asave'):
        asave = nbody.asave
    else:
        asave = []

    noti = BorgNotifier(
        asave=asave, N=cfg.nbody.N, L=cfg.nbody.L,
        omega_m=cpar.omega_m, h=cpar.h, outdir=outdir)
    pm.setStepNotifier(
        noti,
        with_particles=True
    )

    chain @= pm
    chain.setAdjointRequired(False)

    chain.setCosmoParams(cpar)

    # forward model
    logging.info('Running forward...')
    chain.forwardModel_v2(wn)

    Npart = pm.getNumberOfParticles()
    pos = np.empty(shape=(Npart, 3))
    vel = np.empty(shape=(Npart, 3))
    pm.getParticlePositions(pos)
    pm.getParticleVelocities(vel)

    vel *= 100  # km/s

    rho, fvel = rho_and_vfield(
        pos, vel,
        Ngrid=nbody.N,
        BoxSize=nbody.L,
        MAS='CIC',
        omega_m=cpar.omega_m,
        h=cpar.h
    )

    return rho, fvel


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

    # Output directory
    outdir = get_source_path(cfg, "borgpm", check=False)
    os.makedirs(outdir, exist_ok=True)

    # Setup cosmology
    cpar = build_cosmology(*cfg.nbody.cosmo)

    # Get ICs
    # Note: these are loaded in all ranks, to minimize MPI memory usage
    wn = get_ICs(cfg)
    wn *= -1  # BORG uses opposite sign

    # Run density field
    rho, fvel = run_density(wn, cpar, cfg, outdir=outdir)

    # Apply transfer fn to ICs (for CHARM)
    if cfg.nbody.save_transfer:
        rho_transfer = run_transfer(wn, cpar, cfg)
        np.save(pjoin(outdir, 'rho_transfer.npy'), rho_transfer)
        del rho_transfer

    # Save
    save_nbody(outdir, rho, fvel, pos=None, vel=None,
               save_particles=cfg.nbody.save_particles,
               save_velocities=cfg.nbody.save_velocities)
    with open(pjoin(outdir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info("Done!")


if __name__ == '__main__':
    main()
