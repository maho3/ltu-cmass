"""
Simulate density field using BORG PM models.
NOTE: This works with the private BORG version, available to Aquila members.

Note, for MPI implementation, see jobs/mpiborg.sh

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
# os.environ["BORG_TBB_NUM_THREADS"] = "4"  # noqa
# os.environ["OMP_NUM_THREADS"] = "4"  # noqa

from os.path import join as pjoin
import numpy as np
import logging
import hydra
from mpi4py import MPI

from omegaconf import DictConfig, OmegaConf, open_dict
import aquila_borg as borg
from ..utils import get_source_path, timing_decorator, load_params
from .tools import (
    get_ICs, save_nbody, rho_and_vfield)
from .tools_borg import (
    build_cosmology, transfer_EH, transfer_CLASS, run_transfer,
    getMPISlice, gather_MPI)


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
def run_density(wn, cpar, cfg):
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
    chain @= pm
    chain.setAdjointRequired(False)
    chain.setCosmoParams(cpar)

    # forward model
    logging.info('Running forward...')
    chain.forwardModel_v2(wn)

    Npart = pm.getNumberOfParticles()
    pos = np.empty(shape=(Npart, 3), dtype=np.float64)
    vel = np.empty(shape=(Npart, 3), dtype=np.float64)
    pm.getParticlePositions(pos)
    pm.getParticleVelocities(vel)

    pos = pos.astype(np.float32)
    vel = vel.astype(np.float32)

    vel *= 100  # km/s

    del chain, box, pm

    return pos, vel


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load MPI rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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

    # Get MPI slice
    startN0, localN0, _, _ = getMPISlice(cfg)
    wn = wn[startN0:startN0+localN0]

    # Apply transfer fn to ICs (for CHARM)
    if cfg.nbody.save_transfer:
        rho_transfer = run_transfer(wn, cpar, cfg)
        if rank == 0:
            np.save(pjoin(outdir, 'rho_transfer.npy'), rho_transfer)
        del rho_transfer

    # Run density field
    pos, vel = run_density(wn, cpar, cfg)

    # Gather particle positions and velocities
    pos, vel = gather_MPI(pos, vel)
    logging.info(f'rank {rank} done')

    if rank == 0:
        # Calculate density and velocity field
        rho, fvel = rho_and_vfield(
            pos, vel, cfg.nbody.L, cfg.nbody.N, 'CIC',
            omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2], verbose=True)

        # Convert to overdensity field
        rho /= np.mean(rho)
        rho -= 1

        # Convert from comoving -> peculiar velocities
        fvel *= (1 + cfg.nbody.zf)

        # Save
        save_nbody(outdir, rho, fvel, pos, vel,
                   save_particles=cfg.nbody.save_particles,
                   save_velocities=cfg.nbody.save_velocities)
        with open(pjoin(outdir, 'config.yaml'), 'w') as f:
            OmegaConf.save(cfg, f)
        logging.info("Done!")
    comm.Barrier()
    return


if __name__ == '__main__':
    main()
