"""
Simulate density field using BORG Particle Mesh. Integrates with MPI.

Input:
    - wn: initial white noise field

Output:
    - nbody.h5
        - rho: density contrast field
        - fvel: velocity field
        - pos: particle positions [optional]
        - vel: particle velocities [optional]

NOTE:
    - This works with the private BORG version, available to Aquila members.
    - For MPI implementation, see jobs/mpiborg.sh
"""

import os
os.environ["PYBORG_QUIET"] = "yes"  # noqa
# os.environ["BORG_TBB_NUM_THREADS"] = "4"  # noqa
# os.environ["OMP_NUM_THREADS"] = "4"  # noqa

from os.path import join
import numpy as np
import logging
import hydra
from mpi4py import MPI

from omegaconf import DictConfig, OmegaConf
import aquila_borg as borg
from ..utils import get_source_path, timing_decorator, save_cfg
from .tools import (
    parse_nbody_config, get_ICs, save_nbody, save_transfer,
    rho_and_vfield)
from .tools_borg import (
    build_cosmology, transfer_EH, transfer_CLASS, run_transfer,
    getMPISlice, gather_MPI)


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
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'nbody', 'multisnapshot'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(
        "Logging directory: " +
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    # Output directory
    outdir = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, "borgpm",
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid, check=False
    )
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
            save_transfer(outdir, rho_transfer)
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
            omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2], verbose=False)

        if not cfg.nbody.save_particles:
            pos, vel = None, None

        # Convert to overdensity field
        rho /= np.mean(rho)
        rho -= 1

        # Convert from comoving -> physical velocities
        fvel *= (1 + cfg.nbody.zf)

        # Save
        save_nbody(outdir, cfg.nbody.af, rho, fvel, pos, vel)
        save_cfg(outdir, cfg)
        logging.info("Done!")
    comm.Barrier()
    return


if __name__ == '__main__':
    main()
