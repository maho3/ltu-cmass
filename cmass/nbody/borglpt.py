"""
Simulate density field using BORG LPT models.

Requires:
    - borg

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
from mpi4py import MPI

from omegaconf import DictConfig, OmegaConf, open_dict
import aquila_borg as borg
from ..utils import get_source_path, timing_decorator, load_params
from .tools import (
    gen_white_noise, load_white_noise, save_nbody, rho_and_vfield)
from .tools_borg import (
    build_cosmology, transfer_EH, transfer_CLASS, apply_transfer_fn)


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
        return load_white_noise(path_to_ic, N, quijote=nbody.quijote)
    else:
        return gen_white_noise(N, seed=nbody.lhid)


@timing_decorator
def run_density(wn, cpar, cfg):
    nbody = cfg.nbody

    # initialize box and chain
    box = borg.forward.BoxModel()
    box.L = 3*(nbody.L,)
    box.N = 3*(nbody.N,)

    chain = borg.forward.ChainForwardModel(box)
    startN0, localN0, _, _ = chain.getMPISlice()
    if nbody.transfer == 'CLASS':
        chain = transfer_CLASS(chain, box, cpar, a_final=nbody.af)
    elif nbody.transfer == 'EH':
        chain = transfer_EH(chain, box)
    else:
        raise NotImplementedError(
            f'Transfer function "{nbody.transfer}" not implemented.')

    if nbody.order == 1:
        lpt = borg.forward.model_lib.M_LPT_CIC
    elif nbody.order == 2:
        lpt = borg.forward.model_lib.M_2LPT_CIC
    else:
        raise NotImplementedError(f'Order "{nbody.order}" not implemented.')
    lpt = lpt(
        box,
        opts=dict(
            a_initial=1.0,  # ignored, reset by transfer fn
            a_final=nbody.af,
            do_rsd=False,
            supersampling=nbody.supersampling,
            lightcone=False,
            part_factor=1.1
        )
    )
    chain @= lpt

    chain.setCosmoParams(cpar)

    import psutil
    process = psutil.Process()
    print(process.memory_info().rss/1e9)  # in GB

    # forward model
    logging.info('Running forward...')
    chain.forwardModel_v2(wn[startN0:startN0+localN0])

    print(process.memory_info().rss/1e9)  # in GB

    Npart = lpt.getNumberOfParticles()
    pos = np.empty(shape=(Npart, 3))
    vel = np.empty(shape=(Npart, 3))
    lpt.getParticlePositions(pos)
    lpt.getParticleVelocities(vel)

    print(process.memory_info().rss/1e9)  # in GB

    vel *= 100  # km/s

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
    outdir = get_source_path(cfg, f"borg{cfg.nbody.order}lpt", check=False)
    os.makedirs(outdir, exist_ok=True)

    # Setup
    cpar = build_cosmology(*cfg.nbody.cosmo)

    # Get ICs
    wn = get_ICs(cfg)
    wn *= -1  # BORG uses opposite sign

    # Apply transfer fn to ICs (for CHARM)
    if cfg.nbody.save_transfer:
        rho_transfer = apply_transfer_fn(
            wn, cfg.nbody.L, cfg.nbody.N, cpar,
            af=1./(1+99),  # z=99, for CHARM inputs
            transfer=cfg.nbody.transfer)
        rho_transfer = comm.gather(rho_transfer, root=0)
        if rank == 0:
            rho_transfer = np.concatenate(rho_transfer, axis=0)
            np.save(pjoin(outdir, 'rho_transfer.npy'), rho_transfer)
        del rho_transfer

    # Run density field
    pos, vel = run_density(wn, cpar, cfg)

    # Gather particle positions and velocities
    pos = comm.gather(pos, root=0)
    vel = comm.gather(vel, root=0)

    # Post-process and save results
    if rank != 0:
        logging.info(f"Rank {rank} done!")
        return  # Only compute remaining on rank 0
    pos = np.concatenate(pos, axis=0)
    vel = np.concatenate(vel, axis=0)

    # Calculate density and velocity field
    rho, fvel = rho_and_vfield(
        pos, vel, cfg.nbody.L, cfg.nbody.N, 'CIC',
        omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2])

    # Convert to overdensity field
    rho /= np.mean(rho)
    rho -= 1

    # Convert from comoving -> peculiar velocities
    fvel *= (1 + cfg.nbody.zf)

    # Save
    save_nbody(outdir, rho, fvel, pos, vel,
               cfg.nbody.save_particles, cfg.nbody.save_velocities)
    with open(pjoin(outdir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info("Done!")


if __name__ == '__main__':
    main()
