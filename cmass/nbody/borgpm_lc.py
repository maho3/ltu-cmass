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
from omegaconf import DictConfig, OmegaConf
import aquila_borg as borg
from ..utils import get_source_path, timing_decorator
from .tools import (
    parse_nbody_config, get_ICs)
from .tools_borg import (
    build_cosmology, transfer_EH, transfer_CLASS, run_transfer,
    BorgNotifier)


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

    # This is a custom notifier that saves the density and velocity fields
    # at each desired step
    noti = BorgNotifier(
        asave=nbody.asave, N=nbody.N, L=nbody.L,
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

    # Density fields are saved during forward run, so nothing is returned


def delete_outputs(outdir):
    outpath = pjoin(outdir, 'snapshots.h5')
    if os.path.isfile(outpath):
        os.remove(outpath)


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

    # Check if we're in snapshot mode
    if not (hasattr(cfg.nbody, 'snapshot_mode') and cfg.nbody.snapshot_mode):
        raise ValueError("snapshot_mode config is false, but borgpm_lc"
                         "is only for snapshot mode.")

    # Output directory
    outdir = get_source_path(cfg, "borgpm", check=False)
    os.makedirs(outdir, exist_ok=True)

    # Setup cosmology
    cpar = build_cosmology(*cfg.nbody.cosmo)

    # Get ICs
    # Note: these are loaded in all ranks, to minimize MPI memory usage
    wn = get_ICs(cfg)
    wn *= -1  # BORG uses opposite sign

    # Apply transfer fn to ICs (for CHARM)
    if cfg.nbody.save_transfer:
        rho_transfer = run_transfer(wn, cpar, cfg)
        np.save(pjoin(outdir, 'rho_transfer.npy'), rho_transfer)
        del rho_transfer

    # Run and save density field
    delete_outputs(outdir)
    run_density(wn, cpar, cfg, outdir=outdir)

    # Save config
    with open(pjoin(outdir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info("Done!")


if __name__ == '__main__':
    main()
