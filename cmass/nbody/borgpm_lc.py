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
from .tools import (
    gen_white_noise, load_white_noise, save_nbody, rho_and_vfield)
from .tools_borg import build_cosmology, transfer_EH, transfer_CLASS


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

    # initialize box and chain
    box = borg.forward.BoxModel()
    box.L = 3*(nbody.L,)
    box.N = 3*(nbody.N,)

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
            a_initial=1.0,
            a_final=nbody.af,
            do_rsd=False,
            supersampling=nbody.supersampling,
            part_factor=1.01,
            forcesampling=nbody.B,
            pm_start_z=nbody.zi,
            pm_nsteps=nbody.N_steps,
            tcola=nbody.COLA
        )
    )

    class notifier:
        def __init__(self, asave):
            self.step_id = 0
            self.asave = asave
            self.rhos = {}
            self.fvels = {}

        def assign_snap(self):
            # after learning the step intervals, assign where to save snaps
            asteps = np.arange(self.a1, 1, self.da)
            tosave = [
                np.argmin(np.abs(asteps - a)) for a in self.asave
            ]
            tosave = np.unique(tosave)
            self.tosave = tosave

        def __call__(self, a, Np, ids, poss, vels):
            self.step_id += 1
            if self.step_id == 1:  # ignore initial step
                return
            elif self.step_id == 2:  # save the first step
                self.a1 = a
                return
            elif self.step_id == 3:  # learn the step intervals
                self.da = a - self.a1
                self.assign_snap()
            if self.step_id-2 not in self.tosave:  # ignore intermediate steps
                return
            logging.info(f"Saving snap a={a:.6f}, step {self.step_id}")
            rho, fvel = rho_and_vfield(
                poss, vels,
                Ngrid=nbody.N,
                BoxSize=nbody.L,
                MAS='CIC',
                omega_m=cpar.omega_m,
                h=cpar.h
            )
            self.rhos[a] = rho
            self.fvels[a] = fvel

    if hasattr(nbody, 'zsave'):
        zsave = np.array(nbody.zsave)
        asave = 1/(1+zsave)
    else:
        asave = []

    noti = notifier(asave=asave)
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

    snapshots = {
        'rhos': noti.rhos,
        'fvels': noti.fvels
    }

    vel *= 100  # km/s

    rho, fvel = rho_and_vfield(
        pos, vel,
        Ngrid=nbody.N,
        BoxSize=nbody.L,
        MAS='CIC',
        omega_m=cpar.omega_m,
        h=cpar.h
    )

    return rho, fvel, snapshots


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
    rho, fvel, snapshots = run_density(wn, cpar, cfg)

    # Save
    outdir = get_source_path(cfg, "borgpm", check=False)
    save_nbody(outdir, rho, fvel, pos=None, vel=None,
               save_particles=cfg.nbody.save_particles,
               save_velocities=cfg.nbody.save_velocities)
    np.savez(pjoin(outdir, 'snapshots.npz'), **snapshots)
    with open(pjoin(outdir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f)
    logging.info("Done!")


if __name__ == '__main__':
    main()
