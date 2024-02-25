"""
Simulate density field using jax lpt.

Requires:
    - jax_lpt

This script is outdated due to the deprecation of jax_lpt to
alpaga (https://bitbucket.org/aquila-consortium/alpaga/src/main/).

TODO: Update this script to use alpaga.
TODO: Update this script to use hydra for configuration.
"""


import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '16'  # noqa, must go before jax
os.environ["PYBORG_QUIET"] = "yes"  # noqa

import argparse
import logging
from os.path import join as pjoin
import aquila_borg as borg
import numpy as np
from jax_lpt import lpt, simgrid
from ..utils import (attrdict, get_global_config, get_source_path,
                     setup_logger, timing_decorator, load_params)
from .tools import gen_white_noise, load_white_noise, save_nbody
from .tools_borg import build_cosmology, transfer_EH, transfer_CLASS


# Reduce verbosity
console = borg.console()
console.setVerboseLevel(1)

# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='jaxlpt')


def build_config():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-L', type=int, default=3000)  # side length of box in Mpc/h
    parser.add_argument(
        '-N', type=int, default=384)  # number of grid points on one side
    parser.add_argument(
        '--lhid', type=int, required=True)  # which cosmology to use
    parser.add_argument(
        '--order', type=int, default=2)  # LPT order (1 or 2)
    parser.add_argument(
        '--matchIC', action='store_true')  # whether to match ICs to file
    args = parser.parse_args()

    supersampling = 1  # supersampling factor
    transfer = 'EH'    # transfer function 'CLASS' or 'EH
    zi = 127           # initial redshift
    zf = 0.55          # final redshift (default=CMASS)
    ai = 1 / (1 + zi)  # initial scale factor
    af = 1 / (1 + zf)  # final scale factor

    quijote = False  # whether to match ICs to Quijote (True) or custom (False)
    if quijote:
        assert args.L == 1000  # enforce same size of quijote

    # load cosmology
    cosmo = load_params(args.lhid, glbcfg['cosmofile'])

    return attrdict(
        L=args.L, N=args.N, supersampling=supersampling, transfer=transfer,
        lhid=args.lhid, order=args.order, matchIC=args.matchIC,
        zi=zi, zf=zf, ai=ai, af=af,
        quijote=quijote, cosmo=cosmo
    )


def get_ICs(N, lhid, matchIC, quijote):
    if matchIC:
        path_to_ic = pjoin(glbcfg['wdir'], f'wn/N{N}/wn_{lhid}.dat')
        if quijote:
            path_to_ic = pjoin(glbcfg['wdir'],
                               f'borg-quijote/ICs/wn-N{N}'
                               f'wn_{lhid}.dat')
        return load_white_noise(path_to_ic, N, quijote=quijote)
    else:
        return gen_white_noise(N)


@timing_decorator
def generate_initial_density(L, N, cpar, ai, ic, transfer):
    box = borg.forward.BoxModel()
    box.L = (L, L, L)
    box.N = (N, N, N)

    chain = borg.forward.ChainForwardModel(box)
    chain.addModel(borg.forward.models.HermiticEnforcer(box))

    if transfer == 'CLASS':
        transfer_CLASS(chain, box, cpar, ai)
    elif transfer == 'EH':
        transfer_EH(chain, box, ai)

    chain.setCosmoParams(cpar)
    chain.forwardModel_v2(ic)
    rho = np.empty(chain.getOutputBoxModel().N)
    chain.getDensityFinal(rho)
    return rho


@timing_decorator
def run_density(ic, L, N, ai, af, cpar, order, transfer="EH"):
    # Initialize the simulation box
    box = simgrid.Box(L, N)

    # Initial density at initial scale-factor
    rho_init = generate_initial_density(L, N, cpar, ai, ic, transfer)

    # JAX-2LPT model
    if order == 1:
        modelclass = lpt.Jax1LptSolver
    elif order == 2:
        modelclass = lpt.Jax2LptSolver
    else:
        raise NotImplementedError(f"Order {order} not implemented.")
    fwd = modelclass(box, cpar, ai, af, with_velocities=True)

    print("Running forward...")
    rho = fwd.run(rho_init)  # density contrast
    pos = fwd.get_positions()  # particle positions (Mpc/h)
    vel = fwd.get_velocities()  # particle velocities (km/s)

    return rho, pos.T, vel.T


@timing_decorator
def main():
    # Build run config
    cfg = build_config()
    logging.info(f'Running with cosmology: {cfg.cosmo}')

    # Setup
    cpar = build_cosmology(*cfg.cosmo)

    # Get ICs
    wn = get_ICs(cfg.N, cfg.lhid, cfg.matchIC, cfg.quijote)

    # Run
    rho, pos, vel = run_density(
        wn, cfg.L, cfg.N, cfg.ai, cfg.af, cpar, cfg.order, cfg.transfer)

    # Save
    outdir = get_source_path(
        glbcfg["wdir"], f"jax{cfg.order}lpt",
        cfg.L, cfg.N, cfg.lhid, check=False)
    os.makedirs(outdir, exist_ok=True)
    save_nbody(outdir, rho, pos, vel)
    cfg.save(pjoin(outdir, 'config.json'))


if __name__ == "__main__":
    main()
