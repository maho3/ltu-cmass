"""
Simulate density field using Borg2LPT.

Requires:
    - borg

Params:
    - index: index of the cosmological parameters in the
        latin_hypercube_params_bonus.txt file
    - order: LPT order (1 or 2)
    - matchIC: whether to match ICs to file

Output:
    - rho: density field
    - ppos: particle positions
    - pvel: particle velocities
"""


import os
os.environ["PYBORG_QUIET"] = "yes"  # noqa

from os.path import join as pjoin
import argparse
import numpy as np
import logging
import aquila_borg as borg
from ..utils import (attrdict, get_global_config, get_source_path,
                     setup_logger, timing_decorator, load_params)
from .tools import gen_white_noise, load_white_noise, save_nbody
from .tools_borg import build_cosmology, transfer_EH, transfer_CLASS


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='borglpt')


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
def run_density(ic, L, N, supersampling, ai, af, cpar, order, transfer='EH'):
    # initialize box and chain
    box = borg.forward.BoxModel()
    box.L = (L, L, L)
    box.N = (N, N, N)

    chain = borg.forward.ChainForwardModel(box)
    chain.addModel(borg.forward.models.HermiticEnforcer(box))

    if transfer == 'CLASS':
        transfer_CLASS(chain, box, cpar, ai)
    elif transfer == 'EH':
        transfer_EH(chain, box, ai)

    # add lpt
    if order == 1:
        modelclass = borg.forward.models.BorgLpt
    elif order == 2:
        modelclass = borg.forward.models.Borg2Lpt
    else:
        raise NotImplementedError(f'Order {order} not implemented.')
    lpt = modelclass(
        box=box, box_out=box,
        ai=ai, af=af,
        supersampling=supersampling
    )
    chain.addModel(lpt)
    chain.setCosmoParams(cpar)

    # forward model
    logging.info('Running forward...')
    chain.forwardModel_v2(ic)

    Npart = lpt.getNumberOfParticles()
    rho = np.empty(chain.getOutputBoxModel().N)
    pos = np.empty(shape=(Npart, 3))
    vel = np.empty(shape=(Npart, 3))
    chain.getDensityFinal(rho)
    lpt.getParticlePositions(pos)
    lpt.getParticleVelocities(vel)

    return rho, pos, vel


@timing_decorator
def main():
    # Build run config
    cfg = build_config()
    logging.info(f'Running with config: {cfg.cosmo}')

    # Setup
    cpar = build_cosmology(*cfg.cosmo)

    # Get ICs
    wn = get_ICs(cfg.N, cfg.lhid, cfg.matchIC, cfg.quijote)

    # Run
    rho, pos, vel = run_density(
        wn, cfg.L, cfg.N, cfg.supersampling,
        cfg.ai, cfg.af, cpar, cfg.order, cfg.transfer)

    # Save
    outdir = get_source_path(
        glbcfg["wdir"], f"borg{cfg.order}lpt", cfg.L, cfg.N, check=False)
    save_nbody(outdir, rho, pos, vel)
    cfg.save(pjoin(outdir, 'config.json'))


if __name__ == '__main__':
    main()
