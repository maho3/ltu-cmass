"""
Simulate density field using Borg2LPT.

Requires:
    - borg

Input:
    - index: index of the cosmological parameters in the
        latin_hypercube_params_bonus.txt file

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
import borg
from ..tools.utils import get_global_config, setup_logger, timing_decorator


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='borg2lpt')


# define fucntions
@timing_decorator
def load_params(index, cosmofile):
    if index == "fid":
        return [0.3175, 0.049, 0.6711, 0.9624, 0.834]
    with open(cosmofile, 'r') as f:
        content = f.readlines()[index+1]
    content = [np.float64(x) for x in content.split()]
    return content


def build_cosmology(pars):
    cpar = borg.cosmo.CosmologicalParameters()
    cpar.default()
    cpar.omega_m, cpar.omega_b, cpar.h, cpar.n_s, cpar.sigma8 = pars
    cpar.omega_q = 1.0 - cpar.omega_m
    return cpar


def transfer_EH(chain, box, ai):
    chain.addModel(borg.forward.models.Primordial(box, ai))
    chain.addModel(borg.forward.models.EisensteinHu(box))


def transfer_CLASS(chain, box, cpar, ai):
    # not currently used
    sigma8_true = np.copy(cpar.sigma8)
    cpar.sigma8 = 0
    cpar.A_s = 2.3e-9  # will be modified to correspond to correct sigma
    cosmo = borg.cosmo.ClassCosmo(
        cpar, k_per_decade=10, k_max=50, extra={'YHe': '0.24'})
    cosmo.computeSigma8()  # will compute sigma for the provided A_s
    cos = cosmo.getCosmology()
    # Update A_s
    cpar.A_s = (sigma8_true/cos['sigma_8'])**2*cpar.A_s
    # Add primordial fluctuations
    chain.addModel(borg.forward.model_lib.M_PRIMORDIAL_AS(box))
    # Add CLASS transfer function
    transfer_class = borg.forward.model_lib.M_TRANSFER_CLASS(
        box, opts={"a_transfer": ai, "use_class_sign": False})
    transfer_class.setModelParams(
        {"extra_class_arguments": {"YHe": "0.24", "z_max_pk": "200"}})
    chain.addModel(transfer_class)


@timing_decorator
def gen_ICs(N):
    """Generate ICs in Fourier space."""
    ic = np.fft.rfftn(np.random.randn(N, N, N))/N**(1.5)
    return ic


@timing_decorator
def load_ICs(path_to_ic, N):
    """Loading in Fourier space."""
    logging.info(f"Loading ICs from {path_to_ic}...")
    num_modes_last_d = N // 2 + 1
    with open(path_to_ic, 'rb') as f:
        # num_read = np.fromfile(f, np.uint32, 1)[0]
        modes = np.fromfile(f, np.complex128, -1)
        modes = modes.reshape((N, N, num_modes_last_d))
    return modes


@timing_decorator
def run_density(
    ic,
    L, N, supersampling,
    ai, af,
    cpar,
    transfer='EH'
):
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
    lpt = borg.forward.models.Borg2Lpt(
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
def save(savedir, rho, pos, vel):
    os.makedirs(savedir, exist_ok=True)
    np.save(pjoin(savedir, 'rho.npy'), rho)
    np.save(pjoin(savedir, 'ppos.npy'), pos)
    np.save(pjoin(savedir, 'pvel.npy'), vel)
    logging.info(f'Saved to {savedir}.')


@timing_decorator
def main():
    # Reduce verbosity
    console = borg.console()
    console.setVerboseLevel(1)

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--matchIC', action='store_true')
    args = parser.parse_args()

    # Set manually
    L = 3000  # Mpc/h
    N = 384  # number of grid points
    zi = 127  # initial redshift
    zf = 0.  # final redshift
    supersampling = 1
    transfer = 'EH'  # Transfer function 'CLASS' or 'EH'

    ai = 1/(1+zi)
    af = 1/(1+zf)

    # Set up cosmo
    content = load_params(args.lhid, glbcfg['cosmofile'])
    logging.info(f'Cosmology parameters: {content}')
    cpar = build_cosmology(content)

    # Set up output directory
    outdir = pjoin(glbcfg['wdir'], 'borg2lpt',
                   f'L{L}-N{N}', f'{args.lhid}')
    logging.info(f'I will save to: {outdir}.')

    # Get ICs
    if args.matchIC:
        path_to_ic = pjoin(glbcfg['wdir'],
                           f'wn/N{N}/wn_{args.lhid}.dat')
        # path_to_ic = pjoin(glbcfg['wdir'],
        #                    f'borg2lpt/ICs/wn-N{N}/wn_{args.lhid}.dat')
        ic = load_ICs(path_to_ic, N)
    else:
        ic = gen_ICs(N)

    # Run
    rho, pos, vel = run_density(
        ic,
        L, N, supersampling,
        ai, af,
        cpar,
        transfer=transfer
    )

    # Save
    logging.info('Saving...')
    save(outdir, rho, pos, vel)


if __name__ == '__main__':
    main()
