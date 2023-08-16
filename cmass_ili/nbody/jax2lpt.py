import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '16'  # noqa
os.environ["PYBORG_QUIET"] = "yes"  # noqa

import argparse
import logging
from os.path import join as pjoin
import borg
import numpy as np
from jax_lpt import lpt, simgrid, utils
from ..tools.utils import get_global_config, setup_logger, timing_decorator


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='jax2lpt')


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


@timing_decorator
def gen_ICs(N):
    """Generate ICs in Fourier space."""
    ic = np.fft.rfftn(np.random.randn(N, N, N)) / N ** (1.5)
    return ic


@timing_decorator
def load_ICs(path_to_ic, N):
    """Loading in Fourier space."""
    logging.info(f"Loading ICs from {path_to_ic}...")
    num_modes_last_d = N // 2 + 1
    with open(path_to_ic, "rb") as f:
        # num_read = np.fromfile(f, np.uint32, 1)[0]
        modes = np.fromfile(f, np.complex128,
                            -1).reshape((N, N, num_modes_last_d))
    return modes


@timing_decorator
def run_density(ic, L, N, ai, af, cpar, transfer="EH"):
    # Initialize the simulation box
    box = simgrid.Box(L, N)

    # Initial density at initial scale-factor
    rho_init = utils.generate_initial_density(L, N, cpar, ai, ic, transfer)

    # JAX-2LPT model
    fwd = lpt.Jax2LptSolver(box, cpar, ai, af, with_velocities=True)

    print("Running forward...")
    rho = fwd.run(rho_init)
    pos = fwd.get_positions()
    vel = fwd.get_velocities()

    return rho, pos.T, vel.T


@timing_decorator
def save(savedir, rho, pos, vel):
    os.makedirs(savedir, exist_ok=True)
    np.save(pjoin(savedir, "rho.npy"), rho)
    np.save(pjoin(savedir, "ppos.npy"), pos)
    np.save(pjoin(savedir, "pvel.npy"), vel)
    logging.info(f"Saved to {savedir}.")


@timing_decorator
def main():
    # Reduce verbosity
    console = borg.console()
    console.setVerboseLevel(1)

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lhid", type=int, required=True)
    parser.add_argument("--matchIC", action="store_true")
    args = parser.parse_args()

    # Set manually
    L = 3000  # Mpc/h
    N = 384  # number of grid points
    zi = 127  # initial redshift
    zf = 0.0  # final redshift
    ai = 1 / (1 + zi)
    af = 1 / (1 + zf)
    transfer = "EH"  # transfer function 'CLASS' or 'EH'

    # Set up cosmo
    content = load_params(args.lhid, glbcfg["cosmofile"])
    logging.info(f"Cosmology parameters: {content}")
    cpar = build_cosmology(content)

    # Set up output directory
    outdir = pjoin(glbcfg["wdir"], "jaxlpt-quijote",
                   f"latin_hypercube_HR-L{L}-N{N}", f"{args.lhid}")
    logging.info(f"I will save to: {outdir}.")

    # Get ICs
    if args.matchIC:
        path_to_ic = pjoin(glbcfg['wdir'],
                           f'wn/N{N}/wn_{args.lhid}.dat')
        # path_to_ic = pjoin(glbcfg['wdir'],
        #                    f'borg-quijote/ICs/wn-N{N}/wn_{args.lhid}.dat')
        ic = load_ICs(path_to_ic, N)
    else:
        ic = gen_ICs(N)

    # Run
    rho, pos, vel = run_density(ic, L, N, ai, af, cpar, transfer)

    # Save
    save(outdir, rho, pos, vel)


if __name__ == "__main__":
    main()
