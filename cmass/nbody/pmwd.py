import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '16'  # noqa

from pmwd import (
    Configuration,
    Cosmology,
    boltzmann,
    white_noise, linear_modes,
    lpt,
    nbody,
    scatter,
)
import logging
import numpy as np
import argparse
from os.path import join as pjoin
from ..tools.utils import get_global_config, setup_logger, timing_decorator


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='pmwd')


# define fucntions
@timing_decorator
def load_params(index, cosmofile):
    if index == "fid":
        return [0.3175, 0.049, 0.6711, 0.9624, 0.834]
    with open(cosmofile, 'r') as f:
        content = f.readlines()[index+1]
    content = [np.float64(x) for x in content.split()]
    return content


@timing_decorator
def load_white_noise(path_to_ic, N):
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
    conf,
    cosmo,
):
    # initialize box and chain
    ptcl, obsvbl = lpt(ic, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    rho = scatter(ptcl, conf)
    pos = np.array(ptcl.pos())
    vel = ptcl.vel
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
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--matchIC', action='store_true')
    args = parser.parse_args()

    content = load_params(args.lhid, glbcfg['cosmofile'])
    logging.info(f'Cosmology parameters: {content}')

    # Set manually
    L = 3000  # Mpc/h
    N = 384  # number of grid points
    supersampling = 1
    ptcl_spacing = L / N
    ptcl_grid_shape = (N,)*3
    conf = Configuration(ptcl_spacing, ptcl_grid_shape,
                         a_start=1/64., a_nbody_num=16,
                         mesh_shape=supersampling)
    cosmo = Cosmology.from_sigma8(
        conf, sigma8=content[4], n_s=content[3], Omega_m=content[0],
        Omega_b=content[1], h=content[2])

    if args.matchIC:
        path_to_ic = pjoin(glbcfg['wdir'],
                           f'wn/N{N}/wn_{args.lhid}.dat')
        # path_to_ic = pjoin(glbcfg['wdir'],
        #                    f'borg-quijote/ICs/wn-N{N}/wn_{args.lhid}.dat')
        modes = load_white_noise(path_to_ic, N)
    else:
        modes = white_noise(args.seed, conf)
    cosmo = boltzmann(cosmo, conf)
    ic = linear_modes(modes, cosmo, conf)

    # Run
    rho, pos, vel = run_density(ic, conf, cosmo)
    rho -= 1  # make it zero mean
    vel *= 100 * cosmo.h  # km/s
    vel *= 2  # Approx rescaling... TODO: Figure out why!

    # Save
    outdir = pjoin(glbcfg['wdir'], 'pmwd',
                   f'L{L}-N{N}', f'{args.lhid}')
    save(outdir, rho, pos, vel)


if __name__ == '__main__':
    main()
