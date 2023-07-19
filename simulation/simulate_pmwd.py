
from tools.utils import get_global_config, get_logger, timing_decorator
from pmwd.spec_util import powspec
from pmwd.vis_util import simshow
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
import os
from os.path import join as pjoin
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'


# define fucntions
@timing_decorator
def load_params(index):
    if index == "fid":
        return [0.3175, 0.049, 0.6711, 0.9624, 0.834]
    with open('latin_hypercube_params_bonus.txt', 'r') as f:
        content = f.readlines()[index+1]
    content = [np.float64(x) for x in content.split()]
    return content


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
        num_read = np.fromfile(f, np.uint32, 1)[0]
        modes = np.fromfile(f, np.complex128, num_read).reshape(
            (N, N, num_modes_last_d))
    return modes


@timing_decorator
def run_density(
    ic,
    conf,
    cosmo,
):
    # initialize box and chain
    cosmo = boltzmann(cosmo, conf)
    ptcl, obsvbl = lpt(ic, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    rho = scatter(ptcl, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    pos = ptcl.disp
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
    # Load global configuration
    glbcfg = get_global_config()
    get_logger(glbcfg['logdir'])

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lhid', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--matchIC', action='store_true')
    args = parser.parse_args()

    content = load_params(args.lhid)
    logging.info(f'Cosmology parameters: {content}')

    # Set manually
    L = 3000  # Mpc/h
    N = 384  # number of grid points
    supersampling = 2
    ptcl_spacing = L / N
    ptcl_grid_shape = (N,)*3
    conf = Configuration(ptcl_spacing, ptcl_grid_shape,
                         mesh_shape=supersampling)
    if args.seed:
        seed = args.seed
    else:
        seed = 0
    cosmo = Cosmology.from_sigma8(
        conf, content[4], n_s=content[3], Omega_m=content[0],
        Omega_b=content[1], h=content[2])
    if args.matchIC:
        path_to_ic = pjoin(glbcfg['wdir'],
                           f'borg-quijote/ICs/wn_{args.lhid}.dat')
        ic = load_ICs(path_to_ic, N)
    else:
        modes = white_noise(seed, conf)
        cosmo = boltzmann(cosmo, conf)
        ic = linear_modes(modes, cosmo, conf)

    outdir = pjoin(glbcfg['wdir'], 'pmwd-quijote',
                   f'latin_hypercube_HR-L{L}-N{N}', f'{args.lhid}')
    # Run
    rho, pos, vel = run_density(
        ic,
        conf,
        cosmo
    )

    # Save
    save(outdir, rho, pos, vel)


if __name__ == '__main__':
    main()
