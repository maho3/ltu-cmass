import os  # noqa
os.environ['OPENBLAS_NUM_THREADS'] = '16'  # noqa, must go before jax

from pmwd import (
    Configuration,
    Cosmology,
    boltzmann,
    linear_modes,
    lpt,
    nbody,
    scatter,
)
import logging
import numpy as np
import argparse
from os.path import join as pjoin
from ..utils import (attrdict, get_global_config, setup_logger,
                     timing_decorator, load_params)
from .tools import gen_white_noise, load_white_noise, save_nbody


# Load global configuration and setup logger
glbcfg = get_global_config()
setup_logger(glbcfg['logdir'], name='pmwd')


def build_config():
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--lhid', type=int, required=True)  # which cosmology to use
    parser.add_argument(
        '--matchIC', action='store_true')  # whether to match ICs to file
    args = parser.parse_args()

    L = 3000           # length of box in Mpc/h
    N = 384            # number of grid points on one side
    N_steps = 16       # number of PM steps
    supersampling = 1  # supersampling factor
    zi = 127           # initial redshift
    zf = 0.0           # final redshift
    ai = 1 / (1 + zi)  # initial scale factor
    af = 1 / (1 + zf)  # final scale factor

    quijote = False  # whether to match ICs to Quijote (True) or custom (False)
    if quijote:
        assert L == 1000  # enforce same size of quijote

    # load cosmology
    cosmo = load_params(args.lhid, glbcfg['cosmofile'])

    return attrdict(
        L=L, N=N, N_steps=N_steps, supersampling=supersampling,
        lhid=args.lhid, matchIC=args.matchIC,
        zi=zi, zf=zf, ai=ai, af=af,
        quijote=quijote, cosmo=cosmo
    )


def configure_pmwd(L, N, N_steps, supersampling, ai, af, cosmo):
    ptcl_spacing = L/N
    ptcl_grid_shape = (N,)*3
    pmconf = Configuration(ptcl_spacing, ptcl_grid_shape,
                           a_start=ai, a_stop=af,
                           a_nbody_maxstep=(af-ai)/N_steps,
                           mesh_shape=supersampling)
    pmcosmo = Cosmology.from_sigma8(
        pmconf, sigma8=cosmo[4], n_s=cosmo[3], Omega_m=cosmo[0],
        Omega_b=cosmo[1], h=cosmo[2])
    pmcosmo = boltzmann(pmcosmo, pmconf)
    return pmconf, pmcosmo


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
def run_density(wn, pmconf, pmcosmo):
    ic = linear_modes(wn, pmcosmo, pmconf)
    ptcl, obsvbl = lpt(ic, pmcosmo, pmconf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, pmcosmo, pmconf)

    rho = scatter(ptcl, pmconf)
    pos = np.array(ptcl.pos())
    vel = ptcl.vel

    rho -= 1  # make it zero mean
    vel *= 100  # km/s
    return rho, pos, vel


@timing_decorator
def main():
    # Build run config
    cfg = build_config()
    logging.info(f'Running with config: {cfg}')

    # Setup
    pmconf, pmcosmo = configure_pmwd(
        cfg.L, cfg.N, cfg.N_steps, cfg.supersampling,
        cfg.ai, cfg.af, cfg.cosmo)

    # Get ICs
    wn = get_ICs(cfg.N, cfg.lhid, cfg.matchIC, cfg.quijote)

    # Run
    rho, pos, vel = run_density(wn, pmconf, pmcosmo)

    # Save
    outdir = pjoin(glbcfg['wdir'], 'pmwd',
                   f'L{cfg.L}-N{cfg.N}', f'{cfg.lhid}')
    save_nbody(outdir, rho, pos, vel)
    cfg.save(pjoin(outdir, 'config.json'))


if __name__ == '__main__':
    main()
