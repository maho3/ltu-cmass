import numpy as np
from os.path import join
import hydra
import logging
import os
import bigfile
from omegaconf import DictConfig, OmegaConf
import glob

from ..utils import get_source_path, timing_decorator, save_cfg
from .tools import (
    parse_nbody_config, gen_white_noise, load_white_noise, 
    save_white_noise_grafic, generate_pk_file, rho_and_vfield,
    save_nbody
)

@timing_decorator
def get_ICs(cfg, outdir):

    nbody = cfg.nbody
    N = nbody.N

    # Load the ics in Fourier space
    if nbody.matchIC:
        path_to_ic = f'wn/N{N}/wn_{nbody.lhid}.dat'
        if nbody.quijote:
            path_to_ic = join(cfg.meta.wdir, 'quijote', path_to_ic)
        else:
            path_to_ic = join(cfg.meta.wdir, path_to_ic)
        ic = load_white_noise(path_to_ic, N, quijote=nbody.quijote)
    else:
        ic = gen_white_noise(N, seed=nbody.lhid)

    # Convert to real space
    ic = np.fft.irfftn(ic, norm="ortho")
    
    # Save to file
    filename = join(outdir, "WhiteNoise_grafic")
    save_white_noise_grafic(filename, ic, nbody.lhid)
    
    return


def generate_param_file(cfg, outdir):
    
    output_redshifts = [cfg.nbody.zf]
    output_redshifts_lua = "{" + ", ".join(map(str, output_redshifts)) + "}"
    
    lua_content = f"""
boxsize = {cfg.nbody.L}
nc = {cfg.nbody.N}
B = {cfg.nbody.B}
T = {cfg.nbody.N_steps}
prefix = "{outdir}"
read_grafic = "{join(outdir, 'WhiteNoise_grafic')}"


----------------------------------------
--- This file needs to be concatenated with parameters from run.py ---
----------------------------------------
-------- Time Sequence -------- 
-- linspace: Uniform time steps in a
-- time_step = linspace(0.025, 1.0, 39)
-- logspace: Uniform time steps in loga

time_step = linspace({1 / (1 + cfg.nbody.zi)}, {1 / (1 + cfg.nbody.zf)}, T)
output_redshifts= {output_redshifts_lua}  -- redshifts of output


----------------------------------------
-------- Cosmology -------- 
Omega_m   = {cfg.nbody.cosmo[0]}
h         = {cfg.nbody.cosmo[2]}

-- Start with a power spectrum file
-- Initial power spectrum: k P(k) in Mpc/h units
-- Must be compatible with the Cosmology parameter

read_powerspectrum= prefix .. "/input_power_spectrum.txt"
linear_density_redshift = 0.0 -- the redshift of the linear density field.
-- remove_cosmic_variance = true


----------------------------------------
-------- Approximation Method -------- 

force_mode = "fastpm"
pm_nc_factor = B            -- Particle Mesh grid pm_nc_factor*nc per dimension in the beginning
np_alloc_factor= 2.2      -- Amount of memory allocated for particle


----------------------------------------
-------- Output -------- 

-- Dark matter particle outputs (all particles)
write_snapshot= prefix .. "/fastpm_B{cfg.nbody.B}"
particle_fraction = 1.00

write_fof     = prefix .. "/fof_B{cfg.nbody.B}"
fof_linkinglength = 0.200
fof_nmin = 16

-- 1d power spectrum (raw), without shotnoise correction
write_powerspectrum = prefix .. '/powerspec'
"""
    
    output_file = join(outdir, "parameter_file.lua")
    with open(output_file, 'w') as file:
        file.write(lua_content)
        
    return


@timing_decorator
def run_density(cfg, outdir):
    
    param_file = join(outdir, "parameter_file.lua")
    os.system(f'{cfg.nbody.fastpm_exec} {param_file}')
    
    all_a = glob.glob(join(outdir, f'fastpm_B{cfg.nbody.B}_*'))
    all_a = [a[-a[::-1].index('_'):] for a in all_a]
    af = 1 / (1 + cfg.nbody.zf)
    i = np.argmin(np.abs(af - np.array(all_a, dtype=float)))
    a = all_a[i]
    
    f = bigfile.File(join(outdir, f'fastpm_B{cfg.nbody.B}_{a}'))
    ds = bigfile.Dataset(f['1/'], ['Position', 'Velocity', 'ID'])
    pos = np.array(ds[:]['Position'])
    vel = np.array(ds[:]['Velocity'])
    
    return pos, vel


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

    outdir = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, "fastpm",
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid, check=False
    )
    os.makedirs(outdir, exist_ok=True)

    # Setup power spectrum file if needed
    generate_pk_file(cfg, outdir)

    # Convert ICs to correct format
    get_ICs(cfg, outdir)

    # Generate parameter file
    generate_param_file(cfg, outdir)

    # Run
    pos, vel = run_density(cfg, outdir)
    
    # Calculate velocity field
    rho, fvel = rho_and_vfield(
        pos, vel, cfg.nbody.L, cfg.nbody.N, 'CIC',
        omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2])

    if not cfg.nbody.save_particles:
        pos, vel = None, None

    # Convert from comoving -> physical velocities
    fvel *= (1 + cfg.nbody.zf)

    # Save nbody-type outputs
    save_nbody(outdir, cfg.nbody.af, rho, fvel, pos, vel)
    save_cfg(outdir, cfg)
    
    logging.info("Done!")


if __name__ == '__main__':
    main()

