import numpy as np
from os.path import join
import hydra
import logging
import os
import bigfile
from omegaconf import DictConfig, OmegaConf
import shutil
import subprocess
import h5py

from ..utils import get_source_path, timing_decorator, save_cfg
from .tools import (
    parse_nbody_config, get_ICs,
    save_white_noise_grafic, generate_pk_file, rho_and_vfield,
    save_nbody
)


def save_ICs(cfg, outdir):
    ic = get_ICs(cfg)
    ic = np.fft.irfftn(ic, norm="ortho")

    filename = join(outdir, "WhiteNoise_grafic")
    save_white_noise_grafic(filename, ic, cfg.nbody.lhid)


def generate_param_file(
    L, N, supersampling, B, N_steps,
    zf, asave, save_transfer,
    cosmo, outdir
):

    output_redshifts = -1 + 1./np.array(asave, dtype=float)
    if zf not in output_redshifts:
        output_redshifts += [zf]
    if save_transfer and (99. not in output_redshifts):
        output_redshifts = np.append(output_redshifts, 99.)
    output_redshifts_lua = "{" + ", ".join(map(str, output_redshifts)) + "}"

    lua_content = f"""
boxsize = {L}
nc = {N*supersampling}
B = {B}
T = {N_steps}
prefix = "{outdir}"
read_grafic = "{join(outdir, 'WhiteNoise_grafic')}"


----------------------------------------
--- This file needs to be concatenated with parameters from run.py ---
----------------------------------------
-------- Time Sequence --------
-- linspace: Uniform time steps in a
-- time_step = linspace(0.025, 1.0, 39)
-- logspace: Uniform time steps in loga

time_step = linspace({0.01}, {1 / (1 + zf)}, T)
output_redshifts= {output_redshifts_lua}  -- redshifts to output


----------------------------------------
-------- Cosmology --------
Omega_m   = {cosmo[0]}
h         = {cosmo[2]}

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
np_alloc_factor= 2.2        -- Amount of memory allocated for particle
loglevel = 2


----------------------------------------
-------- Output --------

-- Dark matter particle outputs (all particles)
write_snapshot= prefix .. "/fastpm_B{B}"
particle_fraction = 1.00
"""

    output_file = join(outdir, "parameter_file.lua")
    with open(output_file, 'w') as file:
        file.write(lua_content)

    return


@timing_decorator
def run_density(cfg, outdir):

    # Work out how many cpus to use
    #  Need this to exactly divide N * B
    max_cores = os.cpu_count()
    max_divisible_cores = None
    product = cfg.nbody.N * cfg.nbody.B
    for cores in range(max_cores, 0, -1):
        if product % cores == 0:
            max_divisible_cores = cores
            break

    #  Run FastPM
    param_file = join(outdir, "parameter_file.lua")
    command = f'mpirun -n {max_divisible_cores} {cfg.nbody.fastpm_exec} {param_file}'
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    _ = subprocess.run(command, shell=True, check=True, env=env)


@timing_decorator
def process_transfer(cfg, outdir, delete_files=True):
    with h5py.File(join(outdir, 'transfer.h5'), 'w') as outfile:
        a = 1/(1+99.)  # hardcoded for now, from CHARM training
        logging.info(f"Processing transfer function at a={a:.4f}...")
        snapdir = join(outdir, f'fastpm_B{cfg.nbody.B}_{a:.4f}')
        infile = bigfile.File(snapdir)
        ds = bigfile.Dataset(infile['1/'], ['Position', 'Velocity', 'ID'])
        pos = np.array(ds[:]['Position'])
        vel = np.array(ds[:]['Velocity'])

        rho, _ = rho_and_vfield(
            pos, vel, cfg.nbody.L, cfg.nbody.N, 'CIC',
            omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2])

        outfile.create_dataset('rho', data=rho)

    if delete_files:
        infile.close()
        shutil.rmtree(snapdir)


@timing_decorator
def process_outputs(cfg, outdir, delete_files=True):
    with h5py.File(join(outdir, 'nbody.h5'), 'w') as outfile:
        for a in sorted(cfg.nbody.asave):
            logging.info(f"Processing snapshot at a={a:.4f}...")
            # Extract positions and velocities at a given scale factor
            snapdir = join(outdir, f'fastpm_B{cfg.nbody.B}_{a:.4f}')
            infile = bigfile.File(snapdir)
            ds = bigfile.Dataset(infile['1/'], ['Position', 'Velocity', 'ID'])
            pos = np.array(ds[:]['Position'])
            vel = np.array(ds[:]['Velocity'])

            # Measure density and velocity field
            rho, fvel = rho_and_vfield(
                pos, vel, cfg.nbody.L, cfg.nbody.N, 'CIC',
                omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2])

            # Convert from comoving -> physical velocities
            fvel *= 1/a

            # Save to file
            key = f'{a:.6f}'
            group = outfile.create_group(key)
            group.create_dataset('rho', data=rho)
            group.create_dataset('fvel', data=fvel)

            # Clean up
            if delete_files:
                infile.close()
                shutil.rmtree(snapdir)

    return rho, fvel, pos, vel  # return the last snapshot


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'nbody'])

    # Build run config
    cfg = parse_nbody_config(cfg, lightcone=True)
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(
        "Logging directory: " +
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    # Create output directory
    outdir = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, "fastpm",
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid, check=False
    )
    os.makedirs(outdir, exist_ok=True)

    # Setup power spectrum file if needed
    generate_pk_file(cfg, outdir)

    # Convert ICs to correct format
    save_ICs(cfg, outdir)

    # Generate parameter file
    generate_param_file(
        L=cfg.nbody.L, N=cfg.nbody.N, supersampling=cfg.nbody.supersampling,
        B=cfg.nbody.B, N_steps=cfg.nbody.N_steps,
        zf=cfg.nbody.zf, asave=cfg.nbody.asave,
        save_transfer=cfg.nbody.save_transfer,
        cosmo=cfg.nbody.cosmo, outdir=outdir
    )

    # Run
    logging.info("Running FastPM...")
    run_density(cfg, outdir)

    # Process outputs
    logging.info("Processing outputs...")
    if cfg.nbody.save_transfer:
        process_transfer(cfg, outdir, delete_files=True)
    rho, fvel, pos, vel = process_outputs(cfg, outdir, delete_files=True)
    os.remove(join(outdir, 'WhiteNoise_grafic'))  # remove ICs

    if not cfg.nbody.save_particles:
        pos, vel = None, None

    # Save nbody-type outputs
    # save_nbody(outdir, cfg.nbody.af, rho, fvel, pos, vel)
    # TODO: add a way to append particles to the existing nbody.h5 file
    save_cfg(outdir, cfg)

    logging.info("Done!")


if __name__ == '__main__':
    main()
