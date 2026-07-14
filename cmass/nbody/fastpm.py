import numpy as np
from os.path import join
import hydra
import logging
import os
import time
import bigfile
from omegaconf import DictConfig, OmegaConf
import shutil
import subprocess
import h5py
import multiprocessing as mp

from ..utils import get_source_path, timing_decorator, save_cfg, clean_up
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
    # if zf not in output_redshifts:
    #     output_redshifts = np.append(output_redshifts, zf)
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
output_redshifts = {output_redshifts_lua}  -- redshifts to output


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


def get_mpi_info():
    # copied from Deaglan's pinnochio implementation TODO: merge!!

    if 'PBS_NODEFILE' in os.environ:  # assume PBS job
        with open(os.environ['PBS_NODEFILE'], 'r') as f:
            max_cores = len(f.readlines())
        mpi_args = '--hostfile $PBS_NODEFILE'
    elif 'SLURM_JOB_NODELIST' in os.environ:  # assume Slurm job
        # Use scontrol to get the expanded list of nodes
        node_list = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        node_list = node_list.decode('utf-8').splitlines()

        # Calculate number of tasks
        max_cores = int(os.environ.get('SLURM_NTASKS', 1))

        # Write to a hostfile (optional, if required by your MPI setup)
        hostfile = 'slurm_hostfile'
        with open(hostfile, 'w') as f:
            for node in node_list:
                f.write(f"{node}\n")

        # Set MPI args
        mpi_args = f'--hostfile {hostfile}'
        mpi_args = ''
    else:
        max_cores = os.cpu_count()
        mpi_args = ''

    return max_cores, mpi_args


@timing_decorator
def run_density(cfg, outdir):
    max_cores, mpi_args = get_mpi_info()

    # Work out how many cpus to use
    # Need this to exactly divide N * B
    max_divisible_cores = None
    product = cfg.nbody.N * cfg.nbody.B
    for cores in range(max_cores, 0, -1):
        if product % cores == 0:
            max_divisible_cores = cores
            break
    logging.info(f"Using {max_divisible_cores} cores for FastPM")

    # Use srun on Cray systems, mpirun elsewhere
    param_file = join(outdir, "parameter_file.lua")
    launcher = _get_mpi_launcher()
    command = f'{launcher} -n {max_divisible_cores} {mpi_args} '
    command += f'{cfg.meta.fastpm_exec} {param_file}'
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    _ = subprocess.run(command, shell=True, check=True, env=env)


def _get_mpi_launcher():
    """Return srun on Cray/SLURM systems, mpirun otherwise."""
    import shutil
    if shutil.which("srun") is not None:
        return "srun"
    if shutil.which("mpirun") is not None:
        return "mpirun"
    raise RuntimeError("No MPI launcher (srun or mpirun) found in PATH")


@timing_decorator
def process_transfer(cfg, workdir, outdir, delete_files=True):
    with h5py.File(join(outdir, 'transfer.h5'), 'w') as outfile:
        a = 1/(1+99.)  # hardcoded for now, from CHARM training
        logging.info(f"Processing transfer function at a={a:.4f}...")
        snapdir = join(workdir, f'fastpm_B{cfg.nbody.B}_{a:.4f}')
        infile = bigfile.File(snapdir)
        pos = infile['1/Position'][:]
        vel = infile['1/Velocity'][:]

        # Measure density field
        rho, _ = rho_and_vfield(
            pos, vel, cfg.nbody.L, cfg.nbody.N, 'CIC',
            omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2])

        # Convert to overdensity field
        rho /= np.mean(rho)
        rho -= 1

        outfile.create_dataset('rho', data=rho)

    if delete_files:
        infile.close()
        shutil.rmtree(snapdir)


def process_single_snapshot(cfg, workdir, a, delete_files=True):
    # Idempotent: skip if already converted (e.g. by the streaming watcher)
    outpath = join(workdir, f'nbody_{a:.4f}.h5')
    if os.path.exists(outpath):
        return

    logging.info(f"Reading snapshot at a={a:.4f}...")
    snapdir = join(workdir, f'fastpm_B{cfg.nbody.B}_{a:.4f}')

    if not os.path.isdir(snapdir):
        logging.warning(f"Snapshot at a={a:.4f} not found, skipping...")
        return

    infile = bigfile.File(snapdir)
    # Read columns directly: bigfile is columnar, so this touches only the
    # Position/Velocity blocks (a compound Dataset[:] would read all columns
    # including the unused 8B/particle ID, twice)
    pos = infile['1/Position'][:]  # comoving positions [Mpc/h]
    vel = infile['1/Velocity'][:]  # physical velocities [km/s]

    # Measure density and velocity field
    logging.info(f"Processing snapshot at a={a:.4f}...")
    rho, fvel = rho_and_vfield(
        pos, vel, cfg.nbody.L, cfg.nbody.N, 'CIC',
        omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2])

    # Delete pos, vel
    del pos, vel  # TODO: save these?

    # Convert to overdensity field
    rho /= np.mean(rho)
    rho -= 1

    # Convert from physical -> comoving velocities
    fvel *= 1/a

    # Save to file (write to .tmp then rename, so existence == complete)
    with h5py.File(outpath + '.tmp', 'w') as outfile:
        outfile.create_dataset('rho', data=rho)
        outfile.create_dataset('fvel', data=fvel)
    os.replace(outpath + '.tmp', outpath)


def _snapdir(workdir, cfg, a):
    return join(workdir, f'fastpm_B{cfg.nbody.B}_{a:.4f}')


def _move_snapshots(cfg, workdir, localwork, sentinel, poll=10):
    """Streaming mover: as FastPM finishes each snapshot, move it from the
    shared scratch (workdir) to node-local disk (localwork) to free quota.

    Snapshot i is known complete once FastPM has created snapshot i+1's
    directory (writes are strictly sequential), or once the sentinel file
    exists (FastPM exited).
    """
    asave = sorted(cfg.nbody.asave)
    mover_done = join(localwork, '.mover_done')
    try:
        for i, a in enumerate(asave):
            src = _snapdir(workdir, cfg, a)
            dst = _snapdir(localwork, cfg, a)
            marker = join(localwork, f'.moved_{a:.4f}')
            while True:
                nxt = (i + 1 < len(asave)
                       and os.path.isdir(_snapdir(workdir, cfg, asave[i+1])))
                if os.path.isdir(src) and (nxt or os.path.exists(sentinel)):
                    break
                if os.path.exists(sentinel) and not os.path.isdir(src):
                    logging.warning(
                        f"Mover: snapshot a={a:.4f} never appeared, skipping")
                    src = None
                    break
                time.sleep(poll)
            if src is None:
                continue
            if not os.path.isdir(dst):
                # copy to .tmp then rename, so dst existence == complete copy
                shutil.copytree(src, dst + '.tmp', dirs_exist_ok=True)
                os.rename(dst + '.tmp', dst)
            open(marker, 'w').close()
            shutil.rmtree(src)
            logging.info(f"Mover: snapshot a={a:.4f} moved to local disk")
    finally:
        open(mover_done, 'w').close()


def _convert_snapshots(cfg, localwork, poll=10):
    """Streaming converter: convert moved snapshots to nbody_{a}.h5 fields
    (deleting the raw snapshot) while FastPM is still running."""
    asave = sorted(cfg.nbody.asave)
    mover_done = join(localwork, '.mover_done')
    for a in asave:
        marker = join(localwork, f'.moved_{a:.4f}')
        while not os.path.exists(marker):
            if os.path.exists(mover_done):
                break
            time.sleep(poll)
        if not os.path.exists(marker):
            continue  # mover skipped this one; post-run fallback handles it
        process_single_snapshot(cfg, localwork, a, delete_files=True)
        logging.info(f"Converter: snapshot a={a:.4f} converted")

    if delete_files:
        infile.close()
        shutil.rmtree(snapdir)


@timing_decorator
def process_outputs(cfg, workdir, outdir, delete_files=True,
                    fallback_workdir=None):
    asave = sorted(cfg.nbody.asave)
    rho, fvel = None, None

    with mp.Pool(3) as pool:
        _ = pool.starmap(
            process_single_snapshot,
            [(cfg, workdir, a, delete_files) for a in asave]
        )

    logging.info("Concatenating snapshots...")
    with h5py.File(join(outdir, 'nbody.h5'), 'w') as outfile:
        for a in asave:
            # Read snapshot
            filename = join(workdir, f'nbody_{a:.4f}.h5')
            if not os.path.exists(filename) and fallback_workdir is not None:
                filename = join(fallback_workdir, f'nbody_{a:.4f}.h5')
            with h5py.File(filename, 'r') as infile:
                rho = infile['rho'][:]
                fvel = infile['fvel'][:]

            # Save to file
            key = f'{a:.6f}'
            group = outfile.create_group(key)
            group.create_dataset('rho', data=rho)
            group.create_dataset('fvel', data=fvel)

            # Delete temporary file
            if delete_files:
                os.remove(filename)

    return rho, fvel, None, None  # return the last snapshot


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
@clean_up(hydra)
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'nbody', 'multisnapshot'])

    # Build run config
    cfg = parse_nbody_config(cfg)
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(
        "Logging directory: " +
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))

    # Create persistent output directory (holds only final field maps)
    outdir = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, "fastpm",
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid, check=False
    )
    os.makedirs(outdir, exist_ok=True)

    # Create transient work directory for heavy FastPM intermediates (particle
    # snapshots, ICs, param file). Defaults to wdir if meta.scratchdir is unset.
    # On Slurm, point meta.scratchdir at node-local /tmp to keep the ~76GB of
    # particle snapshots off the shared/quota'd filesystem.
    scratch_base = cfg.meta.get('scratchdir', None) or cfg.meta.wdir
    workdir = get_source_path(
        scratch_base, cfg.nbody.suite, "fastpm",
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid, check=False
    )
    if workdir != outdir and os.path.isdir(workdir):
        shutil.rmtree(workdir)  # clear stale data from a crashed prior run
    os.makedirs(workdir, exist_ok=True)

    try:
        # Setup power spectrum file if needed
        generate_pk_file(cfg, workdir)

        # Convert ICs to correct format
        save_ICs(cfg, workdir)

        # Generate parameter file
        generate_param_file(
            L=cfg.nbody.L, N=cfg.nbody.N, supersampling=cfg.nbody.supersampling,
            B=cfg.nbody.B, N_steps=cfg.nbody.N_steps,
            zf=cfg.nbody.zf, asave=cfg.nbody.asave,
            save_transfer=cfg.nbody.save_transfer,
            cosmo=cfg.nbody.cosmo, outdir=workdir
        )

        # Streaming postprocess: move finished snapshots off the shared
        # scratch to node-local disk (meta.localdir) and convert them to
        # field maps while FastPM is still running. Requires localdir on the
        # node running this driver (post-FastPM steps are single-node).
        stream = (cfg.nbody.get('stream_postprocess', False)
                  and cfg.meta.get('localdir', None) is not None
                  and cfg.nbody.get('postprocess', False)
                  and cfg.multisnapshot)
        if stream:
            localwork = get_source_path(
                cfg.meta.localdir, cfg.nbody.suite, "fastpm",
                cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid, check=False
            )
            os.makedirs(localwork, exist_ok=True)
            sentinel = join(workdir, '.fastpm_done')
            mover = mp.Process(
                target=_move_snapshots, args=(cfg, workdir, localwork, sentinel))
            converter = mp.Process(
                target=_convert_snapshots, args=(cfg, localwork))
            mover.start()
            converter.start()

        # Run
        logging.info("Running FastPM...")
        try:
            run_density(cfg, workdir)
        finally:
            if stream:
                open(sentinel, 'w').close()  # unblock workers even on failure
        os.remove(join(workdir, 'WhiteNoise_grafic'))  # remove ICs

        if stream:
            mover.join()
            converter.join()
            # Fallback: convert in place anything the streaming path missed
            for a in sorted(cfg.nbody.asave):
                if (not os.path.exists(join(localwork, f'nbody_{a:.4f}.h5'))
                        and os.path.isdir(_snapdir(workdir, cfg, a))):
                    logging.warning(
                        f"Streaming missed a={a:.4f}; converting in place")
                    process_single_snapshot(cfg, workdir, a, delete_files=True)

        # Process outputs (snapshots read from workdir, field maps -> outdir)
        logging.info("Processing outputs...")
        if cfg.nbody.save_transfer:
            process_transfer(cfg, workdir, outdir, delete_files=True)
        if 'postprocess' in cfg.nbody and cfg.nbody.postprocess:
            if stream:
                rho, fvel, pos, vel = process_outputs(
                    cfg, localwork, outdir, delete_files=True,
                    fallback_workdir=workdir)
            else:
                rho, fvel, pos, vel = process_outputs(
                    cfg, workdir, outdir, delete_files=True)

        if not cfg.nbody.save_particles:
            pos, vel = None, None

        # Save nbody-type outputs (unnecessary because of process_outputs)
        # save_nbody(outdir, cfg.nbody.af, rho, fvel, pos, vel, 'a')
        # TODO: add a way to append particles to the existing nbody.h5 file
        save_cfg(outdir, cfg)
    finally:
        # Always clear the transient work directory (node-local /tmp is purged
        # at job end, but clean up explicitly for the wdir-fallback case too)
        # nbody.harvest=True leaves snapshots in the shared scratch for an
        # external harvester job to convert and clean up
        if (workdir != outdir and os.path.isdir(workdir)
                and not cfg.nbody.get('harvest', False)):
            shutil.rmtree(workdir, ignore_errors=True)
        if (cfg.meta.get('localdir', None) is not None
                and os.path.isdir(cfg.meta.localdir)):
            shutil.rmtree(cfg.meta.localdir, ignore_errors=True)

    # clean up slurm hostfile if it exists
    if os.path.isfile('slurm_hostfile'):
        os.remove('slurm_hostfile')

    logging.info("Done!")


if __name__ == '__main__':
    main()
