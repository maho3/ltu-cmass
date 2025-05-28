"""
Simulate density field and halo catalogs using PINOCCHIO.

Input:
    - wn: initial white noise field

Output:
    - nbody.h5
        - rho: density contrast field
        - fvel: velocity field
        - pos: particle positions [optional]
        - vel: particle velocities [optional]
    - halos.h5
        - pos: halo positions
        - vel: halo velocities
        - mass: halo masses
"""

import os
from os.path import join
import subprocess
import numpy as np
import logging
import hydra
import re
import h5py
from omegaconf import DictConfig, OmegaConf
from ..utils import get_source_path, timing_decorator, save_cfg
from .tools import (
    parse_nbody_config, gen_white_noise, load_white_noise, generate_pk_file)
from .tools_pinocchio import (
    process_snapshot, save_pinocchio_nbody, process_halos, save_cfg_data)


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
        # ic = gen_white_noise(N, seed=nbody.lhid)
        seed = np.random.randint(0, 2**32)
        logging.info(f'Using seed {seed} for simulation {nbody.lhid}')
        nthread, _ = get_mpi_info()
        basedir = os.getcwd()
        path_to_ic = join(cfg.meta.wdir, f'wn/N{N}/')
        os.makedirs(path_to_ic, exist_ok=True)
        path_to_ic = join(path_to_ic, f'wn_{seed}.dat')
        command = f'mpirun -n 1 {basedir}/quijote_wn/NGenicWhiteNoise/ngenic_white_noise {nbody.N} {nbody.N} {seed} {path_to_ic} {nthread}'
        logging.info(command)
        os.system(command)
        ic = load_white_noise(path_to_ic, N, quijote=True)
        if os.path.exists(path_to_ic):
            os.remove(path_to_ic)  # remove the file

    # Convert to real space
    ic = np.fft.irfftn(ic, norm="ortho").astype(np.float32)

    #  Make header
    header = np.array([0, N, N, N, nbody.lhid, 0], dtype=np.int32)

    # Write the white noise field to a binary file
    filename = join(outdir, "WhiteNoise")
    with open(filename, 'wb') as file:
        header.tofile(file)
        # ic.tofile(file)
        planesize = N * N * np.dtype(np.float32).itemsize
        for plane in ic:
            file.write(np.array([planesize], dtype=np.int32).tobytes())
            plane.tofile(file)
            file.write(np.array([planesize], dtype=np.int32).tobytes())

    return


def get_mpi_info():

    if 'PBS_NODEFILE' in os.environ:  # assume PBS job
        with open(os.environ['PBS_NODEFILE'], 'r') as f:
            max_cores = len(f.readlines())
        mpi_args = '--hostfile $PBS_NODEFILE'
    elif 'SLURM_JOB_NODELIST' in os.environ:  # assume Slurm job
        # Use scontrol to get the expanded list of nodes
        node_list = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        node_list = node_list.decode('utf-8').splitlines()

        # Calculate max_cores (total cores allocated)
        cores_per_node = int(os.environ.get('SLURM_CPUS_ON_NODE', 1))
        max_cores = len(node_list) * cores_per_node

        # Write to a hostfile (optional, if required by your MPI setup)
        hostfile = 'slurm_hostfile'
        with open(hostfile, 'w') as f:
            for node in node_list:
                f.write(f"{node}\n")

        # Set MPI args
        mpi_args = f'--hostfile {hostfile}'
    else:
        max_cores = os.cpu_count()
        mpi_args = ''

    return max_cores, mpi_args


def generate_param_file(cfg, outdir):

    random_seed = cfg.nbody.lhid

    # Convert args to variables needed
    filename = join(outdir, "parameter_file")
    output_list = join(outdir, "outputs")
    run_flag = f"pinocchio-L{int(cfg.nbody.L)}-N{cfg.nbody.N}-{cfg.nbody.lhid}"
    omega0, omega_b, h, n_s, sigma8 = cfg.nbody.cosmo
    omega_lambda = 1. - omega0

    # Cosmological constant in current setup
    de_w0 = -1.0
    de_wa = 0.0

    # Choose output mass function
    mass_functions = [
        "Press_Schechter_1974",
        "Sheth_Tormen_2001",
        "Jenkins_2001",
        "Warren_2006",
        "Reed_2007",
        "Crocce_2010",
        "Tinker_2008",
        "Courtin_2010",
        "Angulo_2012",
        "Watson_2013",
        "Crocce_2010_universal"
    ]
    if cfg.nbody.mass_function in mass_functions:
        AMF = mass_functions.index(cfg.nbody.mass_function)
    else:
        AMF = len(mass_functions) - 1
        logging.info(f'Choosing analytic mass function {mass_functions[AMF]}')

    # Make outputs file
    content = (
        "# This file contains the list of output redshifts, in chronological\n"
        "# (i.e. descending) order. The last value is the final redshift of the\n"
        "# run. The past-light cone is NOT generated using these outputs but\n"
        "# is computed with continuous time sampling.\n\n"
    )
    output_redshifts = list(1 / np.array(cfg.nbody.asave) - 1)
    content += "\n".join(map(str, sorted(output_redshifts,
                         reverse=True))) + "\n"
    with open(output_list, 'w') as file:
        file.write(content)

    if cfg.nbody.transfer == 'EH':
        pk_file = "no"
    else:
        pk_file = "input_power_spectrum.txt"

    # Make paramater file

    def make_content(MaxMem, MaxMemPerParticle):

        content = f"""# This is a parameter file for the Pinocchio 4.0 code

# run properties
RunFlag                {run_flag}      % name of the run
OutputList             {output_list}      % name of file with required output redshifts
BoxSize                {cfg.nbody.L}        % physical size of the box in Mpc
BoxInH100                           % specify that the box is in Mpc/h
GridSize               {cfg.nbody.N} % number of grid points per side
RandomSeed             {random_seed}       % random seed for initial conditions

# cosmology
Omega0                 {omega0}         % Omega_0 (total matter)
OmegaLambda            {omega_lambda}         % Omega_Lambda
OmegaBaryon            {omega_b}        % Omega_b (baryonic matter)
Hubble100              {h}         % little h
Sigma8                 {sigma8}          % sigma8; if 0, it is computed from the provided P(k)
PrimordialIndex        {n_s}         % n_s
DEw0                   {de_w0}         % w0 of parametric dark energy equation of state
DEwa                   {de_wa}          % wa of parametric dark energy equation of state
TabulatedEoSfile       no           % equation of state of dark energy tabulated in a file
FileWithInputSpectrum  {pk_file}           % P(k) tabulated in a file
                                    % "no" means that the fit of Eisenstein & Hu is used

# from N-GenIC
InputSpectrum_UnitLength_in_cm 0    % units of tabulated P(k), or 0 if it is in h/Mpc
WDM_PartMass_in_kev    0.0          % WDM cut following Bode, Ostriker & Turok (2001)

# control of memory requirements
BoundaryLayerFactor    1.0          % width of the boundary layer for fragmentation
MaxMem                 {MaxMem}     % max available memory to an MPI task in Mbyte
MaxMemPerParticle      {MaxMemPerParticle} % max available memory in bytes per particle

# output
CatalogInAscii                      % catalogs are written in ascii and not in binary format
OutputInH100                        % units are in H=100 instead of the true H value
NumFiles               1            % number of files in which each catalog is written
MinHaloMass            10           % smallest halo that is given in output
AnalyticMassFunction   {AMF}            % form of analytic mass function given in the .mf.out files

# output options:
WriteSnapshot                     % writes a Gadget2 snapshot as an output
% DoNotWriteCatalogs                % skips the writing of full catalogs (including PLC)
% DoNotWriteHistories               % skips the writing of merger histories

# for debugging or development purposes:
% WriteFmax                         % writes the values of the Fmax field, particle by particle
% WriteVmax                         % writes the values of the Vmax field, particle by particle
% WriteRmax                         % writes the values of the Rmax field, particle by particle
% WriteDensity                      % writes the linear density, particle by particle

# past light cone
StartingzForPLC        0.3          % starting (highest) redshift for the past light cone
LastzForPLC            0.0          % final (lowest) redshift for the past light cone
PLCAperture            30           % cone aperture for the past light cone
% PLCProvideConeData                % read vertex and direction of cone from paramter file
% PLCCenter 0. 0. 0.                % cone vertex in the same coordinates as the BoxSize
% PLCAxis   1. 1. 0.                % un-normalized direction of the cone axis
"""
        return content

    # Initial guess at memory requirements
    MaxMem = 3600
    MaxMemPerParticle = 300
    content = make_content(MaxMem, MaxMemPerParticle)

    # Write the content to a file
    with open(filename, 'w') as file:
        file.write(content)

    max_cores, mpi_args = get_mpi_info()

    # Run memory checking
    log_file = join(outdir, 'memory_log')
    memory_exec = join(os.path.dirname(cfg.meta.pinocchio_exec), 'memorytest')
    command = f'mpirun -np 1 {memory_exec} {filename} {max_cores} > {log_file}'
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    cwd = os.getcwd()
    os.chdir(outdir)
    logging.info(command)
    _ = subprocess.run(command, shell=True, check=False, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.chdir(cwd)

    # Check memory requirements within limits
    with open(log_file, 'r') as f:
        text = f.read()
    bytes_per_particle_pattern = r"bytes per particle:\s+([0-9]*\.?[0-9]+)"
    required_memory_pattern = r"Required memory per task:\s+([0-9]*\.?[0-9]+)Mb"
    total_required_memory_pattern = r"Total required memory:\s+([0-9]*\.?[0-9]+)Gb"
    bytes_per_particle_match = re.search(bytes_per_particle_pattern, text)
    required_memory_match = re.search(required_memory_pattern, text)
    total_required_memory_match = re.search(total_required_memory_pattern, text)
    if bytes_per_particle_match:
        bytes_per_particle = float(bytes_per_particle_match.group(1))
        logging.info(f"Bytes per particle: {bytes_per_particle}")
    else:
        raise ValueError(f"Bytes per particle not found in {log_file}")
    if required_memory_match:
        required_memory_per_task = float(required_memory_match.group(1))
        logging.info(
            f"Required memory per task (in Mb): {required_memory_per_task}")
    else:
        raise ValueError(f"Required memory per task not found in {log_file}")
    if total_required_memory_match:
        total_required_memory = float(total_required_memory_match.group(1))
        logging.info(f"Total required memory (in Gb): {total_required_memory}")
    else:
        raise ValueError(f"Total required memory not found in {log_file}")

    # Change memory requirements if not acceptable (less than 10% buffer)
    if (bytes_per_particle > 0.9 * MaxMemPerParticle) or (required_memory_per_task > 0.9 * MaxMem):
        logging.info(
            'Maximum memory requirements too stringent. Updating file.')
        MaxMem = max(required_memory_per_task/0.9, MaxMem)
        MaxMemPerParticle = max(bytes_per_particle/0.9, MaxMemPerParticle)
        content = make_content(MaxMem, MaxMemPerParticle)
        with open(filename, 'w') as file:
            file.write(content)

    # Check job has the required resources if SLURM job
    if 'SLURM_JOB_NODELIST' in os.environ:

        slurm_job_id = os.getenv("SLURM_JOBID")
        if slurm_job_id is None:
            raise ValueError(
                "Error: SLURM_JOBID environment variable is not set.")

        result = subprocess.run(
            ["scontrol", "show", "job", slurm_job_id],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        match = re.search(r"AllocTRES=([^ ]+)", output)
        if not match:
            raise ValueError("TRES information not found in SLURM job details.")
        # Extract the TRES details
        tres_info = match.group(1)

        # Parse the TRES details
        tres_dict = {}
        for resource in tres_info.split(","):
            key, value = resource.split("=")
            tres_dict[key] = value

        # Compute total memory and memory per core
        mem = tres_dict.get("mem", "0M")
        if mem[-1] == "M":
            total_memory = float(mem.rstrip("M")) / 1000
        elif mem[-1] == "G":
            total_memory = float(mem.rstrip("G"))  # Total memory in GB
        total_cpus = int(tres_dict.get("cpu", "1"))  # Total CPUs allocated
        memory_per_core = total_memory * 1000 / total_cpus if total_cpus > 0 else 0

        # Print results
        logging.info(f"Allocated total memory (in Gb): {total_memory}")
        logging.info(
            f"Allocated memory per task (in Mb): {memory_per_core:.2f}")

        # Check we have the resources
        if total_memory < total_required_memory:
            raise ValueError('Insufficient total memory for job to run')
        if memory_per_core < required_memory_per_task:
            raise ValueError('Insufficient memory per core for job to run')

    return


def delete_files(cfg, outdir):

    all_files = os.listdir(outdir)

    # Check files we need exist
    if ('nbody.h5' not in all_files) or ('halos.h5' not in all_files):
        logging.info(
            'Not deleting Pinocchio files as nbody.h5 and/or halos.h5 does not exist')
        return

    # Check we have saved the relevant scale factors
    for fname in ['nbody.h5', 'halos.h5']:
        with h5py.File(join(outdir, fname)) as f:
            keys = list(f.keys())
            asave = [f'{a:.6f}' for a in cfg.nbody.asave]
            keys.sort()
            asave.sort()
            if not (asave == keys):
                logging.info(
                    f'Not deleting Pinocchio files as {fname} does not have all redshifts saved')
                return

    files_to_keep = ['input_power_spectrum.txt', 'halos.h5',
                     'nbody.h5', 'parameter_file', 'config.yaml']

    for f in all_files:
        if f not in files_to_keep:
            os.remove(join(outdir, f))

    return


@timing_decorator
def run_density(cfg, outdir):

    # Run from output dir
    cwd = os.getcwd()
    os.chdir(outdir)

    max_cores, mpi_args = get_mpi_info()

    # Run pinoccio
    command = f'mpirun -n {max_cores} {mpi_args} '
    command += f'{cfg.meta.pinocchio_exec} parameter_file'
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    _ = subprocess.run(command, shell=True, check=True, env=env)

    # Process snapshots
    for f in ['nbody.h5', 'halos.h5']:
        if os.path.isfile(join(outdir, f)):
            os.remove(join(outdir, f))

    if len(cfg.nbody.asave) == 1:
        logging.info('Processing output on a single core')
        if hasattr(cfg.nbody, 'supersampling'):
            supersampling = cfg.nbody.supersampling
        else:
            supersampling = None
        z = 1 / cfg.nbody.asave[0] - 1
        rho, fvel, pos_fname, vel_fname = process_snapshot(
            outdir, z, cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid,
            cfg.nbody.cosmo[0], cfg.nbody.cosmo[2],
            supersampling=supersampling)
        save_pinocchio_nbody(outdir, rho, fvel, pos_fname, vel_fname, z,
                             save_particles=cfg.nbody.save_particles)
        process_halos(outdir, cfg.nbody.zf, cfg.nbody.L,
                      cfg.nbody.N, cfg.nbody.lhid)
    else:
        ncore = min(max_cores, len(cfg.nbody.asave))
        save_cfg_data(outdir, cfg)
        command = f'mpirun -n {ncore} {mpi_args} env PYTHONPATH={cwd} '
        command += 'python -u -m cmass.nbody.tools_pinocchio '
        command += join(outdir, 'snapshot_data.yaml')
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        logging.info(f'Launching output processing on {ncore} cores')
        _ = subprocess.run(command, shell=True, check=True, env=env)

    os.chdir(cwd)

    return


@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
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

    outdir = get_source_path(
        cfg.meta.wdir, cfg.nbody.suite, "pinocchio",
        cfg.nbody.L, cfg.nbody.N, cfg.nbody.lhid, check=False
    )
    os.makedirs(outdir, exist_ok=True)

    # Setup power spectrum file if needed
    if cfg.nbody.transfer != 'EH':
        generate_pk_file(cfg, outdir)

    # Convert ICs to correct format
    get_ICs(cfg, outdir)

    # Generate parameter file
    generate_param_file(cfg, outdir)

    # Run
    run_density(cfg, outdir)
    save_cfg(outdir, cfg)

    delete_files(cfg, outdir)

    logging.info("Done!")


if __name__ == '__main__':
    main()
