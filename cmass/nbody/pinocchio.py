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
    parse_nbody_config, gen_white_noise, load_white_noise,
    save_nbody, rho_and_vfield, generate_pk_file)
from ..bias.rho_to_halo import save_snapshot


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
        node_list = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
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
    output_redshifts = [cfg.nbody.zf]
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
    memory_exec = join(os.path.dirname(cfg.nbody.pinocchio_exec), 'memorytest')
    command = f'mpirun -np 1 {memory_exec} {filename} {max_cores} > {log_file}'
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    cwd = os.getcwd()
    os.chdir(outdir)
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
        logging.info(f"Required memory per task (in Mb): {required_memory_per_task}")
    else:
        raise ValueError(f"Required memory per task not found in {log_file}")
    if total_required_memory_match:
        total_required_memory = float(total_required_memory_match.group(1))
        logging.info(f"Total required memory (in Gb): {total_required_memory}")
    else:
        raise ValueError(f"Total required memory not found in {log_file}")

    # Change memory requirements if not acceptable (less than 10% buffer)
    if (bytes_per_particle > 0.9 * MaxMemPerParticle) or (required_memory_per_task > 0.9 * MaxMem):
        logging.info('Maximum memory requirements too stringent. Updating file.')
        MaxMem = max(required_memory_per_task/0.9, MaxMem)
        MaxMemPerParticle = max(bytes_per_particle/0.9, MaxMemPerParticle)
        content = make_content(MaxMem, MaxMemPerParticle)
        with open(filename, 'w') as file:
            file.write(content)

    # Check job has the required resources if SLURM job
    if 'SLURM_JOB_NODELIST' in os.environ:
        
        slurm_job_id = os.getenv("SLURM_JOBID")
        if slurm_job_id is None:
            raise ValueError("Error: SLURM_JOBID environment variable is not set.")

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
        total_memory = int(tres_dict.get("mem", "0M").rstrip("M")) / 1000  # Total memory in GB
        total_cpus = int(tres_dict.get("cpu", "1"))  # Total CPUs allocated
        memory_per_core = total_memory * 1000 / total_cpus if total_cpus > 0 else 0

        # Print results
        logging.info(f"Allocated total memory (in Gb): {total_memory}")
        logging.info(f"Allocated memory per task (in Mb): {memory_per_core:.2f}")

        # Check we have the resources
        if total_memory < total_required_memory:
            raise ValueError('Insufficient total memory for job to run')
        if memory_per_core < required_memory_per_task:
            raise ValueError('Insufficient memory per core for job to run')

    return


@timing_decorator
def run_density(cfg, outdir):

    # Run from output dir
    cwd = os.getcwd()
    os.chdir(outdir)

    max_cores, mpi_args = get_mpi_info()

    # Run pinoccio
    command = f'mpirun -n {max_cores} {mpi_args} '
    command += f'{cfg.nbody.pinocchio_exec} parameter_file'
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    _ = subprocess.run(command, shell=True, check=True, env=env)
    os.chdir(cwd)

    # Load the data
    filename = join(
        outdir,
        f'pinocchio.{cfg.nbody.zf:.4f}.pinocchio-L{cfg.nbody.L}-'
        f'N{cfg.nbody.N}-{cfg.nbody.lhid}.snapshot.out')
    data = {}

    with open(filename, 'rb') as f:

        # Function to read a block
        def read_block(expected_name, dtype, count):
            initial_block_size = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            block_name = np.fromfile(f, dtype='S4', count=1)[
                0].decode().strip()
            block_size_with_name = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            if block_name != expected_name:
                raise ValueError(
                    f"Expected block name '{expected_name}', "
                    f"but got '{block_name}'")
            data_block = np.fromfile(f, dtype=dtype, count=count)
            trailing_block_size = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            return data_block


        # Function to read a large block
        def read_large_block(expected_name, dummy_dtype, data_dtype, num_particles, chunk_size = 900000):

            # Save to file to prevent storing too much in memory
            out_filename = join(
                outdir, f'pinocchio.{cfg.nbody.zf:.4f}.pinocchio-L{cfg.nbody.L}-'
                f'N{cfg.nbody.N}-{cfg.nbody.lhid}.particle_{expected_name}.h5')

            def write_initial_chunk(data_chunk):
                with h5py.File(out_filename, 'w') as fout:
                    if expected_name == 'VEL':
                        for i, suffix in enumerate(['X', 'Y', 'Z']):
                            dset = fout.create_dataset(
                                expected_name + suffix,
                                data=data_chunk[:,i],
                                maxshape=(None,),
                                chunks=True)
                    else:
                        dset = fout.create_dataset(
                            expected_name, 
                            data=data_chunk, 
                            maxshape=(None,)+data_chunk.shape[1:], 
                            chunks=True)

            def append_chunk(new_chunk):
                with h5py.File(out_filename, 'a') as fout:
                    if expected_name == 'VEL':
                        for i, suffix in enumerate(['X', 'Y', 'Z']):
                            dset = fout[expected_name+suffix]
                            dset.resize(dset.shape[0] + new_chunk.shape[0], axis=0)
                            dset[-new_chunk.shape[0]:] = new_chunk[:,0]
                    else:
                        dset = fout[expected_name]
                        dset.resize(dset.shape[0] + new_chunk.shape[0], axis=0)
                        dset[-new_chunk.shape[0]:] = new_chunk

            initial_block_size = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            block_name = np.fromfile(f, dtype='S4', count=1)[
                0].decode().strip()
            block_size_with_name = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used
            if block_name != expected_name:
                raise ValueError(
                    f"Expected block name '{expected_name}', "
                    f"but got '{block_name}'")

            _ = np.fromfile(f, dtype=dummy_dtype, count=1)[0]

            assert chunk_size % 3 == 0, f"{chunk_size} is not divisible by 3"
            total_read = 0

            while total_read < num_particles:
                count = min(chunk_size, num_particles - total_read)
                data = np.fromfile(f, dtype=data_dtype, count=count)
                
                if expected_name in ['POS', 'VEL']:
                    data = data.reshape(-1,3)
                    data[:, [0, 1, 2]] = data[:, [2, 1, 0]]

                # Periodic boundary conditions
                if expected_name == 'POS':
                    data = np.mod(data, cfg.nbody.L)

                if total_read == 0:
                    write_initial_chunk(data)
                else:
                    append_chunk(data)
                total_read += count

            trailing_block_size = np.fromfile(
                f, dtype=np.int32, count=1)[0]  # not used

            return out_filename


        # Read the HEADER block
        header_dtype = np.dtype([
            ('dummy', np.int64),
            ('NPart', np.uint32, 6),
            ('Mass', np.float64, 6),
            ('Time', np.float64),
            ('RedShift', np.float64),
            ('flag_sfr', np.int32),
            ('flag_feedback', np.int32),
            ('NPartTotal', np.uint32, 6),
            ('flag_cooling', np.int32),
            ('num_files', np.int32),
            ('BoxSize', np.float64),
            ('Omega0', np.float64),
            ('OmegaLambda', np.float64),
            ('HubbleParam', np.float64),
            ('flag_stellarage', np.int32),
            ('flag_metals', np.int32),
            ('npartTotalHighWord', np.uint32, 6),
            ('flag_entropy_instead_u', np.int32),
            ('flag_metalcooling', np.int32),
            ('flag_stellarevolution', np.int32),
            ('fill', np.int8, 52)
        ])
        header_block = read_block('HEAD', header_dtype, 1)[0]
        data['header'] = {name: header_block[name]
                          for name in header_dtype.names}

        # Number of particles
        num_particles = header_block['NPart'][1]

        # Read INFO block
        info_dtype = np.dtype(
            [('name', 'S4'), ('type', 'S8'), ('ndim', np.int32),
             ('active', np.int32, 6)])
        read_block('INFO', info_dtype, 4)

        # Read empty FMAX block
        fmax_dtype = np.dtype([('dummy', np.int64), ('fmax', np.float32)])
        read_block('FMAX', fmax_dtype, 2)

        # Read empty RMAX block
        rmax_dtype = np.dtype([('dummy', np.int64), ('rmax', np.int64)])
        read_block('RMAX', rmax_dtype, 2)

        # Read ID block in chunks (otherwise can be too much memory)
        logging.info("Reading particle IDs...")
        ids_fname = read_large_block('ID', np.int64, np.uint32, num_particles,)

        # Read POS block and save to file to reduce memory allocated
        logging.info("Reading particle positions...")
        pos_fname = read_large_block('POS', np.int64, np.float32, num_particles*3)

        # Read VEL block and save to file to reduce memory allocated
        logging.info("Reading particle velocities...")
        vel_fname  = read_large_block('VEL', np.int64, np.float32, num_particles*3)

    # Load halo data
    #    0) group ID
    #    1) group mass (Msun/h)
    # 2- 4) initial position (Mpc/h)
    # 5- 7) final position (Mpc/h)
    # 8-10) velocity (km/s)
    #   11) number of particles
    filename = join(
        outdir, f'pinocchio.{cfg.nbody.zf:.4f}.pinocchio-L{cfg.nbody.L}-'
                f'N{cfg.nbody.N}-{cfg.nbody.lhid}.catalog.out')
    hmass = np.log10(np.loadtxt(filename, unpack=False, usecols=(1,)))
    hpos = np.loadtxt(filename, unpack=False, usecols=(7, 6, 5))
    hvel = np.loadtxt(filename, unpack=False, usecols=(10, 9, 8))

    return pos_fname, vel_fname, hpos, hvel, hmass


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
    pos_fname, vel_fname, hpos, hvel, hmass = run_density(cfg, outdir)

    # Acces pos and vel with memory mapping
    posfile = h5py.File(pos_fname, 'r')
    velfile = h5py.File(vel_fname, 'r')
    pos = posfile['POS']
    vel = [velfile['VELX'], velfile['VELY'], velfile['VELZ']]

    # Calculate velocity field
    if hasattr(cfg.nbody, 'supersampling'):
        Ngrid = cfg.nbody.N // cfg.nbody.supersampling
    else:
        Ngrid = cfg.nbody.N
    logging.info(f'Running field construction on grid {Ngrid}...')
    rho, fvel = rho_and_vfield(
        pos, vel, cfg.nbody.L, Ngrid, 'CIC',
        omega_m=cfg.nbody.cosmo[0], h=cfg.nbody.cosmo[2],
        chunk_size = 512**3)

    if not cfg.nbody.save_particles:
        pos, vel = None, None

    # Convert from comoving -> physical velocities
    fvel *= (1 + cfg.nbody.zf)

    # Save nbody-type outputs
    save_nbody(outdir, cfg.nbody.af, rho, fvel, pos, vel)
    save_cfg(outdir, cfg)

    posfile.close()
    velfile.close()

    # Delete temporary position and velocity files
    for fname in [pos_fname, vel_fname]:
        os.remove(fname)

    # Save bias-type outputs
    outpath = join(outdir, 'halos.h5')
    logging.info('Saving cube to ' + outpath)
    if os.path.isfile(outpath):
        os.remove(outpath)
    save_snapshot(outdir, cfg.nbody.af, hpos, hvel, hmass)

    logging.info("Done!")


if __name__ == '__main__':
    main()
