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
import numpy as np
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from ..utils import get_source_path, timing_decorator, save_cfg
from .tools import (
    parse_nbody_config, gen_white_noise, load_white_noise,
    save_nbody, rho_and_vfield,
    get_camb_pk, get_class_pk, get_syren_pk)
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


@timing_decorator
def generate_pk_file(cfg, outdir):

    if cfg.nbody.transfer == 'EH':
        return

    kmin = 2 * np.pi / cfg.nbody.L
    #  Larger than Nyquist
    kmax = 2 * np.sqrt(3) * np.pi * cfg.nbody.N / cfg.nbody.L
    k = np.logspace(np.log10(kmin), np.log10(kmax), 2*cfg.nbody.N)

    if cfg.nbody.transfer.upper() == 'CAMB':
        pk = get_camb_pk(k, *cfg.nbody.cosmo)
    elif cfg.nbody.transfer.upper() == 'CLASS':
        pk = get_class_pk(k, *cfg.nbody.cosmo)
    elif cfg.nbody.transfer.upper() == 'SYREN':
        pk = get_syren_pk(k, *cfg.nbody.cosmo)
    else:
        raise NotImplementedError(
            f"Unknown power spectrum method: {cfg.nbody.power_spectrum}")

    np.savetxt(join(outdir, "input_power_spectrum.txt"),
               np.transpose(np.array([k, pk])))

    return


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

    content = f"""# This is a parameter file for the Pinocchio 4.0 code

# run properties
RunFlag                {run_flag}      % name of the run
OutputList             {output_list}      % name of file with required output redshifts
BoxSize                {cfg.nbody.L}        % physical size of the box in Mpc
BoxInH100                           % specify that the box is in Mpc/h
GridSize               {cfg.nbody.N}          % number of grid points per side
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
MaxMem                 3600         % max available memory to an MPI task in Mbyte
MaxMemPerParticle      300          % max available memory in bytes per particle

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

    # Write the content to a file
    with open(filename, 'w') as file:
        file.write(content)

    return


@timing_decorator
def run_density(cfg, outdir):

    # Run pinoccio
    cwd = os.getcwd()
    os.chdir(outdir)
    os.system(f'{cfg.nbody.pinocchio_exec} parameter_file')
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

        # Read ID block
        id_dtype = np.dtype([
            ('dummy', np.int64),
            ('ids', np.uint32, num_particles)
        ])
        ids = read_block('ID', id_dtype, 1)
        data['ids'] = ids['ids'][0]

        # Read POS block
        pos_dtype = np.dtype([
            ('dummy', np.int64),
            ('pos', np.float32, num_particles * 3)
        ])
        positions = read_block('POS', pos_dtype, 1)
        data['positions'] = positions['pos'].reshape(-1, 3)

        # Read VEL block
        vel_dtype = np.dtype([
            ('dummy', np.int64),
            ('vel', np.float32, num_particles * 3)
        ])
        positions = read_block('VEL', vel_dtype, 1)
        data['velocities'] = positions['vel'].reshape(-1, 3)

    # Align axes
    data['positions'][:, [0, 1, 2]] = data['positions'][:, [2, 1, 0]]
    data['velocities'][:, [0, 1, 2]] = data['velocities'][:, [2, 1, 0]]

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

    return data['positions'], data['velocities'], hpos, hvel, hmass


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
    generate_pk_file(cfg, outdir)

    # Convert ICs to correct format
    get_ICs(cfg, outdir)

    # Generate parameter file
    generate_param_file(cfg, outdir)

    # Run
    pos, vel, hpos, hvel, hmass = run_density(cfg, outdir)

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

    # Save bias-type outputs
    logging.info('Saving cube...')
    save_snapshot(outdir, cfg.nbody.af, hpos, hvel, hmass)

    logging.info("Done!")


if __name__ == '__main__':
    main()
