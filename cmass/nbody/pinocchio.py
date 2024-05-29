import os
from os.path import join as pjoin
import numpy as np
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from ..utils import get_source_path, timing_decorator, load_params
from .tools import gen_white_noise, load_white_noise, get_camb_pk, get_class_pk, get_syren_pk

def parse_config(cfg):
    with open_dict(cfg):
        nbody = cfg.nbody
        nbody.ai = 1 / (1 + nbody.zi)  # initial scale factor
        nbody.af = 1 / (1 + nbody.zf)  # final scale factor
        nbody.quijote = nbody.matchIC == 2  # whether to match ICs to Quijote
        nbody.matchIC = nbody.matchIC > 0  # whether to match ICs to file

        # load cosmology
        nbody.cosmo = load_params(nbody.lhid, cfg.meta.cosmofile)

    if cfg.nbody.quijote:
        logging.info('Matching ICs to Quijote')
        assert cfg.nbody.L == 1000  # enforce same size of quijote

    return cfg


def get_ICs(cfg, outdir):
    
    nbody = cfg.nbody
    N = nbody.N
    
    # Load the ics in Fourier space
    if nbody.matchIC:
        path_to_ic = f'wn/N{N}/wn_{nbody.lhid}.dat'
        if nbody.quijote:
            path_to_ic = pjoin(cfg.meta.wdir, 'quijote', path_to_ic)
        else:
            path_to_ic = pjoin(cfg.meta.wdir, path_to_ic)
        ic = load_white_noise(path_to_ic, N, quijote=nbody.quijote)
    else:
        ic = gen_white_noise(N, seed=nbody.lhid)
        
    # Convert to real space
    ic = np.fft.irfftn(ic, norm="ortho").astype(np.float32)

    # Make header
    header = np.array([0, N, N, N, nbody.lhid, 0], dtype=np.int32)
    
    # Write the white noise field to a binary file
    filename = pjoin(outdir, "WhiteNoise")
    with open(filename, 'wb') as file:
        header.tofile(file)
        # ic.tofile(file)
        planesize = N * N * np.dtype(np.float32).itemsize
        for plane in ic:
            file.write(np.array([planesize], dtype=np.int32).tobytes())
            plane.tofile(file)
            file.write(np.array([planesize], dtype=np.int32).tobytes())
    
    return


def generate_pk_file(cfg, outdir):
    
    if cfg.nbody.transfer == 'EH':
        return
    
    kmin = 2 * np.pi / cfg.nbody.L
    kmax = 2 * np.sqrt(3) * np.pi * cfg.nbody.N / cfg.nbody.L  # Larger than Nyquist
    k = np.logspace(np.log10(kmin), np.log10(kmax), 2*cfg.nbody.N)
    
    if cfg.nbody.transfer == 'CAMB':
        pk = get_camb_pk(k, *cfg.nbody.cosmo)
    elif cfg.nbody.transfer == 'CLASS':
        pk = get_class_pk(k, *cfg.nbody.cosmo)
    elif cfg.nbody.transfer == 'SYREN':
        pk = get_syren_pk(k, *cfg.nbody.cosmo)
    else:
        raise NotImplementedError(f"Unknown power spectrum method: {cfg.nbody.power_spectrum}")
    
    np.savetxt(pjoin(outdir, "input_power_spectrum.txt"), np.transpose(np.array([k, pk])))
    
    return


def generate_param_file(cfg, outdir):
    
    random_seed=486604
    
    # Convert args to variables needed
    filename = pjoin(outdir, "parameter_file")
    output_list= pjoin(outdir, "outputs")
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
    content += "\n".join(map(str, sorted(cfg.nbody.output_redshifts, reverse=True))) + "\n"
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


def run_pinoccio(cfg, outdir):
    
    cwd = os.getcwd()
    os.chdir(outdir)
    os.system(f'{cfg.nbody.pinocchio_exec} parameter_file')
    os.chdir(cwd)
    
    return



@timing_decorator
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Filtering for necessary configs
    cfg = OmegaConf.masked_copy(cfg, ['meta', 'nbody'])

    # Build run config
    cfg = parse_config(cfg)
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(
        "Logging directory: " +
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info('Running with config:\n' + OmegaConf.to_yaml(cfg))
    
    outdir = get_source_path(cfg, f"pinocchio", check=False)
    
    # Setup power spectrum file if needed
    generate_pk_file(cfg, outdir)
    # quit()
    
    # Convert ICs to correct format
    get_ICs(cfg, outdir)
    
    # Generate parameter file
    generate_param_file(cfg, outdir)
    
    # Run
    run_pinoccio(cfg, outdir)
 
    logging.info("Done!")


if __name__ == '__main__':
    main()
    
    
"""
python -m cmass.nbody.pinocchio nbody=quijote_z0

TO DO:
- Check ICs are used correctly
- Load halos and save to correct format
"""

