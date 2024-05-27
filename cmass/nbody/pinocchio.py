import os
from os.path import join as pjoin
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from ..utils import get_source_path, timing_decorator, load_params

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


def generate_param_file(cfg, outdir):
    
    random_seed=486604
    
    filename = pjoin(outdir, "parameter_file")
    output_list= pjoin(outdir, "outputs")
    run_flag = f"pinocchio-L{int(cfg.nbody.L)}-N{cfg.nbody.N}-{cfg.nbody.lhid}"
    omega0, omega_b, h, n_s, sigma8 = cfg.nbody.cosmo
    omega_lambda = 1. - omega0
    
    # Cosmological constant in current setup
    de_w0 = -1.0
    de_wa = 0.0
    
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
    
    content = f"""# This is an example parameter file for the Pinocchio 4.0 code

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
FileWithInputSpectrum  no           % P(k) tabulated in a file
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
% WriteSnapshot                     % writes a Gadget2 snapshot as an output
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
    generate_param_file(cfg, outdir)
    
#     # Setup
#     cpar = build_cosmology(*cfg.nbody.cosmo)

#     # Get ICs
#     wn = get_ICs(cfg)

#     # Run
#     rho, pos, vel = run_density(wn, cpar, cfg)

#     # Calculate velocity field
#     fvel = None
#     if cfg.nbody.save_velocities:
#         fvel = vfield_CIC(pos, vel, cfg)
#         # convert from comoving -> peculiar velocities
#         fvel *= (1 + cfg.nbody.zf)

#     # Save
#     outdir = get_source_path(cfg, f"borg{cfg.nbody.order}lpt", check=False)
#     save_nbody(outdir, rho, fvel, pos, vel,
#                cfg.nbody.save_particles, cfg.nbody.save_velocities)
#     with open(pjoin(outdir, 'config.yaml'), 'w') as f:
#         OmegaConf.save(cfg, f)
    logging.info("Done!")


if __name__ == '__main__':
    main()
    
    
# python -m cmass.nbody.pinocchio nbody=quijote_z0

