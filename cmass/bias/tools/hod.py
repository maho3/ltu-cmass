"""
Tools for initialising and implementing Halo Occupation
Distribution (HOD) models.

Currently implemented models:
- Zheng+07
- Leauthaud+11
- Zu & Mandelbaum+15

The models themselves are described in `hod_models.py`.
"""

import logging
import numpy as np
from omegaconf import open_dict

from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.empirical_models import halo_mass_to_halo_radius, NFWProfile

from .hod_models import (
    Zheng07, Leauthaud11, Zu_mandelbaum15
    Zheng07zdepCens, Zheng07zdepSats, 
    Zheng07zinterpCens, Zheng07zinterpSats
)

def parse_hod(cfg):
    """
    Parse HOD parameters in the config file, and set them
    in the `cfg` object

    Args:
        cfg (object)
            config object
    Returns:
        cfg (object)
            modified config object
    """
    with open_dict(cfg):
        # Check the chosen mass definition
        if cfg.sim in ["borg1lpt", "borg2lpt", "borgpm", "fastpm", "pmwd"]:
            if cfg.bias.hod.mdef != "200c":
                logging.warning(
                    f"Configuration specified a {cfg.bias.hod.mdef} mass "
                    f"definition, but the {cfg.sim} simulation is a 200c "
                    "simulation. So, changing mass definition configuration to "
                    "200c."
                )
                cfg.bias.hod.mdef = "200c"
        elif cfg.sim == "pinocchio":
            if cfg.bias.hod.mdef != "vir":
                logging.warning(
                    f"Configuration specified a {cfg.bias.hod.mdef} mass "
                    f"definition, but the {cfg.sim} simulation is a vir "
                    "simulation. So, changing mass definition configuration to "
                    "vir."
                )
                cfg.bias.hod.mdef = "vir"

        # Check model is available
        if not hasattr(cfg.bias.hod, "model"):
            model = Zheng07()  # for backwards compatibility
        elif cfg.bias.hod.model == "zheng07":
            model = Zheng07()
        elif cfg.bias.hod.model == 'zheng07zdep':
            model = Zheng07zdep()
        elif cfg.bias.hod.model == 'zheng07zinterp':
            model = Zheng07zinterp(len(cfg.bias.hod.zpivot))
        elif cfg.bias.hod.model == 'leauthaud11':
            model = Leauthaud11()
        elif cfg.bias.hod.model == "zu_mandelbaum15":
            model = Zu_mandelbaum15()
        else:
            raise NotImplementedError

        # Check if we're using default parameters
        if hasattr(cfg.bias.hod, "default_params"):
            if cfg.bias.hod.default_params is not None:
                # Assign parameters given this default
                getattr(model, cfg.bias.hod.default_params)()

        # Check if 'seed' set
        if hasattr(cfg.bias.hod, "seed"):
            if cfg.bias.hod.seed is not None:
                # If -1, set to some random value
                if cfg.bias.hod.seed == -1:
                    cfg.bias.hod.seed = np.random.randint(0, 1e5)

                # If 0, don't change default values
                if cfg.bias.hod.seed > 0:
                    # Set numpy seed
                    np.random.seed(cfg.bias.hod.seed)

                    # Sample parameters from the HOD model
                    model.sample_parameters()

        # Overwrite any previously defined parameters
        if hasattr(cfg.bias.hod, "theta"):
            for key in model.parameters:
                if hasattr(cfg.bias.hod.theta, key):
                    param = float(getattr(cfg.bias.hod.theta, key))
                    model.set_parameter(key, param)

        # Get the parameter values
        cfg.bias.hod.theta = model.get_parameters()

        # Check if any values are None
        for k, v in cfg.bias.hod.theta.items():
            if v is None:
                raise ValueError(f'Parameter {k} is None. Make sure to '
                                 'set default parameters or hod.seed>0.')

    return cfg


def build_HOD_model(
    cosmology,
    model,
    theta,
    zf,
    mdef="vir",
    zpivot=None,
):
    """Build a HOD model from the given HOD parameters.

    Args:
        cosmology (astropy.cosmology.Cosmology):
            The cosmology used for the simulation.
        model (str): The HOD model to use. Options are:
            - 'zheng07'
            - 'leauthaud11'
            - 'zu_mandelbaum15'
        theta (dict): The HOD parameters.
        mdef (str, optional):
            Halo mass definition. Defaults to 'vir'.
        zpivot (str, optional):
            Pivot redshifts to be used if interpolating between redshifts. Defaults to None

    Returns:
        hod_model (HODMockFactory): A HOD model object
            that can be used with Halotools.
    """
    if model == "zheng07":
        model = Zheng07(mass_def=mdef)
    elif model == "leauthaud11":
        model = Leauthaud11(mass_def=mdef)
    elif model == 'zheng07zdep':
        model = Zheng07zdep(mass_def=mdef)    
    elif model == 'zheng07zdep':
        model = Zheng07zinterp(mass_def=mdef, zpivot=zpivot)   
    elif model == "zu_mandelbaum15":
        model = Zu_mandelbaum15(mass_def=mdef)
    else:
        raise NotImplementedError

    model.set_parameters(dict(theta))
    model.set_occupation()
    model.set_profiles(cosmology=cosmology, zf=zf)

    return model
 
#     # determine mass column
#     mkey = 'halo_m' + mdef

#     # Get HOD parameters
#     hod_params = dict(theta)

#     # Occupation functions
#     if model == 'zheng07':
#         cenocc = Zheng07Cens(prim_haloprop_key=mkey)
#         satocc = Zheng07Sats(
#             prim_haloprop_key=mkey,
#             cenocc_model=cenocc,
#             modulate_with_cenocc=True
#         )
#     elif model == 'zheng07zdep':
#         cenocc = Zheng07zdepCens(prim_haloprop_key=mkey)
#         satocc = Zheng07zdepSats(
#             prim_haloprop_key=mkey,
#             cenocc_model=cenocc,
#             modulate_with_cenocc=True
#         )
#     elif model == 'zheng07zinterp':
#         cenocc = Zheng07zinterpCens(zpivot, prim_haloprop_key=mkey)
#         satocc = Zheng07zinterpSats(
#             zpivot,
#             prim_haloprop_key=mkey,
#             cenocc_model=cenocc,
#             modulate_with_cenocc=True
#         )
#     elif model == 'leauthaud11':
#         cenocc = Leauthaud11Cens(prim_haloprop_key=mkey, redshift=zf)
#         satocc = Leauthaud11Sats(
#             prim_haloprop_key=mkey,
#             cenocc_model=cenocc, redshift=zf,
#         )
#     elif model == 'zu_mandelbaum15':
#         cenocc = ZuMandelbaum15Cens(prim_haloprop_key=mkey, redshift=zf)
#         satocc = ZuMandelbaum15Sats(prim_haloprop_key=mkey)
#         satocc.central_occupation_model = cenocc  # need to set this manually
#         # m0 and m1 are desired in real units
#         hod_params['smhm_m0'] = 10**hod_params['smhm_m0']
#         hod_params['smhm_m1'] = 10**hod_params['smhm_m1']
#     else:
#         raise NotImplementedError

#     # Set HOD parameters
#     cenocc.param_dict.update(hod_params)
#     satocc.param_dict.update(hod_params)
#     satocc._suppress_repeated_param_warning = True

#     # profile functions
#     censprof = TrivialPhaseSpace(
#         cosmology=cosmology,
#         redshift=zf,
#         mdef=mdef
#     )
#     satsprof = NFWPhaseSpace(
#         conc_mass_model='direct_from_halo_catalog',
#         cosmology=cosmology,
#         redshift=zf,
#         mdef=mdef
#     )

#     # make the model
#     model = dict(
#         centrals_occupation=cenocc,
#         centrals_profile=censprof,
#         satellites_occupation=satocc,
#         satellites_profile=satsprof
#     )
#     return HodModelFactory(**model)


def build_halo_catalog(
    pos,
    vel,
    mass,
    redshift,
    BoxSize,
    cosmo,
    radius=None,
    conc=None,
    mdef="vir",
):
    """Build a halo catalog from the given halo properties.

    Args:
        pos (array_like): Halo comoving positions in Mpc/h. Shape (N, 3).
        vel (array_like): Halo physical velocities in km/s. Shape (N, 3).
        mass (array_like): Halo masses in Msun/h. Shape (N,).
        redshift (float): The redshift of the halo catalog.
        BoxSize (float): The size of the simulation box in Mpc/h.
        cosmo (astropy.cosmology.Cosmology):
            The cosmology used for the simulation.
        radius (array_like, optional): Halo radius in Mpc/h. Defaults to None.
        conc (array_like, optional): Halo concentration parameter.
            Defaults to None.
        mdef (str, optional): Halo mass definition. Defaults to 'vir'.

    Returns:
        catalog (UserSuppliedHaloCatalog): A halo catalog object
            that can be used with Halotools.
    """
    mkey = f"halo_m{mdef}"
    rkey = f"halo_r{mdef}"

    if radius is None:
        radius = halo_mass_to_halo_radius(mass, cosmo, redshift, mdef)
    if conc is None:
        conc = mass_to_concentration(mass, redshift, cosmo, mdef)

    # Specify arguments
    kws = {
        # halo properties
        "halo_x": pos[:, 0],
        "halo_y": pos[:, 1],
        "halo_z": pos[:, 2],
        "halo_vx": vel[:, 0],
        "halo_vy": vel[:, 1],
        "halo_vz": vel[:, 2],
        mkey: mass,
        rkey: radius,
        "halo_nfw_conc": conc,
        "halo_id": np.arange(len(mass)),
        "halo_hostid": np.zeros(len(mass), dtype=int),
        "halo_upid": np.zeros(len(mass)) - 1,
        "halo_local_id": np.arange(len(mass), dtype="i8"),
        # metadata
        "cosmology": cosmo,
        "redshift": redshift,
        "particle_mass": 1,  # not used
        "Lbox": BoxSize,
        "mdef": mdef,
    }
    if mdef != "vir":
        # these are necessary to satisfy default halo properties, but not used
        kws["halo_mvir"] = np.full(len(mass), np.nan)
        kws["halo_rvir"] = np.full(len(mass), np.nan)

    # convert to Halotools format
    return UserSuppliedHaloCatalog(**kws)


def mass_to_concentration(mass, redshift, cosmo, mdef="vir"):
    model = NFWProfile(
        cosmology=cosmo,
        conc_mass_model="dutton_maccio14",
        mdef=mdef,
        redshift=redshift,
    )
    return model.conc_NFWmodel(prim_haloprop=mass)
