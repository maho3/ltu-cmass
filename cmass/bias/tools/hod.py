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
    Zheng07, Leauthaud11, Zu_mandelbaum15,
    Zheng07zdep, Zheng07zinterp
)


def lookup_hod_model(model=None, assem_bias=False, vel_assem_bias=False, zpivot=None, custom_prior=None):
    if model is None:
        return Zheng07()  # for backwards compatibility
    elif model == "zheng07":
        return Zheng07(assem_bias=assem_bias,
                       vel_assem_bias=vel_assem_bias)
    elif model == 'zheng07zdep':
        return Zheng07zdep(assem_bias=assem_bias,
                           vel_assem_bias=vel_assem_bias)
    elif model == 'zheng07zinterp':
        return Zheng07zinterp(zpivot, assem_bias=assem_bias,
                              vel_assem_bias=vel_assem_bias,
                              custom_prior=custom_prior)
    elif model == 'leauthaud11':
        return Leauthaud11()
    elif model == "zu_mandelbaum15":
        return Zu_mandelbaum15()
    else:
        raise NotImplementedError(
            f'Model {model} not implemented.')


def parse_hod(cfg):
    """
    Parse HOD parameters in the config file, and set them
    in the `cfg` object. 
    TODO: IS THIS STILL NEEDED? 
    MATT: It is, but there's probably a more elegant way to code this

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

        if ((cfg.bias.hod.assem_bias or cfg.bias.hod.vel_assem_bias) and
                (not cfg.bias.hod.model.startswith("zheng07"))):
            raise NotImplementedError

        # Check model is available
        model = lookup_hod_model(
            model=cfg.bias.hod.model if hasattr(
                cfg.bias.hod, "model") else None,
            assem_bias=cfg.bias.hod.assem_bias,
            vel_assem_bias=cfg.bias.hod.vel_assem_bias,
            zpivot=cfg.bias.hod.zpivot if hasattr(
                cfg.bias.hod, "zpivot") else None,
            custom_prior=cfg.bias.hod.custom_prior if hasattr(
                cfg.bias.hod, "custom_prior") else None,
        )

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
    assem_bias=False,
    vel_assem_bias=False,
    custom_prior=None,
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
        assem_bias (bool, optional):
            Whether to include assembly bias
        vel_assem_bias (bool, optional):
            Whether to include velocity assembly bias

    Returns:
        hod_model (HODMockFactory): A HOD model object
            that can be used with Halotools.
    """

    if (assem_bias or vel_assem_bias) and (not model.startswith("zheng07")):
        raise NotImplementedError

    if model == "zheng07":
        model = Zheng07(mass_def=mdef, assem_bias=assem_bias,
                        vel_assem_bias=vel_assem_bias)
    elif model == "leauthaud11":
        model = Leauthaud11(mass_def=mdef, zf=zf)
    elif model == 'zheng07zdep':
        model = Zheng07zdep(mass_def=mdef, assem_bias=assem_bias,
                            vel_assem_bias=vel_assem_bias)
    elif model == 'zheng07zinterp':
        model = Zheng07zinterp(mass_def=mdef, zpivot=zpivot,
                               assem_bias=assem_bias,
                               vel_assem_bias=vel_assem_bias,
                               custom_prior=custom_prior)
    elif model == "zu_mandelbaum15":
        model = Zu_mandelbaum15(mass_def=mdef)
    else:
        raise NotImplementedError

    model.set_parameters(dict(theta))
    model.set_occupation()
    model.set_profiles(cosmology=cosmology, zf=zf)

    return model.get_model()


def build_halo_catalog(
    pos, vel, mass, redshift, BoxSize, cosmo,
    radius=None, conc=None, halo_redshift=None, mdef='vir'
):
    '''Build a halo catalog from the given halo properties.

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
    '''
    mkey = f'halo_m{mdef}'
    rkey = f'halo_r{mdef}'

    if radius is None:
        radius = halo_mass_to_halo_radius(mass, cosmo, redshift, mdef)
    if conc is None:
        conc = mass_to_concentration(mass, redshift, cosmo, mdef)

    # Specify arguments
    kws = {
        # halo properties
        'halo_x': pos[:, 0],
        'halo_y': pos[:, 1],
        'halo_z': pos[:, 2],
        'halo_vx': vel[:, 0],
        'halo_vy': vel[:, 1],
        'halo_vz': vel[:, 2],
        mkey: mass,
        rkey: radius,
        'halo_nfw_conc': conc,
        'halo_redshift': halo_redshift if halo_redshift is not None else redshift,
        'halo_id': np.arange(len(mass)),
        'halo_hostid': np.zeros(len(mass), dtype=int),
        'halo_upid': np.zeros(len(mass)) - 1,
        'halo_local_id': np.arange(len(mass), dtype='i8'),

        # metadata
        'cosmology': cosmo,
        'redshift': redshift,
        'particle_mass': 1,  # not used
        'Lbox': BoxSize,
        'mdef': mdef,
    }
    if mdef != 'vir':
        # these are necessary to satisfy default halo properties, but not used
        kws['halo_mvir'] = np.full(len(mass), np.nan)
        kws['halo_rvir'] = np.full(len(mass), np.nan)

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
