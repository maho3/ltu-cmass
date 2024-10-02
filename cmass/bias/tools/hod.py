
import numpy as np

from halotools.empirical_models import NFWProfile
from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.empirical_models import halo_mass_to_halo_radius
from halotools.empirical_models import (
    Zheng07Cens, Zheng07Sats,
    Leauthaud11Cens, Leauthaud11Sats,
    ZuMandelbaum15Cens, ZuMandelbaum15Sats,
    NFWPhaseSpace, TrivialPhaseSpace,
)
from halotools.empirical_models import HodModelFactory


def thetahod_literature(paper, model='zheng07'):
    ''' best-fit HOD parameters from the literature.

    Currently, HOD values from the following papers are available:
    * 'parejko2013_lowz'
    * 'manera2015_lowz_ngc'
    * 'manera2015_lowz_sgc'
    * 'redi2014_cmass'

    Args:
        model (str)
            The HOD model for which we wish to obtain parameters
        paper (str)
            The paper of best-fit parameters
    '''
    if model == 'zheng07':
        if paper == 'parejko2013_lowz':
            # lowz catalog from Parejko+2013 Table 3. Note that the
            # parameterization is slightly different so the numbers need to
            # be converted.
            p_hod = {
                'logMmin': 13.25,
                'sigma_logM': 0.43,  # 0.7 * sqrt(2) * log10(e)
                'logM0': 13.27,  # log10(kappa * Mmin)
                'logM1': 14.18,
                'alpha': 0.94
            }
        elif paper == 'manera2015_lowz_ngc':
            # best-fit HOD of the lowz catalog NGC from Table 2 of Manera et al.(2015)
            p_hod = {
                'logMmin': 13.20,
                'sigma_logM': 0.62,
                'logM0': 13.24,
                'logM1': 14.32,
                'alpha': 0.9
            }
        elif paper == 'manera2015_lowz_sgc':
            # best-fit HOD of the lowz catalog SGC from Table 2 of Manera et al.(2015)
            # Manera+(2015) actually uses a redshift dependent HOD. The HOD that's
            # currently implemented is primarily for the 0.2 < z < 0.35 population,
            # which has nbar~3x10^-4 h^3/Mpc^3
            p_hod = {
                'logMmin': 13.14,
                'sigma_logM': 0.55,
                'logM0': 13.43,
                'logM1': 14.58,
                'alpha': 0.93
            }
        elif paper == 'reid2014_cmass':
            # best-fit HOD from Reid et al. (2014) Table 4
            p_hod = {
                'logMmin': 13.03,
                'sigma_logM': 0.38,
                'logM0': 13.27,
                'logM1': 14.08,
                'alpha': 0.76
            }
        else:
            raise NotImplementedError
    elif model == 'leauthaud11':
        if paper == 'behroozi10':
            """
            Best-fit HOD from Behroozi+10 (arXiv:1001.0015; table 2, column 1) 
            with redshift dependence, as well as bes fit satellite occupation
            parameters from Leauthaud+12 (arXiv:1104.0928, Table 5, SIGMOD1).
            """
            p_hod = {
                'smhm_m0_0': 10.72,
                'smhm_m0_a': 0.55,
                'smhm_m1_0': 12.35,
                'smhm_m1_a': 0.28,
                'smhm_beta_0': 0.44,
                'smhm_beta_a': 0.18,
                'smhm_delta_0': 0.57,
                'smhm_delta_a': 0.17, 
                'smhm_gamma_0': 1.56, 
                'smhm_gamma_a': 2.51,
                'scatter_model_param1': 0.15,
                'alphasat': 1,
                'betasat': 0.859,
                'bsat': 10.62,
                'betacut': -0.13,
                'bcut': 1.47,
            }
        else:
            raise NotImplementedError
    elif model == 'zu_mandelbaum15':
        # if paper == 'zu_mandelbaum15':
        """
        Zu \& Mandelbaum+15, arXiv:1505.02781
        Table 2, iHOD
        """
        p_hod = {
            'smhm_m0': 10.31,
            'smhm_m1': 12.10,
            'smhm_beta': 0.33,
            'smhm_delta': 0.42,
            'smhm_gamma': 1.21,
            'smhm_sigma': 0.50,
            'smhm_sigma_slope': -0.04,
            'alphasat': 1.00,
            'betasat': 0.90,
            'bsat': 8.98,
            'betacut': 0.41,
            'bcut': 0.86,
        }
    else:
        raise ValueError(
            '`model` not recognised, please provide a supported HOD model:'
            ' zheng07, leauthaud11, zu_mandelbaum15' 
        )

    return p_hod


def mass_to_concentration(mass, redshift, cosmo, mdef='vir'):
    model = NFWProfile(
        cosmology=cosmo,
        conc_mass_model='dutton_maccio14',
        mdef=mdef,
        redshift=redshift
    )
    return model.conc_NFWmodel(prim_haloprop=mass)


def build_halo_catalog(
    pos, vel, mass, redshift, BoxSize, cosmo,
    radius=None, conc=None, mdef='vir'
):
    '''Build a halo catalog from the given halo properties.

    Args:
        pos (array_like): Halo positions in Mpc/h. Shape (N, 3).
        vel (array_like): Halo velocities in km/s. Shape (N, 3).
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

    # convert to Halotools format
    return UserSuppliedHaloCatalog(**kws)


def build_HOD_model(
    cosmology, redshift, hod_model='zheng07', mdef='vir', **hod_params
):
    '''Build a HOD model from the given HOD parameters.

    Args:
        cosmology (astropy.cosmology.Cosmology):
            The cosmology used for the simulation.
        redshift (float): The redshift of the halo catalog.
        hod_model (str, optional): The HOD model to use.
            Defaults to 'zheng07'.
        mdef (str, optional): Halo mass definition. Defaults to 'vir'.
        **kwargs: Additional arguments to pass to the HOD model.

    Returns:
        hod_model (HODMockFactory): A HOD model object
            that can be used with Halotools.
    '''
    # determine mass column
    mkey = 'halo_m' + mdef

    # parse HOD parameters
    if hod_model == 'zheng07':
        t_ = thetahod_literature('manera2015_lowz_ngc', hod_model)
    elif hod_model == 'leauthaud11':
        t_ = thetahod_literature('behroozi10', hod_model)
    else:
        raise ValueError('`hod_model` not recognised.')

    t_.update(hod_params)
    hod_params = t_
    
    hod_params.update(
        {'cosmology': cosmology, 'redshift': redshift, 'mdef': mdef})

    # occupation functions
    if hod_model == 'zheng07':
        cenocc = Zheng07Cens(prim_haloprop_key=mkey, **hod_params)
        satocc = Zheng07Sats(
            prim_haloprop_key=mkey,
            cenocc_model=cenocc,
            modulate_with_cenocc=True,
            **hod_params
        )
    elif hod_model == 'leauthaud11':
        cenocc = Leauthaud11Cens(prim_haloprop_key=mkey, **hod_params)
        satocc = Leauthaud11Sats(
            prim_haloprop_key=mkey,
            cenocc_model=cenocc,
            **hod_params
        )
    elif hod_model == 'zu_mandelbaum15':
        cenocc = ZuMandelbaum15Cens(prim_haloprop_key=mkey, **hod_params)
        satocc = ZuMandelbaum15Sats(
            prim_haloprop_key=mkey,
            cenocc_model=cenocc,
            **hod_params
        )                
    else:
        raise NotImplementedError
    satocc._suppress_repeated_param_warning = True

    # profile functions
    censprof = TrivialPhaseSpace(**hod_params)
    satsprof = NFWPhaseSpace(**hod_params)

    # make the model
    model = {}
    model['centrals_occupation'] = cenocc
    model['centrals_profile'] = censprof
    model['satellites_occupation'] = satocc
    model['satellites_profile'] = satsprof
    return HodModelFactory(**model)
