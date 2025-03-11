"""
Tools for initilaising and implementing Halo Occupation 
Distribution (HOD) models.

Currently implemented models:
- Zheng+07
- Leauthaud+11
- Zu & Mandelbaum+15

Each model derives from the `Hod_model` parent class,
which additionally uses the `Hod_parameter` helper class
for each parameter.
"""

import logging
import numpy as np
from omegaconf import open_dict
from .hod_models import Zheng07zdepCens, Zheng07zdepSats

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
        if cfg.sim in ['borg1lpt', 'borg2lpt', 'borgpm', 'fastpm', 'pmwd']:
            if cfg.bias.hod.mdef != '200c':
                logging.warning(
                    f'Configuration specified a {cfg.bias.hod.mdef} mass '
                    f'definition, but the {cfg.sim} simulation is a 200c '
                    'simulation. So, changing mass definition configuration to '
                    '200c.')
                cfg.bias.hod.mdef = '200c'
        elif cfg.sim == 'pinocchio':
            if cfg.bias.hod.mdef != 'vir':
                logging.warning(
                    f'Configuration specified a {cfg.bias.hod.mdef} mass '
                    f'definition, but the {cfg.sim} simulation is a vir '
                    'simulation. So, changing mass definition configuration to '
                    'vir.')
                cfg.bias.hod.mdef = 'vir'

        # Check model is available
        if not hasattr(cfg.bias.hod, 'model'):
            model = Zheng07()  # for backwards compatibility
        elif cfg.bias.hod.model == 'zheng07':
            model = Zheng07()
        elif cfg.bias.hod.model == 'zheng07zdep':
            model = Zheng07zdep()
        elif cfg.bias.hod.model == 'zheng07zinterp':
            model = None
        elif cfg.bias.hod.model == 'leauthaud11':
            model = Leauthaud11()
        elif cfg.bias.hod.model == 'zu_mandelbaum15':
            model = Zu_mandelbaum15()
        else:
            raise NotImplementedError

        # Check if we're using default parameters
        if hasattr(cfg.bias.hod, 'default_params'):
            if cfg.bias.hod.default_params is not None:
                # Assign parameters given this default
                getattr(model, cfg.bias.hod.default_params)()

        # Check if 'seed' set
        if hasattr(cfg.bias.hod, 'seed'):
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
        if hasattr(cfg.bias.hod, 'theta'):
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


class Hod_parameter:
    """
    Helper class defining a HOD parameter, including the value
    and upper and lower bounds (for a flat prior)
    """

    def __init__(self, key, value=None, upper=None, lower=None):
        self.key = key
        self.value = value
        self.upper = upper
        self.lower = lower


class Hod_model:
    """
    Parent class defining a HOD model.
    Includes methods for setting, getting and sampling parameters.
    """

    def __init__(self, parameters, lower_bound, upper_bound):
        self.parameters = parameters

        # Loop through parameters and initialise
        for _param, _lower, _upper in zip(
            self.parameters,
            lower_bound,
            upper_bound,
        ):
            setattr(
                self,
                _param,
                Hod_parameter(
                    key=_param,
                    lower=_lower,
                    upper=_upper
                )
            )

    def set_parameter(self, key, new_parameter):
        """
        Set a single parameter
        """
        getattr(self, key).value = new_parameter

    def set_parameters(self, new_parameters):
        """
        Set all parameters using a dict
        """
        for _param in self.parameters:
            getattr(self, _param).value = new_parameters[_param]

    def sample_parameters(self):
        for _param in self.parameters:
            # Get upper and lower bounds for this parameter
            _lower = getattr(self, _param).lower
            _upper = getattr(self, _param).upper

            # Sample new parameter
            sampled_param = np.random.uniform(_lower, _upper)

            # Set new parameter
            getattr(self, _param).value = sampled_param

    def get_parameters(self):
        """
        Return a dict of parameter key / value pairs
        """
        out_params = {}
        for _param in self.parameters:
            out_params[_param] = getattr(self, _param).value

        return out_params

    
class Zheng07(Hod_model):
    """
    Zheng+07 HOD model
    """

    def __init__(
        self,
        parameters=['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha'],
        lower_bound=np.array([12.0, 0.1, 13.0, 13.0, 0.]),
        upper_bound=np.array([14.0, 0.6, 15.0, 15.0, 1.5]),
        param_defaults=None
    ):
        super().__init__(parameters, lower_bound, upper_bound)

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == 'parejko2013_lowz':
                self.parejko2013_lowz()
            elif self.param_defaults == 'manera2015_lowz_ngc':
                self.manera2015_lowz_ngc()
            elif self.param_defaults == 'manera2015_lowz_sgc':
                self.manera2015_lowz_sgc()
            elif self.param_defaults == 'reid2014_cmass':
                self.reid2014_cmass()
            else:
                raise NotImplementedError

    def parejko2013_lowz(self):
        """
        lowz catalog from Parejko+2013 Table 3. Note that the
        parameterization is slightly different so the numbers need to
        be converted.
        """
        p_hod = {
            'logMmin': 13.25,
            'sigma_logM': 0.43,  # 0.7 * sqrt(2) * log10(e)
            'logM0': 13.27,  # log10(kappa * Mmin)
            'logM1': 14.18,
            'alpha': 0.94
        }
        self.set_parameters(p_hod)

    def manera2015_lowz_ngc(self):
        """
        best-fit HOD of the lowz catalog NGC from Table 2 of Manera et al.(2015)
        """
        p_hod = {
            'logMmin': 13.20,
            'sigma_logM': 0.62,
            'logM0': 13.24,
            'logM1': 14.32,
            'alpha': 0.9
        }
        self.set_parameters(p_hod)

    def manera2015_lowz_sgc(self):
        """
        best-fit HOD of the lowz catalog SGC from Table 2 of Manera et al.(2015)
        Manera+(2015) actually uses a redshift dependent HOD. The HOD that's
        currently implemented is primarily for the 0.2 < z < 0.35 population,
        which has nbar~3x10^-4 h^3/Mpc^3
        """
        p_hod = {
            'logMmin': 13.14,
            'sigma_logM': 0.55,
            'logM0': 13.43,
            'logM1': 14.58,
            'alpha': 0.93
        }
        self.set_parameters(p_hod)

    def reid2014_cmass(self):
        """
        best-fit HOD from Reid et al. (2014) Table 4
        """
        p_hod = {
            'logMmin': 13.03,
            'sigma_logM': 0.38,
            'logM0': 13.27,
            'logM1': 14.08,
            'alpha': 0.76
        }
        self.set_parameters(p_hod)
        
        
class Zheng07zdep(Hod_model):
    """
    Zheng+07 HOD model with redshift dependent mass parameters
    """

    def __init__(
        self,
        parameters=['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha', 'mucen', 'musat'],
        lower_bound=np.array([12.0, 0.1, 13.0, 13.0, 0., -30.0, -10.0]),
        upper_bound=np.array([14.0, 0.6, 15.0, 15.0, 1.5, 0., 0.]),
        param_defaults=None
    ):
        super().__init__(parameters, lower_bound, upper_bound)

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == 'parejko2013_lowz':
                self.parejko2013_lowz()
            elif self.param_defaults == 'manera2015_lowz_ngc':
                self.manera2015_lowz_ngc()
            elif self.param_defaults == 'manera2015_lowz_sgc':
                self.manera2015_lowz_sgc()
            elif self.param_defaults == 'reid2014_cmass':
                self.reid2014_cmass()
            else:
                raise NotImplementedError

    def parejko2013_lowz(self):
        """
        lowz catalog from Parejko+2013 Table 3. Note that the
        parameterization is slightly different so the numbers need to
        be converted.
        """
        p_hod = {
            'logMmin': 13.25,
            'sigma_logM': 0.43,  # 0.7 * sqrt(2) * log10(e)
            'logM0': 13.27,  # log10(kappa * Mmin)
            'logM1': 14.18,
            'alpha': 0.94,
            'mucen': 0.0,
            'musat': 0.0,
        }
        self.set_parameters(p_hod)

    def manera2015_lowz_ngc(self):
        """
        best-fit HOD of the lowz catalog NGC from Table 2 of Manera et al.(2015)
        """
        p_hod = {
            'logMmin': 13.20,
            'sigma_logM': 0.62,
            'logM0': 13.24,
            'logM1': 14.32,
            'alpha': 0.9,
            'mucen': 0.0,
            'musat': 0.0,
        }
        self.set_parameters(p_hod)

    def manera2015_lowz_sgc(self):
        """
        best-fit HOD of the lowz catalog SGC from Table 2 of Manera et al.(2015)
        Manera+(2015) actually uses a redshift dependent HOD. The HOD that's
        currently implemented is primarily for the 0.2 < z < 0.35 population,
        which has nbar~3x10^-4 h^3/Mpc^3
        """
        p_hod = {
            'logMmin': 13.14,
            'sigma_logM': 0.55,
            'logM0': 13.43,
            'logM1': 14.58,
            'alpha': 0.93,
            'mucen': 0.0,
            'musat': 0.0,
        }
        self.set_parameters(p_hod)

    def reid2014_cmass(self):
        """
        best-fit HOD from Reid et al. (2014) Table 4
        """
        p_hod = {
            'logMmin': 13.03,
            'sigma_logM': 0.38,
            'logM0': 13.27,
            'logM1': 14.08,
            'alpha': 0.76,
            'mucen': 0.0,
            'musat': 0.0,
        }
        self.set_parameters(p_hod)
        
        
class Zheng07zinterp(Hod_model):
    """
    Zheng+07 HOD model with redshift dependent mass parameters
    """

    def __init__(
        self,
        npivot,
        parameters=['logMmin', 'sigma_logM', 'logM0', 'logM1', 'alpha'],
        lower_bound=np.array([12.0, 0.1, 13.0, 13.0, 0.,]),
        upper_bound=np.array([14.0, 0.6, 15.0, 15.0, 1.5,]),
        param_defaults=None
    ):
        pars = []
        low = []
        up = []
        self.npivot = npivot
        if param_defaults is None:
            defaults = None
        else:
            defaults = []
        for i, (p, v0, v1) in enumerate(zip(parameters, lower_bound, upper_bound)):
            if p in ['logMmin', 'logM0', 'logM1']:
                for j in range(npivot):
                    pars.append(p + '_z' + str(j))
                    low.append(v0)
                    up.append(v1)
                    if param_defaults is not None:
                        defaults.append(param_defaults[i])
            else:
                pars.append(p)
                low.append(v0)
                up.append(v1)
                if param_defaults is not None:
                    defaults.append(param_defaults[i])
        super().__init__(parameters, lower_bound, upper_bound)

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == 'parejko2013_lowz':
                self.parejko2013_lowz()
            elif self.param_defaults == 'manera2015_lowz_ngc':
                self.manera2015_lowz_ngc()
            elif self.param_defaults == 'manera2015_lowz_sgc':
                self.manera2015_lowz_sgc()
            elif self.param_defaults == 'reid2014_cmass':
                self.reid2014_cmass()
            else:
                raise NotImplementedError
                
    def process_measured_hod(self, p_hod):
        new_p_hod = {}
        for k, v in p_hod.items():
            if k in ['logMmin', 'logM0', 'logM1']:
                for j in range(self.npivot):
                    new_p_hod[k + '_z' + str(j)] = v
            else:
                new_p_hod = v
        return new_p_hod
                
    def parejko2013_lowz(self):
        """
        lowz catalog from Parejko+2013 Table 3. Note that the
        parameterization is slightly different so the numbers need to
        be converted.
        """
        p_hod = {
            'logMmin': 13.25,
            'sigma_logM': 0.43,  # 0.7 * sqrt(2) * log10(e)
            'logM0': 13.27,  # log10(kappa * Mmin)
            'logM1': 14.18,
            'alpha': 0.94,
        }
        new_p_hod = self.process_measured_hod(p_hod)
        self.set_parameters(new_p_hod)

    def manera2015_lowz_ngc(self):
        """
        best-fit HOD of the lowz catalog NGC from Table 2 of Manera et al.(2015)
        """
        p_hod = {
            'logMmin': 13.20,
            'sigma_logM': 0.62,
            'logM0': 13.24,
            'logM1': 14.32,
            'alpha': 0.9,
            'mucen': 0.0,
            'musat': 0.0,
        }
        self.set_parameters(p_hod)
        new_p_hod = self.process_measured_hod(p_hod)
        self.set_parameters(new_p_hod)

    def manera2015_lowz_sgc(self):
        """
        best-fit HOD of the lowz catalog SGC from Table 2 of Manera et al.(2015)
        Manera+(2015) actually uses a redshift dependent HOD. The HOD that's
        currently implemented is primarily for the 0.2 < z < 0.35 population,
        which has nbar~3x10^-4 h^3/Mpc^3
        """
        p_hod = {
            'logMmin': 13.14,
            'sigma_logM': 0.55,
            'logM0': 13.43,
            'logM1': 14.58,
            'alpha': 0.93,
            'mucen': 0.0,
            'musat': 0.0,
        }
        self.set_parameters(p_hod)
        new_p_hod = self.process_measured_hod(p_hod)
        self.set_parameters(new_p_hod)

    def reid2014_cmass(self):
        """
        best-fit HOD from Reid et al. (2014) Table 4
        """
        p_hod = {
            'logMmin': 13.03,
            'sigma_logM': 0.38,
            'logM0': 13.27,
            'logM1': 14.08,
            'alpha': 0.76,
            'mucen': 0.0,
            'musat': 0.0,
        }
        self.set_parameters(p_hod)
        new_p_hod = self.process_measured_hod(p_hod)
        self.set_parameters(new_p_hod)


class Leauthaud11(Hod_model):
    """
    Leauthaud+11 HOD model
    """

    def __init__(
        self,
        parameters=[
            'smhm_m0_0',
            'smhm_m0_a',
            'smhm_m1_0',
            'smhm_m1_a',
            'smhm_beta_0',
            'smhm_beta_a',
            'smhm_delta_0',
            'smhm_delta_a',
            'smhm_gamma_0',
            'smhm_gamma_a',
            'scatter_model_param1',
            'alphasat',
            'betasat',
            'bsat',
            'betacut',
            'bcut'
        ],
        lower_bound=np.array([
            10.0, -1.0, 12.0, -0.5, 0.3, -0.1, 0.5, -
            0.4, 1.0, 0.5, 0.1, 0, 0.5, 10.0, -0.2, 1.0,
        ]),
        upper_bound=np.array([
            11.0, 1.0, 13.0, 0.8, 0.6, 0.4, 0.8, 0.6, 1.8, 3.0, 0.2, 1.5, 1.3, 11.0, 0.1, 2.0,
        ]),
        param_defaults=None
    ):
        super().__init__(parameters, lower_bound, upper_bound)

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == 'behroozi10':
                self.behroozi10()
            else:
                raise NotImplementedError

    def behroozi10(self):
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
        self.set_parameters(p_hod)


class Zu_mandelbaum15(Hod_model):
    """
    Zu & Mandelbaum+15 HOD model
    """

    def __init__(
        self,
        parameters=[
            'smhm_m0',
            'smhm_m1',
            'smhm_beta',
            'smhm_delta',
            'smhm_gamma',
            'smhm_sigma',
            'smhm_sigma_slope',
            'alphasat',
            'betasat',
            'bsat',
            'betacut',
            'bcut',
        ],
        lower_bound=np.array([
            9.0, 9.5, 0.0, 0.0, -0.1, 0.01, -0.4, 0.5, 0.1, 0.01, -0.05, 0.0,
        ]),
        upper_bound=np.array([
            13.0, 14.0, 2.0, 1.5, 4.9, 3.0, 0.4, 1.5, 1.8, 25.0, 1.50, 6.0,
        ]),
        param_defaults=None
    ):
        super().__init__(parameters, lower_bound, upper_bound)

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == 'zu_mandelbaum15':
                self.behroozi10()
            else:
                raise NotImplementedError

    def zu_mandelbaum15(self):
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
        self.set_parameters(p_hod)


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


def build_HOD_model(
    cosmology, model, theta, zf, mdef='vir',
):
    '''Build a HOD model from the given HOD parameters.

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

    Returns:
        hod_model (HODMockFactory): A HOD model object
            that can be used with Halotools.
    '''
    # determine mass column
    mkey = 'halo_m' + mdef

    # Get HOD parameters
    hod_params = dict(theta)

    # Occupation functions
    if model == 'zheng07':
        cenocc = Zheng07Cens(prim_haloprop_key=mkey)
        satocc = Zheng07Sats(
            prim_haloprop_key=mkey,
            cenocc_model=cenocc,
            modulate_with_cenocc=True
        )
    elif model == 'zheng07zdep':
        cenocc = Zheng07zdepCens(prim_haloprop_key=mkey)
        satocc = Zheng07zdepSats(
            prim_haloprop_key=mkey,
            cenocc_model=cenocc,
            modulate_with_cenocc=True
        )
    elif model == 'leauthaud11':
        cenocc = Leauthaud11Cens(prim_haloprop_key=mkey, redshift=zf)
        satocc = Leauthaud11Sats(
            prim_haloprop_key=mkey,
            cenocc_model=cenocc, redshift=zf,
        )
    elif model == 'zu_mandelbaum15':
        cenocc = ZuMandelbaum15Cens(prim_haloprop_key=mkey, redshift=zf)
        satocc = ZuMandelbaum15Sats(prim_haloprop_key=mkey)
        satocc.central_occupation_model = cenocc  # need to set this manually
        # m0 and m1 are desired in real units
        hod_params['smhm_m0'] = 10**hod_params['smhm_m0']
        hod_params['smhm_m1'] = 10**hod_params['smhm_m1']
    else:
        raise NotImplementedError

    # Set HOD parameters
    cenocc.param_dict.update(hod_params)
    satocc.param_dict.update(hod_params)
    satocc._suppress_repeated_param_warning = True

    # profile functions
    censprof = TrivialPhaseSpace(
        cosmology=cosmology,
        redshift=zf,
        mdef=mdef
    )
    satsprof = NFWPhaseSpace(
        conc_mass_model='direct_from_halo_catalog',
        cosmology=cosmology,
        redshift=zf,
        mdef=mdef
    )

    # make the model
    model = dict(
        centrals_occupation=cenocc,
        centrals_profile=censprof,
        satellites_occupation=satocc,
        satellites_profile=satsprof
    )
    return HodModelFactory(**model)
