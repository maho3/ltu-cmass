"""
HOD models. Currently implemented models:
- Zheng+07
- Leauthaud+11
- Zu & Mandelbaum+15

Each model derives from the `Hod_model` parent class,
which additionally uses the `Hod_parameter` helper class
for each parameter.
"""

import numpy as np

from halotools.empirical_models import (
    Zheng07Cens,
    Zheng07Sats,
    Leauthaud11Cens,
    Leauthaud11Sats,
    ZuMandelbaum15Cens,
    ZuMandelbaum15Sats,
    NFWPhaseSpace,
    TrivialPhaseSpace,
    AssembiasZheng07Cens,
    AssembiasZheng07Sats,
    BiasedNFWPhaseSpace,
)
from halotools.empirical_models import HodModelFactory


class Hod_model:
    """
    Parent class defining a HOD model.
    Includes methods for getting, setting and sampling parameters,
    and for initialising `halotools` objects.
    """

    def __init__(
        self,
        param_keys,
        param_values=None,
        lower_bound=None,
        upper_bound=None,
        mass_def="vir",
        assem_bias=False,
        vel_assem_bias=False,
    ):
        self.parameters = {_param: None for _param in param_keys}
        self.mass_def = mass_def
        self.mass_key = "halo_m" + self.mass_def
        self.lower_bound = {
            _param: _lb for _param, _lb in zip(param_keys, lower_bound)
        }
        self.upper_bound = {
            _param: _ub for _param, _ub in zip(param_keys, upper_bound)
        }

        if param_values is not None:
            self.set_parameters(param_values)

        self.cenocc = None
        self.satocc = None
        self.censprof = None
        self.satsprof = None

        self.assem_bias = assem_bias
        self.vel_assem_bias = vel_assem_bias

    def set_parameter(self, key, new_parameter):
        """
        Set a single parameter
        """
        self.parameters[key] = new_parameter

    def set_parameters(self, new_parameters):
        """
        Set all parameters using a dict
        """
        for _param in self.parameters:
            self.parameters[_param] = new_parameters[_param]

    def sample_parameters(self):
        if (self.lower_bound is None) or (self.upper_bound is None):
            raise ValueError(
                "Lower and upper bounds for each parameter must be set"
            )

        for _param in self.parameters.keys():
            # Get upper and lower bounds for this parameter
            _lower = self.lower_bound[_param]
            _upper = self.upper_bound[_param]

            # Sample new parameter
            sampled_param = np.random.uniform(_lower, _upper)

            # Set new parameter
            self.parameters[_param] = sampled_param

    def get_parameters(self):
        """
        Return a dict of parameter key / value pairs
        """
        # out_params = {}
        # for _param in self.parameters:
        #     out_params[_param] = getattr(self, _param).value

        # return out_params
        return self.parameters

    def get_model(self):
        if (self.cenocc is None) | (self.satocc is None):
            raise ValueError("Occupation models not set")

        if (self.censprof is None) | (self.satsprof is None):
            raise ValueError("Profile models not set")

        model = dict(
            centrals_occupation=self.cenocc,
            centrals_profile=self.censprof,
            satellites_occupation=self.satocc,
            satellites_profile=self.satsprof,
        )
        return HodModelFactory(**model)


class Zheng07(Hod_model):
    """
    Zheng+07 HOD model.
    """

    def __init__(
        self,
        param_keys=["logMmin", "sigma_logM", "logM0", "logM1", "alpha"],
        lower_bound=np.array([12.0, 0.1, 13.0, 13.0, 0.0]),
        upper_bound=np.array([14.0, 0.6, 15.0, 15.0, 1.5]),
        param_defaults=None,
    ):
        super().__init__(
            param_keys=param_keys,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == "parejko2013_lowz":
                self.parejko2013_lowz()
            elif self.param_defaults == "manera2015_lowz_ngc":
                self.manera2015_lowz_ngc()
            elif self.param_defaults == "manera2015_lowz_sgc":
                self.manera2015_lowz_sgc()
            elif self.param_defaults == "reid2014_cmass":
                self.reid2014_cmass()
            else:
                raise NotImplementedError

    def set_occupation(self, **kwargs):
        if self.assem_bias:
            self.cenocc = AssembiasZheng07Cens(
                prim_haloprop_key=self.mass_key, **kwargs
            )
            self.satocc = AssembiasZheng07Sats(
                prim_haloprop_key=self.mass_key,
                cenocc_model=self.cenocc,
                **kwargs,
            )
        else:
            self.cenocc = Zheng07Cens(prim_haloprop_key=self.mass_key)
            self.satocc = Zheng07Sats(
                prim_haloprop_key=self.mass_key,
                cenocc_model=self.cenocc,
                modulate_with_cenocc=True,
            )

        self.cenocc.param_dict.update(self.get_parameters())
        self.satocc.param_dict.update(self.get_parameters())
        self.satocc._suppress_repeated_param_warning = True

    def set_profiles(
        self, cosmology, zf, conc_mass_model="dutton_maccio14", **kwargs
    ):
        if self.assem_bias:
            self.censprof = TrivialPhaseSpace(**kwargs)
            self.satsprof = BiasedNFWPhaseSpace(
                conc_mass_model=conc_mass_model, **kwargs
            )
        else:
            self.censprof = TrivialPhaseSpace(
                cosmology=cosmology, redshift=zf, mdef=self.mass_def
            )
            self.satsprof = NFWPhaseSpace(
                conc_mass_model="direct_from_halo_catalog",
                cosmology=cosmology,
                redshift=zf,
                mdef=self.mass_def,
            )

    def parejko2013_lowz(self):
        """
        lowz catalog from Parejko+2013 Table 3. Note that the
        parameterization is slightly different so the numbers need to
        be converted.
        """
        p_hod = {
            "logMmin": 13.25,
            "sigma_logM": 0.43,  # 0.7 * sqrt(2) * log10(e)
            "logM0": 13.27,  # log10(kappa * Mmin)
            "logM1": 14.18,
            "alpha": 0.94,
        }
        self.set_parameters(p_hod)

    def manera2015_lowz_ngc(self):
        """
        best-fit HOD of the lowz catalog NGC from Table 2 of Manera et al.(2015)
        """
        p_hod = {
            "logMmin": 13.20,
            "sigma_logM": 0.62,
            "logM0": 13.24,
            "logM1": 14.32,
            "alpha": 0.9,
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
            "logMmin": 13.14,
            "sigma_logM": 0.55,
            "logM0": 13.43,
            "logM1": 14.58,
            "alpha": 0.93,
        }
        self.set_parameters(p_hod)

    def reid2014_cmass(self):
        """
        best-fit HOD from Reid et al. (2014) Table 4
        """
        p_hod = {
            "logMmin": 13.03,
            "sigma_logM": 0.38,
            "logM0": 13.27,
            "logM1": 14.08,
            "alpha": 0.76,
        }
        self.set_parameters(p_hod)


class Leauthaud11(Hod_model):
    """
    Leauthaud+11 HOD model
    """

    def __init__(
        self,
        parameters=[
            "smhm_m0_0",
            "smhm_m0_a",
            "smhm_m1_0",
            "smhm_m1_a",
            "smhm_beta_0",
            "smhm_beta_a",
            "smhm_delta_0",
            "smhm_delta_a",
            "smhm_gamma_0",
            "smhm_gamma_a",
            "scatter_model_param1",
            "alphasat",
            "betasat",
            "bsat",
            "betacut",
            "bcut",
        ],
        lower_bound=np.array(
            [
                10.0,
                -1.0,
                12.0,
                -0.5,
                0.3,
                -0.1,
                0.5,
                -0.4,
                1.0,
                0.5,
                0.1,
                0,
                0.5,
                10.0,
                -0.2,
                1.0,
            ]
        ),
        upper_bound=np.array(
            [
                11.0,
                1.0,
                13.0,
                0.8,
                0.6,
                0.4,
                0.8,
                0.6,
                1.8,
                3.0,
                0.2,
                1.5,
                1.3,
                11.0,
                0.1,
                2.0,
            ]
        ),
        param_defaults=None,
    ):
        super().__init__(
            parameters=parameters,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == "behroozi10":
                self.behroozi10()
            else:
                raise NotImplementedError

    def set_occupation(self, zf):
        self.cenocc = Leauthaud11Cens(
            prim_haloprop_key=self.mass_key, redshift=zf
        )
        self.satocc = Leauthaud11Sats(
            prim_haloprop_key=self.mass_key,
            cenocc_model=self.cenocc,
            redshift=zf,
        )

    def set_profiles(self, cosmology, zf):
        self.censprof = TrivialPhaseSpace(
            cosmology=cosmology, redshift=zf, mdef=self.mass_def
        )
        self.satsprof = NFWPhaseSpace(
            conc_mass_model="direct_from_halo_catalog",
            cosmology=cosmology,
            redshift=zf,
            mdef=self.mass_def,
        )

    def behroozi10(self):
        """
        Best-fit HOD from Behroozi+10 (arXiv:1001.0015; table 2, column 1)
        with redshift dependence, as well as bes fit satellite occupation
        parameters from Leauthaud+12 (arXiv:1104.0928, Table 5, SIGMOD1).
        """
        p_hod = {
            "smhm_m0_0": 10.72,
            "smhm_m0_a": 0.55,
            "smhm_m1_0": 12.35,
            "smhm_m1_a": 0.28,
            "smhm_beta_0": 0.44,
            "smhm_beta_a": 0.18,
            "smhm_delta_0": 0.57,
            "smhm_delta_a": 0.17,
            "smhm_gamma_0": 1.56,
            "smhm_gamma_a": 2.51,
            "scatter_model_param1": 0.15,
            "alphasat": 1,
            "betasat": 0.859,
            "bsat": 10.62,
            "betacut": -0.13,
            "bcut": 1.47,
        }
        self.set_parameters(p_hod)


class Zu_mandelbaum15(Hod_model):
    """
    Zu & Mandelbaum+15 HOD model
    """

    def __init__(
        self,
        parameters=[
            "smhm_m0",
            "smhm_m1",
            "smhm_beta",
            "smhm_delta",
            "smhm_gamma",
            "smhm_sigma",
            "smhm_sigma_slope",
            "alphasat",
            "betasat",
            "bsat",
            "betacut",
            "bcut",
        ],
        lower_bound=np.array(
            [
                9.0,
                9.5,
                0.0,
                0.0,
                -0.1,
                0.01,
                -0.4,
                0.5,
                0.1,
                0.01,
                -0.05,
                0.0,
            ]
        ),
        upper_bound=np.array(
            [
                13.0,
                14.0,
                2.0,
                1.5,
                4.9,
                3.0,
                0.4,
                1.5,
                1.8,
                25.0,
                1.50,
                6.0,
            ]
        ),
        param_defaults=None,
    ):
        super().__init__(
            parameters=parameters,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == "zu_mandelbaum15":
                self.behroozi10()
            else:
                raise NotImplementedError

    def set_occupation(self, zf):
        self.cenocc = ZuMandelbaum15Cens(
            prim_haloprop_key=self.mass_key, redshift=zf
        )
        self.satocc = ZuMandelbaum15Sats(prim_haloprop_key=self.mass_key)
        self.satocc.central_occupation_model = (
            self.cenocc  # need to set this manually
        )

        # m0 and m1 are desired in real units
        self.parameters["smhm_m0"] = 10 ** self.parameters["smhm_m0"]
        self.parameters["smhm_m1"] = 10 ** self.parameters["smhm_m1"]

        self.cenocc.param_dict.update(self.parameters)
        self.satocc.param_dict.update(self.parameters)
        self.satocc._suppress_repeated_param_warning = True

    def set_profiles(self, cosmology, zf):
        self.censprof = TrivialPhaseSpace(
            cosmology=cosmology, redshift=zf, mdef=self.mass_def
        )
        self.satsprof = NFWPhaseSpace(
            conc_mass_model="direct_from_halo_catalog",
            cosmology=cosmology,
            redshift=zf,
            mdef=self.mass_def,
        )

    def zu_mandelbaum15(self):
        """
        Zu \& Mandelbaum+15, arXiv:1505.02781
        Table 2, iHOD
        """
        p_hod = {
            "smhm_m0": 10.31,
            "smhm_m1": 12.10,
            "smhm_beta": 0.33,
            "smhm_delta": 0.42,
            "smhm_gamma": 1.21,
            "smhm_sigma": 0.50,
            "smhm_sigma_slope": -0.04,
            "alphasat": 1.00,
            "betasat": 0.90,
            "bsat": 8.98,
            "betacut": 0.41,
            "bcut": 0.86,
        }
        self.set_parameters(p_hod)
