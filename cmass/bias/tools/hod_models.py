"""
HOD models. Currently implemented models:
- Zheng+07
- Leauthaud+11
- Zu & Mandelbaum+15

Each model derives from the `Hod_model` parent class,
which additionally uses the `Hod_parameter` helper class
for each parameter.

QUESTIONS:
- In self.satsprof = Satellites_vBiasedNFWPhaseSpace, should I supply conc_gal_bias_bins:
"""

import warnings
import numpy as np
from scipy.special import erf
from itertools import zip_longest
from scipy.stats import truncnorm

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
)
from halotools.empirical_models import HodModelFactory, HeavisideAssembias

from .phase_space_models import (
    Centrals_vBiasedNFWPhaseSpace,
    Satellites_vBiasedNFWPhaseSpace,
)


def truncated_gaussian(mean, std, lower, upper, size=1):
    a, b = (lower - mean) / std, (upper - mean) / std  # Compute standardised bounds
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


class Hod_parameter:
    """
    Helper class defining a HOD parameter, including the value
    and upper and lower bounds (for a flat prior)
    """

    def __init__(
        self, key, value=None, upper=None, lower=None, sigma=None, distribution=None
    ):
        self.key = key
        self.value = value
        self.upper = upper
        self.lower = lower
        self.sigma = sigma
        setattr(
            self, "distribution", "uniform" if distribution is None else distribution
        )


class Hod_model:
    """
    Parent class defining a HOD model.
    Includes methods for setting, getting and sampling parameters,
    and for initialising `halotools` objects.
    """

    def __init__(
        self,
        parameters,
        lower_bound,
        upper_bound,
        distribution=None,
        sigma=None,
        mass_def="vir",
        assem_bias=False,
        vel_assem_bias=False,
    ):
        self.parameters = parameters

        # Loop through parameters and initialise
        zipped = zip_longest(
            self.parameters,
            lower_bound,
            upper_bound,
            distribution or [],
            sigma or [],
            fillvalue=None,
        )

        for _param, _lower, _upper, _dist, _sigma in zipped:
            setattr(
                self,
                _param,
                Hod_parameter(
                    key=_param,
                    lower=_lower,
                    upper=_upper,
                    sigma=_sigma,
                    distribution=_dist,
                ),
            )

        self.mass_def = mass_def
        self.mass_key = "halo_m" + self.mass_def
        self.conc_key = "halo_nfw_conc"

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
        getattr(self, key).value = new_parameter

    def set_parameters(self, new_parameters):
        """
        Set all parameters using a dict
        """
        for _param in self.parameters:
            getattr(self, _param).value = new_parameters[_param]

    def sample_parameters(self):
        for _param in self.parameters:
            _dist = getattr(self, _param).distribution
            if _dist == "uniform":
                _lower = getattr(self, _param).lower
                _upper = getattr(self, _param).upper
                sampled_param = np.random.uniform(_lower, _upper)
            elif _dist == "norm":
                sampled_param = np.random.normal(0, getattr(self, _param).sigma)
            elif _dist == "truncnorm":
                _lower = getattr(self, _param).lower
                _upper = getattr(self, _param).upper
                _sigma = getattr(self, _param).sigma
                sampled_param = float(truncated_gaussian(0, _sigma, _lower, _upper)[0])
            else:
                raise NotImplementedError
            getattr(self, _param).value = sampled_param

    def get_parameters(self):
        """
        Return a dict of parameter key / value pairs
        """
        out_params = {}
        for _param in self.parameters:
            out_params[_param] = getattr(self, _param).value

        return out_params

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

    def set_profiles(self, cosmology, zf, conc_mass_model="dutton_maccio14", **kwargs):
        if self.vel_assem_bias:
            self.censprof = Centrals_vBiasedNFWPhaseSpace(
                cosmology=cosmology,
                redshift=zf,
                mdef=self.mass_def,
                conc_mass_model="direct_from_halo_catalog",
            )
            self.censprof.set_parameters(self.get_parameters())
            self.satsprof = Satellites_vBiasedNFWPhaseSpace(
                conc_key=self.conc_key,
                cosmology=cosmology,
                redshift=zf,
                mdef=self.mass_def,
                conc_mass_model="direct_from_halo_catalog",
            )
            self.satsprof.set_parameters(self.get_parameters())
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


class Zheng07(Hod_model):
    """
    Zheng+07 HOD model.
    """

    def __init__(
        self,
        parameters=[
            "logMmin",
            "sigma_logM",
            "logM0",
            "logM1",
            "alpha",
            "eta_vb_centrals",
            "eta_vb_satellites",
            "conc_gal_bias_satellites",
            "mean_occupation_centrals_assembias_param1",
            "mean_occupation_satellites_assembias_param1",
        ],
        lower_bound=[12.0, 0.1, 13.0, 13.0, 0.0, 0.0, 0.2, 0.2, -1, -1],
        upper_bound=[14.0, 0.6, 15.0, 15.0, 1.5, 0.7, 2.0, 2.0, 1, 1],
        sigma=[None, None, None, None, None, None, None, None, 0.2, 0.2],
        distribution=[
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "truncnorm",
            "truncnorm",
        ],
        param_defaults=None,
        mass_def="vir",
        assem_bias=False,
        vel_assem_bias=False,
    ):
        # Determine which parameters we need
        pars = ["logMmin", "sigma_logM", "logM0", "logM1", "alpha"]
        if assem_bias:
            pars += [
                "mean_occupation_centrals_assembias_param1",
                "mean_occupation_satellites_assembias_param1",
            ]
        if vel_assem_bias:
            pars += ["eta_vb_centrals", "eta_vb_satellites", "conc_gal_bias_satellites"]
        low = [lower_bound[parameters.index(p)] for p in pars]
        up = [upper_bound[parameters.index(p)] for p in pars]
        sig = [sigma[parameters.index(p)] for p in pars]
        dist = [distribution[parameters.index(p)] for p in pars]

        super().__init__(
            parameters=pars,
            lower_bound=low,
            upper_bound=up,
            sigma=sig,
            distribution=dist,
            mass_def=mass_def,
            assem_bias=assem_bias,
            vel_assem_bias=vel_assem_bias,
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

    def set_occupation(
        self,
    ):
        if self.assem_bias:
            self.cenocc = AssembiasZheng07Cens(
                prim_haloprop_key=self.mass_key,
                sec_haloprop_key=self.conc_key,
                split=0.5,
                assembias_strength=self.get_parameters()[
                    "mean_occupation_centrals_assembias_param1"
                ],
            )
            self.satocc = AssembiasZheng07Sats(
                prim_haloprop_key=self.mass_key,
                sec_haloprop_key=self.conc_key,
                split=0.5,
                assembias_strength=self.get_parameters()[
                    "mean_occupation_satellites_assembias_param1"
                ],
                cenocc_model=self.cenocc,
                modulate_with_cenocc=True,
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
        if self.assem_bias:
            p_hod["mean_occupation_centrals_assembias_param1"] = 0.0
            p_hod["mean_occupation_satellites_assembias_param1"] = 0.0
        if self.vel_assem_bias:
            p_hod["eta_vb_centrals"] = 0.0
            p_hod["eta_vb_satellites"] = 1.0
            p_hod["conc_gal_bias_satellites"] = 1.0
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
        if self.assem_bias:
            p_hod["mean_occupation_centrals_assembias_param1"] = 0.0
            p_hod["mean_occupation_satellites_assembias_param1"] = 0.0
        if self.vel_assem_bias:
            p_hod["eta_vb_centrals"] = 0.0
            p_hod["eta_vb_satellites"] = 1.0
            p_hod["conc_gal_bias_satellites"] = 1.0
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
        if self.assem_bias:
            p_hod["mean_occupation_centrals_assembias_param1"] = 0.0
            p_hod["mean_occupation_satellites_assembias_param1"] = 0.0
        if self.vel_assem_bias:
            p_hod["eta_vb_centrals"] = 0.0
            p_hod["eta_vb_satellites"] = 1.0
            p_hod["conc_gal_bias_satellites"] = 1.0
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
        if self.assem_bias:
            p_hod["mean_occupation_centrals_assembias_param1"] = 0.0
            p_hod["mean_occupation_satellites_assembias_param1"] = 0.0
        if self.vel_assem_bias:
            p_hod["eta_vb_centrals"] = 0.0
            p_hod["eta_vb_satellites"] = 1.0
            p_hod["conc_gal_bias_satellites"] = 1.0
        self.set_parameters(p_hod)


class Zheng07zdep(Hod_model):
    """
    Zheng+07 HOD model with redshift dependent mass parameters
    """

    def __init__(
        self,
        parameters=[
            "logMmin",
            "sigma_logM",
            "logM0",
            "logM1",
            "alpha",
            "mucen",
            "musat",
            "eta_vb_centrals",
            "eta_vb_satellites",
            "conc_gal_bias_satellites",
            "mean_occupation_centrals_assembias_param1",
            "mean_occupation_satellites_assembias_param1",
        ],
        lower_bound=np.array([12.0, 0.1, 13.0, 13.0, 0.0, -30.0, -10.0, 0.0, 0.2, 0.2, -1, -1]),
        upper_bound=np.array([14.0, 0.6, 15.0, 15.0, 1.5, 0.0, 0.0, 0.7, 2.0, 2.0, 1, 1]),
        sigma=[None, None, None, None, None, None, None, None, None, None, 0.2, 0.2],
        distribution=[
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "truncnorm",
            "truncnorm",
        ],
        param_defaults=None,
        mass_def="vir",
        assem_bias=False,
        vel_assem_bias=False,
    ):
        
        # Determine which parameters we need
        pars = ["logMmin", "sigma_logM", "logM0", "logM1", "alpha", "mucen", "musat"]
        if assem_bias:
            pars += [
                "mean_occupation_centrals_assembias_param1",
                "mean_occupation_satellites_assembias_param1",
            ]
        if vel_assem_bias:
            pars += ["eta_vb_centrals", "eta_vb_satellites", "conc_gal_bias_satellites"]
        low = [lower_bound[parameters.index(p)] for p in pars]
        up = [upper_bound[parameters.index(p)] for p in pars]
        sig = [sigma[parameters.index(p)] for p in pars]
        dist = [distribution[parameters.index(p)] for p in pars]

        super().__init__(
            parameters=pars,
            lower_bound=low,
            upper_bound=up,
            sigma=sig,
            distribution=dist,
            mass_def=mass_def,
            assem_bias=assem_bias,
            vel_assem_bias=vel_assem_bias,
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
            self.cenocc = AssembiasZheng07zdepCens(
                prim_haloprop_key=self.mass_key,
                sec_haloprop_key=self.conc_key,
                split=0.5,
                assembias_strength=self.get_parameters()[
                    "mean_occupation_centrals_assembias_param1"
                ],
            )
            self.satocc = AssembiasZheng07zdepSats(
                prim_haloprop_key=self.mass_key,
                sec_haloprop_key=self.conc_key,
                split=0.5,
                assembias_strength=self.get_parameters()[
                    "mean_occupation_satellites_assembias_param1"
                ],
                cenocc_model=self.cenocc,
                modulate_with_cenocc=True,
            )
        else:
            self.cenocc = Zheng07zdepCens(prim_haloprop_key=self.mass_key)
            self.satocc = Zheng07zdepSats(
                prim_haloprop_key=self.mass_key,
                cenocc_model=self.cenocc,
                modulate_with_cenocc=True,
            )

        self.cenocc.param_dict.update(self.get_parameters())
        self.satocc.param_dict.update(self.get_parameters())
        self.satocc._suppress_repeated_param_warning = True

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
            "mucen": 0.0,
            "musat": 0.0,
        }
        if self.assem_bias:
            p_hod["mean_occupation_centrals_assembias_param1"] = 0.0
            p_hod["mean_occupation_satellites_assembias_param1"] = 0.0
        if self.vel_assem_bias:
            p_hod["eta_vb_centrals"] = 0.0
            p_hod["eta_vb_satellites"] = 1.0
            p_hod["conc_gal_bias_satellites"] = 1.0
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
            "mucen": 0.0,
            "musat": 0.0,
        }
        if self.assem_bias:
            p_hod["mean_occupation_centrals_assembias_param1"] = 0.0
            p_hod["mean_occupation_satellites_assembias_param1"] = 0.0
        if self.vel_assem_bias:
            p_hod["eta_vb_centrals"] = 0.0
            p_hod["eta_vb_satellites"] = 1.0
            p_hod["conc_gal_bias_satellites"] = 1.0
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
            "mucen": 0.0,
            "musat": 0.0,
        }
        if self.assem_bias:
            p_hod["mean_occupation_centrals_assembias_param1"] = 0.0
            p_hod["mean_occupation_satellites_assembias_param1"] = 0.0
        if self.vel_assem_bias:
            p_hod["eta_vb_centrals"] = 0.0
            p_hod["eta_vb_satellites"] = 1.0
            p_hod["conc_gal_bias_satellites"] = 1.0
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
            "mucen": 0.0,
            "musat": 0.0,
        }
        if self.assem_bias:
            p_hod["mean_occupation_centrals_assembias_param1"] = 0.0
            p_hod["mean_occupation_satellites_assembias_param1"] = 0.0
        if self.vel_assem_bias:
            p_hod["eta_vb_centrals"] = 0.0
            p_hod["eta_vb_satellites"] = 1.0
            p_hod["conc_gal_bias_satellites"] = 1.0
        self.set_parameters(p_hod)


class Zheng07zinterp(Hod_model):
    """
    Zheng+07 HOD model with redshift dependent mass parameters
    """

    def __init__(
        self,
        zpivot,
        parameters=[
            "logMmin",
            "sigma_logM",
            "logM0",
            "logM1",
            "alpha",
            "eta_vb_centrals",
            "eta_vb_satellites",
            "conc_gal_bias_satellites",
            "mean_occupation_centrals_assembias_param1",
            "mean_occupation_satellites_assembias_param1",
        ],
        lower_bound=[12.0, 0.1, 13.0, 13.0, 0.0, 0.0, 0.2, 0.2, -1, -1],
        upper_bound=[14.0, 0.6, 15.0, 15.0, 1.5, 0.7, 2.0, 2.0, 1, 1],
        sigma=[None, None, None, None, None, None, None, None, 0.2, 0.2],
        distribution=[
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "truncnorm",
            "truncnorm",
        ],
        param_defaults=None,
        mass_def="vir",
        assem_bias=False,
        vel_assem_bias=False,
    ):

        # Obtain single mass parameter for each pivot point
        pars = []
        low = []
        up = []
        sig = []
        dist = []
        self.zpivot = zpivot
        self.npivot = len(zpivot)
        assert self.npivot > 1, "Cannot interpolate between fewer than 2 points"
        if param_defaults is None:
            defaults = None
        else:
            defaults = []
        init_pars = ["logMmin", "sigma_logM", "logM0", "logM1", "alpha"]
        for p, v0, v1 in zip(init_pars, lower_bound, upper_bound):
            i = parameters.index(p)
            if p in ["logMmin", "logM0", "logM1"]:
                for j in range(self.npivot):
                    pars.append(p + "_z" + str(j))
                    low.append(v0)
                    up.append(v1)
                    sig.append(sigma[i])
                    dist.append(distribution[i])
                    if param_defaults is not None:
                        defaults.append(param_defaults[i])
            else:
                pars.append(p)
                low.append(v0)
                up.append(v1)
                sig.append(sigma[i])
                dist.append(distribution[i])
                if param_defaults is not None:
                    defaults.append(param_defaults[i])

        # Add assembly bias parameters if needed
        if assem_bias:
            for p in ["mean_occupation_centrals_assembias_param1", "mean_occupation_satellites_assembias_param1"]:
                pars.append(p)
                low.append(lower_bound[parameters.index(p)])
                up.append(upper_bound[parameters.index(p)])
                sig.append(sigma[parameters.index(p)])
                dist.append(distribution[parameters.index(p)])
        if vel_assem_bias:
            for p in ["eta_vb_centrals", "eta_vb_satellites", "conc_gal_bias_satellites"]:
                pars.append(p)
                low.append(lower_bound[parameters.index(p)])
                up.append(upper_bound[parameters.index(p)])
                sig.append(sigma[parameters.index(p)])
                dist.append(distribution[parameters.index(p)])

        super().__init__(
            parameters=pars,
            lower_bound=low,
            upper_bound=up,
            sigma=sig,
            distribution=dist,
            mass_def=mass_def,
            assem_bias=assem_bias,
            vel_assem_bias=vel_assem_bias,
        )

        # If using, set literature values for parameters
        self.param_defaults = defaults
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
            self.cenocc = AssembiasZheng07zinterpCens(
                zpivot=self.zpivot,
                prim_haloprop_key=self.mass_key,
                sec_haloprop_key=self.conc_key,
                split=0.5,
                assembias_strength=self.get_parameters()[
                    "mean_occupation_centrals_assembias_param1"
                ],
            )
            self.satocc = AssembiasZheng07zinterpSats(
                zpivot=self.zpivot,
                prim_haloprop_key=self.mass_key,
                sec_haloprop_key=self.conc_key,
                split=0.5,
                assembias_strength=self.get_parameters()[
                    "mean_occupation_satellites_assembias_param1"
                ],
                cenocc_model=self.cenocc,
                modulate_with_cenocc=True,
            )
        else:
            self.cenocc = Zheng07zinterpCens(
                self.zpivot, prim_haloprop_key=self.mass_key
            )
            self.satocc = Zheng07zinterpSats(
                self.zpivot,
                prim_haloprop_key=self.mass_key,
                cenocc_model=self.cenocc,
                modulate_with_cenocc=True,
            )

        self.cenocc.param_dict.update(self.get_parameters())
        self.satocc.param_dict.update(self.get_parameters())
        self.satocc._suppress_repeated_param_warning = True

    def process_measured_hod(self, p_hod):
        new_p_hod = {}
        for k, v in p_hod.items():
            if k in ["logMmin", "logM0", "logM1"]:
                for j in range(self.npivot):
                    new_p_hod[k + "_z" + str(j)] = v
            else:
                new_p_hod[k] = v
        if self.assem_bias:
            new_p_hod["mean_occupation_centrals_assembias_param1"] = 0.0
            new_p_hod["mean_occupation_satellites_assembias_param1"] = 0.0
        if self.vel_assem_bias:
            new_p_hod["eta_vb_centrals"] = 0.0
            new_p_hod["eta_vb_satellites"] = 1.0
            new_p_hod["conc_gal_bias_satellites"] = 1.0
        return new_p_hod

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
        new_p_hod = self.process_measured_hod(p_hod)
        self.set_parameters(new_p_hod)

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
            "mucen": 0.0,
            "musat": 0.0,
        }
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
            "logMmin": 13.14,
            "sigma_logM": 0.55,
            "logM0": 13.43,
            "logM1": 14.58,
            "alpha": 0.93,
            "mucen": 0.0,
            "musat": 0.0,
        }
        new_p_hod = self.process_measured_hod(p_hod)
        self.set_parameters(new_p_hod)

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
            "mucen": 0.0,
            "musat": 0.0,
        }
        new_p_hod = self.process_measured_hod(p_hod)
        self.set_parameters(new_p_hod)


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
        mass_def="vir",
        zf=None,
        assem_bias=False,
        vel_assem_bias=False,
    ):
        if assem_bias or vel_assem_bias:
            raise NotImplementedError

        super().__init__(
            parameters=parameters,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mass_def=mass_def,
        )
        self.zf = zf

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == "behroozi10":
                self.behroozi10()
            else:
                raise NotImplementedError

    def set_occupation(
        self,
    ):
        self.cenocc = Leauthaud11Cens(prim_haloprop_key=self.mass_key, redshift=self.zf)
        self.satocc = Leauthaud11Sats(
            prim_haloprop_key=self.mass_key,
            cenocc_model=self.cenocc,
            redshift=self.zf,
        )
        self.cenocc.param_dict.update(self.get_parameters())
        self.satocc.param_dict.update(self.get_parameters())
        self.satocc._suppress_repeated_param_warning = True

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
        mass_def="vir",
        zf=None,
        assem_bias=False,
        vel_assem_bias=False,
    ):
        if assem_bias or vel_assem_bias:
            raise NotImplementedError

        super().__init__(
            parameters=parameters,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            mass_def=mass_def,
            assem_bias=assem_bias,
            vel_assem_bias=vel_assem_bias,
        )
        self.zf = zf

        # If using, set literature values for parameters
        self.param_defaults = param_defaults
        if self.param_defaults is not None:
            if self.param_defaults == "zu_mandelbaum15":
                self.behroozi10()
            else:
                raise NotImplementedError

    def set_occupation(
        self,
    ):
        self.cenocc = ZuMandelbaum15Cens(
            prim_haloprop_key=self.mass_key, redshift=self.zf
        )
        self.satocc = ZuMandelbaum15Sats(prim_haloprop_key=self.mass_key)
        self.satocc.central_occupation_model = (
            self.cenocc  # need to set this manually
        )

        # m0 and m1 are desired in real units
        pars = self.get_parameters()
        self.set_parameter("smhm_m0", 10 ** pars["smhm_m0"])
        self.set_parameter("smhm_m1", 10 ** pars["smhm_m1"])

        self.cenocc.param_dict.update(self.get_parameters())
        self.satocc.param_dict.update(self.get_parameters())
        self.satocc._suppress_repeated_param_warning = True

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


def logM_i(z, logM_i_pivot, mu_i_p, z_pivot):
    """
    Apply a linear dependence in a = 1 / (1 + z) to the logarithm of a mass variables
    
    Args:
        :z (float): Cosmological redshift
        :logM_i_pivot (float): The value of the mass parameter at the pivot redshift
        :mu_i_p (float): Slope of the logmass-a relation
        :z_pivot (float): The pivot redshift
        
    Returns:
        float: The log-mass variable at the requested cosmological redshift
    """
    return logM_i_pivot + mu_i_p * ((1 / (1 + z)) - (1 / (1 + z_pivot)))


class Zheng07zdepCens(Zheng07Cens):
    # Params: logMmin, sigma_logM, mucen, zpivot
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.param_dict["zpivot"] = 0.5

        self.list_of_haloprops_needed = ["halo_redshift"]

    def mean_occupation(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        mass = kwargs["table"][self.prim_haloprop_key]
        redshift = kwargs["table"]["halo_redshift"]
        logM = np.log10(mass)

        logMmin = logM_i(
            redshift,
            self.param_dict["logMmin"],
            self.param_dict["mucen"],
            self.param_dict["zpivot"],
        )
        mean_ncen = 0.5 * (1.0 + erf((logM - logMmin) / self.param_dict["sigma_logM"]))

        return mean_ncen


class Zheng07zdepSats(Zheng07Sats):
    # Params: logM0, logM1, alpha, musat, zpivot
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.param_dict["zpivot"] = 0.5

        self.list_of_haloprops_needed = ["halo_redshift"]

    def mean_occupation(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        mass = kwargs["table"][self.prim_haloprop_key]
        redshift = kwargs["table"]["halo_redshift"]

        logM0 = logM_i(
            redshift,
            self.param_dict["logM0"],
            self.param_dict["musat"],
            self.param_dict["zpivot"],
        )
        logM1 = logM_i(
            redshift,
            self.param_dict["logM1"],
            self.param_dict["musat"],
            self.param_dict["zpivot"],
        )
        M0 = 10.0**logM0
        M1 = 10.0**logM1

        # ~~~ COPIED FROM HALOTOOLS BELOW ~~~
        # Call to np.where raises a harmless RuntimeWarning exception if
        # there are entries of input logM for which mean_nsat = 0
        # Evaluating mean_nsat using the catch_warnings context manager
        # suppresses this warning
        mean_nsat = np.zeros_like(mass)

        idx_nonzero = np.where(mass - M0 > 0)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            mean_nsat[idx_nonzero] = (
                (mass[idx_nonzero] - M0[idx_nonzero]) / M1[idx_nonzero]
            ) ** self.param_dict["alpha"]

        # If a central occupation model was passed to the constructor,
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.modulate_with_cenocc:
            # compatible with AB models
            mean_ncen = getattr(
                self.central_occupation_model,
                "baseline_mean_occupation",
                self.central_occupation_model.mean_occupation,
            )(**kwargs)
            mean_nsat *= mean_ncen

        return mean_nsat


def linear_interp_extrap(x, xp, yp):
    """
    Perform linear interpolation within a given range and linear extrapolation
    outside that range.

    Args:
        :x (array-like): Points to evaluate the interpolation
        :xp (array-like): Known x-values (must be sorted)
        :yp (array-like): Known y-values corresponding to xp

    Returns:
        y (array-like): Interpolated or extrapolated values at x
    """
    x = np.asarray(x)
    xp = np.asarray(xp)
    yp = np.asarray(yp)

    # Compute slopes for interpolation
    slopes = np.diff(yp) / np.diff(xp)

    # Find indices where each x belongs in xp
    indices = np.searchsorted(xp, x) - 1
    indices = np.clip(indices, 0, len(slopes) - 1)  # Clip to valid range

    # Perform interpolation/extrapolation
    y = yp[indices] + slopes[indices] * (x - xp[indices])

    return y


class Zheng07zinterpCens(Zheng07Cens):
    # Params: logMmin, sigma_logM, mucen, zpivot
    def __init__(self, zpivot, **kwargs):
        super().__init__(**kwargs)

        self.zpivot = zpivot
        self.npivot = len(zpivot)

        self.list_of_haloprops_needed = ["halo_redshift"]

    def mean_occupation(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        mass = kwargs["table"][self.prim_haloprop_key]
        redshift = kwargs["table"]["halo_redshift"]
        logM = np.log10(mass)

        yp = [self.param_dict[f"logMmin_z{i}"] for i in range(self.npivot)]
        logMmin = linear_interp_extrap(redshift, self.zpivot, yp)

        mean_ncen = 0.5 * (1.0 + erf((logM - logMmin) / self.param_dict["sigma_logM"]))

        return mean_ncen

class Zheng07zinterpSats(Zheng07Sats):
    # Params: logM0, logM1, alpha, musat, zpivot
    def __init__(self, zpivot, **kwargs):
        super().__init__(**kwargs)

        self.zpivot = zpivot
        self.npivot = len(zpivot)
        self.list_of_haloprops_needed = ["halo_redshift"]

    def mean_occupation(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        mass = kwargs["table"][self.prim_haloprop_key]
        redshift = kwargs["table"]["halo_redshift"]

        yp = [self.param_dict[f"logM0_z{i}"] for i in range(self.npivot)]
        logM0 = linear_interp_extrap(redshift, self.zpivot, yp)

        yp = [self.param_dict[f"logM1_z{i}"] for i in range(self.npivot)]
        logM1 = linear_interp_extrap(redshift, self.zpivot, yp)

        M0 = 10.0**logM0
        M1 = 10.0**logM1

        # ~~~ COPIED FROM HALOTOOLS BELOW ~~~
        # Call to np.where raises a harmless RuntimeWarning exception if
        # there are entries of input logM for which mean_nsat = 0
        # Evaluating mean_nsat using the catch_warnings context manager
        # suppresses this warning
        mean_nsat = np.zeros_like(mass)

        idx_nonzero = np.where(mass - M0 > 0)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            mean_nsat[idx_nonzero] = (
                (mass[idx_nonzero] - M0[idx_nonzero]) / M1[idx_nonzero]
            ) ** self.param_dict["alpha"]

        # If a central occupation model was passed to the constructor,
        # multiply mean_nsat by an overall factor of mean_ncen
        if self.modulate_with_cenocc:
            # compatible with AB models
            mean_ncen = getattr(
                self.central_occupation_model,
                "baseline_mean_occupation",
                self.central_occupation_model.mean_occupation,
            )(**kwargs)
            mean_nsat *= mean_ncen

        return mean_nsat


class AssembiasZheng07zinterpCens(Zheng07zinterpCens, HeavisideAssembias):
    r"""Assembly-biased modulation of `Zheng07zinterpCens`."""
    
    def __init__(self, zpivot, **kwargs):

        Zheng07zinterpCens.__init__(self, zpivot, **kwargs)
        HeavisideAssembias.__init__(
            self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate="mean_occupation",
            **kwargs
        )


class AssembiasZheng07zinterpSats(Zheng07zinterpSats, HeavisideAssembias):
    r"""Assembly-biased modulation of `Zheng07zinterpSats`."""
    
    def __init__(self, zpivot, **kwargs):

        Zheng07zinterpSats.__init__(self, zpivot, **kwargs)
        HeavisideAssembias.__init__(
            self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate="mean_occupation",
            **kwargs
        )


class AssembiasZheng07zdepCens(Zheng07zdepCens, HeavisideAssembias):
    r"""Assembly-biased modulation of `Zheng07zdepCens`."""
    
    def __init__(self, **kwargs):

        Zheng07zdepCens.__init__(self, **kwargs)
        HeavisideAssembias.__init__(
            self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate="mean_occupation",
            **kwargs
        )

class AssembiasZheng07zdepSats(Zheng07zdepSats, HeavisideAssembias):
    r"""Assembly-biased modulation of `Zheng07zdepSats`."""
    
    def __init__(self, **kwargs):

        Zheng07zdepSats.__init__(self, **kwargs)
        HeavisideAssembias.__init__(
            self,
            lower_assembias_bound=self._lower_occupation_bound,
            upper_assembias_bound=self._upper_occupation_bound,
            method_name_to_decorate="mean_occupation",
            **kwargs
        )
