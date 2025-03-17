import warnings
import numpy as np
from scipy.special import erf
from halotools.empirical_models import Zheng07Cens, Zheng07Sats


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

        self.param_dict['zpivot'] = 0.5

        self.list_of_haloprops_needed = ['halo_redshift']

    def mean_occupation(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        mass = kwargs["table"][self.prim_haloprop_key]
        redshift = kwargs["table"]["halo_redshift"]
        logM = np.log10(mass)

        logMmin = logM_i(
            redshift, self.param_dict["logMmin"], self.param_dict["mucen"],
            self.param_dict["zpivot"]
        )
        mean_ncen = 0.5 * (
            1.0
            + erf((logM - logMmin) / self.param_dict["sigma_logM"])
        )

        return mean_ncen


class Zheng07zdepSats(Zheng07Sats):
    # Params: logM0, logM1, alpha, musat, zpivot
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.param_dict['zpivot'] = 0.5

        self.list_of_haloprops_needed = ['halo_redshift']

    def mean_occupation(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        mass = kwargs["table"][self.prim_haloprop_key]
        redshift = kwargs["table"]["halo_redshift"]

        logM0 = logM_i(
            redshift, self.param_dict["logM0"], self.param_dict["musat"],
            self.param_dict["zpivot"]
        )
        logM1 = logM_i(
            redshift, self.param_dict["logM1"], self.param_dict["musat"],
            self.param_dict["zpivot"]
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

        self.list_of_haloprops_needed = ['halo_redshift']

    def mean_occupation(self, **kwargs):
        # Retrieve the array storing the mass-like variable
        mass = kwargs["table"][self.prim_haloprop_key]
        redshift = kwargs["table"]["halo_redshift"]
        logM = np.log10(mass)

        yp = [self.param_dict[f"logMmin_z{i}"] for i in range(self.npivot)]
        logMmin = linear_interp_extrap(redshift, self.zpivot, yp)
        
        mean_ncen = 0.5 * (
            1.0
            + erf((logM - logMmin) / self.param_dict["sigma_logM"])
        )

        return mean_ncen
    
    
class Zheng07zinterpSats(Zheng07Sats):
    # Params: logM0, logM1, alpha, musat, zpivot
    def __init__(self, zpivot, **kwargs):
        super().__init__(**kwargs)

        self.zpivot = zpivot
        self.npivot = len(zpivot)

        self.list_of_haloprops_needed = ['halo_redshift']

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
    