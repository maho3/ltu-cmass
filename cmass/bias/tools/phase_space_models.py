import numpy as np
import scipy.special
from halotools.empirical_models import NFWPhaseSpace, BiasedNFWPhaseSpace
from halotools.empirical_models.phase_space_models import MonteCarloGalProf
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models.phase_space_models.analytic_models.satellites.nfw.kernels.mass_profile import _g_integral

def unbiased_dimless_vrad_disp_kernel(scaled_radius, conc):
    r"""
    Analytical solution to the isotropic jeans equation for an NFW potential,
    rendered dimensionless via scaling by the virial velocity.

    :math:`\tilde{\sigma}^{2}_{r}(\tilde{r})\equiv\sigma^{2}_{r}(\tilde{r})/V_{\rm vir}^{2} = \frac{c^{2}\tilde{r}(1 + c\tilde{r})^{2}}{g(c)}\int_{c\tilde{r}}^{\infty}{\rm d}y\frac{g(y)}{y^{3}(1 + y)^{2}}`
    
    Halotools solves this numerically despite there being an analytic solution, hence
    this function uses that analytic result, which dramatically speeds up the calculation.

    See :ref:`nfw_jeans_velocity_profile_derivations` for derivations and implementation details.

    Parameters
    -----------
    scaled_radius : array_like
        Length-Ngals numpy array storing the halo-centric distance
        *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
        :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`.

    conc : float
        Concentration of the halo.

    Returns
    -------
    result : array_like
        Radial velocity dispersion profile scaled by the virial velocity.
        The returned result has the same dimension as the input ``scaled_radius``.
    """
    
    x = np.atleast_1d(scaled_radius).astype(np.float64)
    
    a = conc * x
    prefactor = conc * (conc * x) * (1.0 + conc * x) ** 2 / _g_integral(conc)
    
    result = 0.5 * (np.pi ** 2 + 6 * scipy.special.spence(1+a)
                    + 3 * np.log(1 + a) ** 2
                    - (7*a**2 + 9*a + 1)/a/(1+a)**2
                    - np.log(a)
                    - np.log(1+a)/(1+a) * (-a + 5 + 3/a - 1/a**2)
                   )
        
    result = np.sqrt(result * prefactor)
    
    return result


def biased_dimless_vrad_disp(scaled_radius, halo_conc, gal_conc):
    r"""
    Analytical solution to the isotropic jeans equation for an NFW potential,
    rendered dimensionless via scaling by the virial velocity.

    :math:`\tilde{\sigma}^{2}_{r}(\tilde{r})\equiv\sigma^{2}_{r}(\tilde{r})/V_{\rm vir}^{2} = \frac{c^{2}\tilde{r}(1 + c\tilde{r})^{2}}{g(c)}\int_{c\tilde{r}}^{\infty}{\rm d}y\frac{g(y)}{y^{3}(1 + y)^{2}}`
    
    Halotools solves this numerically despite there being an analytic solution, hence
    this function uses that analytic result, which dramatically speeds up the calculation.

    See :ref:`nfw_jeans_velocity_profile_derivations` for derivations and implementation details.

    Parameters
    -----------
    scaled_radius : array_like
        Length-Ngals numpy array storing the halo-centric distance
        *r* scaled by the halo boundary :math:`R_{\Delta}`, so that
        :math:`0 <= \tilde{r} \equiv r/R_{\Delta} <= 1`.

    halo_conc : float
        Concentration of the halo.
        
    gal_conc : float
        Concentration of the galaxy.

    Returns
    -------
    result : array_like
        Radial velocity dispersion profile scaled by the virial velocity.
        The returned result has the same dimension as the input ``scaled_radius``.
    """
    
    b = halo_conc / gal_conc
    
    if b == 1:
        return unbiased_dimless_vrad_disp_kernel(scaled_radius, halo_conc)
    
    x = np.atleast_1d(scaled_radius).astype(np.float64)
    a = gal_conc * x
    prefactor = (
        gal_conc * gal_conc * x * (1.0 + gal_conc * x) ** 2 / _g_integral(halo_conc)
    )
    
    if b > 1:
        result = (
            a*(-1 + b)*(-(b*(-1 + a*(-3 + b) + b)) + 2*a*(1 + a)*(-1 + \
            b)*np.pi**2) - a**2 *(1 + a)*(-1 + b)**2*b**2*np.log(a) \
            - a**2 *(1 + a)*(2*b*(-3 + 4*b)*np.log(1 + a) - 3*(-1 + b)**2*np.log(-1 + b)**2 
                             + b*(-6 + b*(9 + (-2 + b)*b))*np.log(b)) + \
            np.log(1 + a*b) + (-3*a*(-1 + b)**2 + (-2 + b)*b 
                               + a**2*(-6 + b*(1 + b)*(6 + (-3 + b)*b)) + a**3*b*(-6 + b*(9 + (-2 + b)*b)) + \
            6*a**2*(1 + a)*(-1 + b)**2*np.log(((1 + a)*b)/(-1 + b)))*np.log(1 \
            + a*b) + 6*a**2*(1 + a)*(-1 + b)**2*(scipy.special.spence(1+(a*b)) + \
            scipy.special.spence(1+(1 + a*b)/(b-1))))/(2.* a**2 * (1 + a)*(-1 + b)**2)
    else:
       
        # Need Re(Li2((1+ab)/(1-b))). Use the identity
        # Li2(z) + Li2(1/z) = pi^2/3 - log(z)^2/2 - i * pi * log(z)
        # and the fact that Li2(z) is real for z real and < 1.
        # Hence Re(Li2(z)) = pi^2/3 - log(z)^2/2 -  Li2(1/z)
        # for z real and > 1.
        re_li2 = np.pi**2/3 - np.log((1 + a*b)/(1 - b))**2/2 - \
            scipy.special.spence(1 - (1 - b)/(1 + a*b))
        
        result = (a*(-1 + b)*(-(b*(-1 + a*(-3 + b) + b)) - a*(1 + a)*(-1 + \
            b)*np.pi**2) - a**2*(1 + a)*(-1 + b)**2*b**2*np.log(a) \
            - a**2*(1 + a)*(2*b*(-3 + 4*b)*np.log(1 + a) - 3*(-1 + b)**2*np.log(1 - b)**2 
                            + b*(-6 + b*(9 + (-2 + b)*b))*np.log(b)) + \
            np.log(1 + a*b) + (-3*a*(-1 + b)**2 + (-2 + b)*b + a**2*(-6 + \
            b*(1 + b)*(6 + (-3 + b)*b)) + a**3*b*(-6 + b*(9 + (-2 + b)*b)) + \
            6*a**2*(1 + a)*(-1 + b)**2*np.log(-(((1 + a)*b)/(-1 + \
            b))))*np.log(1 + a*b) + 6*a**2*(1 + a)*(-1 + b)**2*scipy.special.spence(1+a*b) 
            + 6*a**2*(1 + a)*(-1 + b)**2*re_li2)/(2.* a**2 * (1 + a)*(-1 + b)**2)
        
    result = np.sqrt(result * prefactor)
    
    return result


class Centrals_vBiasedNFWPhaseSpace(NFWPhaseSpace):
    """
    Model for the phase space distribution of mass and/or galaxies
    in isotropic Jeans equilibrium in an NFW halo profile,
    based on Navarro, Frenk and White (1995),
    where the concentration of the galaxies is the same
    as the concentration of the parent halo
    
    We allow a velocity bias, where the central galaxy's velocity
    is drawn from a Gaussian of width equal to that from solving the 
    Jeans' equation, but multiplied by eta_vb_centrals
    """
    
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.param_dict = {'eta_vb_centrals': 0.0}
        
    def set_parameters(self, new_parameters):
        """
        Set all parameters using a dict
        """
        for p in self.param_dict.keys():
            self.param_dict[p] = new_parameters[p]
        
    def assign_phase_space(self, table, seed=None):
        
        # Place centrals at the centre of the halo
        phase_space_keys = ["x", "y", "z"]
        for key in phase_space_keys:
            table[key][:] = table["halo_" + key][:]

        # Sample velocities
        if seed is not None:
            seed += 1
        MonteCarloGalProf.mc_vel(self, table, seed=seed)
        
    def mc_radial_velocity(self, scaled_radius, total_mass, *profile_params, **kwargs):
        
        virial_velocities = self.virial_velocity(total_mass)
        radial_dispersions = virial_velocities * self.param_dict['eta_vb_centrals'] 

        seed = kwargs.get('seed', None)
        with NumpyRNGContext(seed):
            radial_velocities = np.random.normal(scale=radial_dispersions)

        return radial_velocities
    
    def dimensionless_radial_velocity_dispersion(self, scaled_radius, *conc):
        return unbiased_dimless_vrad_disp_kernel(scaled_radius, *conc)
    
    
class Satellites_vBiasedNFWPhaseSpace(BiasedNFWPhaseSpace): 
    """
    Model for the phase space distribution of galaxies
    in isotropic Jeans equilibrium in an NFW halo profile,
    based on Navarro, Frenk and White (1995),
    where the concentration of the tracers is permitted to differ from the
    host halo concentration.
    
    We also allow a velocity bias, akin to that in Centrals_vBiasedNFWPhaseSpace
    """
    
    def __init__(self, conc_key='halo_nfw_con', **kwargs): 
        self.conc_key = conc_key
        super().__init__(**kwargs)
        
    def _initialize_conc_bias_param_dict(self, **kwargs):
        if 'conc_gal_bias_logM_abscissa' in list(kwargs.keys()):
            raise NotImplementedError
        else:
            return {'conc_gal_bias_satellites': 1., 'eta_vb_satellites': 1.}
        
    def set_parameters(self, new_parameters):
        """
        Set all parameters using a dict
        """
        for p in self.param_dict.keys():
            self.param_dict[p] = new_parameters[p]
        
    def assign_phase_space(self, table, seed=None):
        MonteCarloGalProf.mc_pos(self, table=table, seed=seed)
        if seed is not None:
            seed += 1
        self.mc_vel(table, seed=seed)
        
        
    def mc_vel(self, table, seed=None):
        vx, vy, vz = MonteCarloGalProf.mc_vel(self, table, seed=seed, 
                                              overwrite_table_velocities=False, 
                                              return_velocities=True)
        if vx is None: 
            return 
        
        # scale velocity by satellite velocity bias
        table['vx'][:] += vx * self.param_dict['eta_vb_satellites']
        table['vy'][:] += vy * self.param_dict['eta_vb_satellites']
        table['vz'][:] += vz * self.param_dict['eta_vb_satellites']
        
        
    def calculate_conc_gal_bias(self, seed=None, **kwargs):
        """
        Calculate the ratio of the galaxy concentration to the halo concentration,
        """
        
        if 'table' in list(kwargs.keys()):
            table = kwargs['table']
            mass = table[self.prim_haloprop_key]
        elif 'prim_haloprop' in list(kwargs.keys()):
            mass = np.atleast_1d(kwargs['prim_haloprop']).astype('f4')
        else:
            msg = ("\nYou must pass either a ``table`` or ``prim_haloprop`` argument \n"
                "to the ``assign_conc_gal_bias`` function of the ``BiasedNFWPhaseSpace`` class.\n")
            raise KeyError(msg)

        result = np.full_like(mass, self.param_dict['conc_gal_bias_satellites'])

        if 'table' in list(kwargs.keys()):
            table['conc_gal_bias'][:] = result
            halo_conc = table[self.conc_key]
            gal_conc = self._clipped_galaxy_concentration(halo_conc, result)
            table['conc_galaxy'][:] = gal_conc
        else:
            return result
        
    def dimensionless_radial_velocity_dispersion(
        self, scaled_radius, halo_conc, conc_gal_bias
    ):
        gal_conc = self._clipped_galaxy_concentration(halo_conc, conc_gal_bias)
        return biased_dimless_vrad_disp(
            scaled_radius,
            halo_conc,
            gal_conc,
        )
    
    
        