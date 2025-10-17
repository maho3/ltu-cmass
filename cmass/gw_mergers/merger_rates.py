import numpy as np

import scipy.integrate as integrate
from scipy.interpolate import CubicSpline, interp1d # interpolate.interp1d is now consideredlegacy and might be removed!

from astropy import  constants
from .cosmo_utils import dc_interpolator, dVc_over_dz, dt_over_dz, z_to_t

global c_light
c_light = constants.c.to('km/s')

# Star formation Madau-Dicksinson, two versions ;  Psi0 from MD is in solar masses per Mpc cube per year
def Madau_Dickinson(zf, R0, alpha=2.7, beta = 5.6, C = 2.9, Psi0 = 0.015):
    #return Psi0*(1.0+zf)**alpha/(1.0+((1+zf)/C)**beta)
    returnR0*(1.0+zf)**alpha/(1.0+((1+zf)/C)**beta)

def Madau_Dickinson_peak(zf, R0, alpha = 2.7, beta = 5.6, z_peak = 1.9, Psi0 = 0.015):
     C = 1+(1+z_peak)**(-alpha-beta)
     #return Psi0*C*((1+zf)**alpha)/(1+((1+zf)/(1+z_peak))**(alpha+beta))
     return R0*C*((1+zf)**alpha)/(1+((1+zf)/(1+z_peak))**(alpha+beta))

Madau_Dickinson_vec = np.vectorize(Madau_Dickinson)
Madau_Dickinson_peak_vec = np.vectorize(Madau_Dickinson_peak)

# For interpolation methods
dictInterp = {"linear":interp1d, "cubic": CubicSpline}

#%% Class to compute a number N(z) of BBH mergers given a cosmology and merger rate
class redshift_evolution:

    def __init__(self, paramDict, cosmo, nz_interp = 10000):
        
        self.cosmo = cosmo
        self.model = paramDict["model"]
        self.z_max = paramDict["z_max"]
        self.zf_max = paramDict["zf_max"]
        self.dz = paramDict["dz"]
        self.td_min = paramDict['td_min']
        self.tf_min = z_to_t(self.cosmo, self.zf_max).value
        self.d = paramDict['delay_power']
        self.z_plus = paramDict['z_plus']
        self.R0 = paramDict["R0"]
        self.z_min =  paramDict["z_min"] if "z_min" in paramDict else 1e-4
        self.dc_method = dictInterp(paramDict["dc_method"]) if "dc_method" in paramDict else dictInterp["linear"]
        self.t_to_z_method = dictInterp(paramDict["t_to_z_method"]) if "t_to_z_method" in paramDict else dictInterp["linear"]
        self.R_of_z_method = dictInterp(paramDict["R_of_z_method"]) if "R_of_z_method" in paramDict else dictInterp["linear"]
        self.dc_interpolation = dc_interpolator(self.cosmo, self.z_max, nz = nz_interp, interp_func=self.dc_method)

        if paramDict["model"] == "constant":
            
            self.evolution = self.R0 

        elif paramDict["model"]== "PowerLaw":

            self.kappa = paramDict["kappa"] if "kappa" in paramDict else 2.9
            self.evolution = self.R_power()
            #self.evolution = self.R_power(self.z_min,self.z_max)
            
        elif paramDict["model"] == "delay_SFR":
            
            Psi0 = paramDict["Psi0"] if "Psi0" in paramDict else 0.015
            alpha = paramDict["alphaPsi"] if "alphaPsi" in paramDict else 2.7
            beta = paramDict["betaPsi"] if "betaPsi" in paramDict else 5.6
            C = paramDict["CPsi"] if "CPsi" in paramDict else 2.9

            # SFR
            self.Psi = lambda zf : Madau_Dickinson_vec(zf, self.R0, alpha=alpha, beta = beta, C = C, Psi0 = Psi0)
            self.evolution = self.time_delay_interpolation()

        elif paramDict["model"] == "delay_SFR_peak":
            
            Psi0 = paramDict["Psi0"] if "Psi0" in paramDict else 0.015
            alpha = paramDict["alphaPsi"] if "alphaPsi" in paramDict else 2.7
            beta = paramDict["betaPsi"] if "betaPsi" in paramDict else 5.6
            z_peak = paramDict["z_peak"] if "z_peak" in paramDict else 1.9

            # SFR
            self.Psi = lambda zf : Madau_Dickinson_peak_vec(zf, self.R0, alpha=alpha, beta = beta, z_peak =z_peak, Psi0 = Psi0)
            #Equation 2 in https://arxiv.org/pdf/2003.12152.pdf
            self.evolution = self.time_delay_interpolation() # CAREFUL: interpolator instance
            
        else:
             raise ValueError("Redshift-dependent evolution model not available")

    # Equations E16 to E18 of https://arxiv.org/pdf/2010.14533
    def R_power(self, z_nbins = 100):
        
        zm = np.linspace(0.,self.zf_max, z_nbins)
        R = self.R0 * np.power((1.0 + zm), self.kappa)
        R_of_z = self.R_of_z_method(zm, R)
        return R_of_z

    # Equation 4 evaluated at redshit z (https://arxiv.org/pdf/2210.05724.pdf)
    def time_delay_interpolation(self):
        '''
        Returns an interpolator instance
        '''

        zm = np.linspace(0.,self.zf_max,1000)
        tm = z_to_t(self.cosmo, zm).value
        
        self.t_to_z = self.t_to_z_method(z_to_t(self.cosmo, zm).value, zm)
        self.tf_min = z_to_t(self.cosmo, self.zf_max).value

        zf_low, zf_up = self.redshifts_limits_delay(zm,tm)
        
        idx = np.where((zf_up <= self.zf_max)&(zf_low<zf_up))[0]
        R = np.zeros((len(zm),),dtype = float)
        for i in idx:
            R[i] = self.R_delay(zf_low[i],zf_up[i],tm[i])
        zf_min_0, zf_max_0 = self.redshifts_limits_delay(np.array([0.]), np.array([z_to_t(self.cosmo,0.).value]))

        # Actual normalization happens here, equation 4 in https://arxiv.org/pdf/2210.05724.pdf
        R_0 = self.R_delay(zf_min_0[0], zf_max_0[0], z_to_t(self.cosmo, 0.).value)
        R_of_z = self.R_of_z_method(zm, self.R0*R/R_0)
        print("Value at z=0 is %.2f, and desired R0 is %.2f"%(Rz(0.), self.R0))
        return R_of_z # scipy interpolator instance
    
    # Numerator of equation 4 (https://arxiv.org/pdf/2210.05724.pdf)
    def R_delay(self,zf_min, zf_max,tm):

        # Number of integrand bins
        N = int((zf_max-zf_min)/self.dz)
        
        if N>1:
            zs = np.linspace(zf_min,zf_max,N).reshape(-1)
            tf = z_to_t(self.cosmo, zs).value.reshape(-1)
            R_SFR = self.Psi(zs).reshape(-1)
            P_t = self.P_t(tm,tf).reshape(-1)
            dt_dz = dt_over_dz(self.cosmo, zs).reshape(-1)
            return integrate.simpson(P_t*R_SFR*dt_dz, x = zs) # simps in earlier scipy versions
        else:
            return 0.

    # Default time-delay probability as in GWSim paper, Karathanasis et al.
    def P_t(self,tm,tf):
        return (tm - tf)**(-self.d)

    # Redhsifts limits when using time-delay based merger rates.
    def redshifts_limits_delay(self,zm,tm):
        
        tf_max = tm - self.td_min # array of earlier formation times given array orf merger times tm(zm)
    
        idx = np.where(tf_max>self.tf_min)[0]
        zf_low, zf_up = np.zeros(len(zm)),np.zeros(len(zm))
        zf_low[idx] = self.t_to_z(tf_max[idx]) # interpolator instance from time_delay_interpolation
        zf_up[idx] = zm[idx] + self.z_plus
    
        idx = np.where(zf_up < zf_low)[0]
        zf_low[idx], zf_up[idx] = 0, 0

        return zf_low, zf_up 
    
    # Evaluates left member of equation 6 (https://arxiv.org/pdf/2210.05724.pdf)
    def MergersPerYearPerGpc3(self,z, fSky = 1.0):
        return int(self.evolution(z)*dVc_over_dz(self.cosmo, z,fSky=fSky).value/(1.0 + z))

    # Actually expected N(z) in given redshift bin, given sky coverage
    def number_of_events(self, z,delta_z, t_obs, fSky = 1.0):
        return self.MergersPerYearPerGpc3(self, z, fSky)*delta_z*t_obs

    def N_of_z(self,z_start, z_stop, delta_z, t_obs, fSky = 1.0):
        
        z_arr = np.arange(start = z_start + delta_z/2, stop  = z_stop - delta_z/2, step = delta_z)
        
        N = len(z_arr)
        Nz = np.zeros((N,), dtype = int)
        for n in range(N):
            Nz[n] = self.MergersPerYearPerGpc3(z_arr[n], fSky = 1.0)*delta_z*t_obs
        return z_arr, Nz