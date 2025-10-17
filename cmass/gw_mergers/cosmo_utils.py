import numpy as np
from astropy import cosmology, units, constants
from scipy.interpolate import CubicSpline, interp1d

#%%

global c_light
c_light = constants.c.to('km/s')

#%%

def dc_interpolator(myCosmo, z_max=100.0, nz=10000, interp_func = interp1d):
    myz = np.linspace(0, z_max, nz)
    dcArr = myCosmo.comoving_distance(myz).value # .value to pass Quantity to interpolate module
    interp_dc = interp_func(myz, dcArr)
    
    return interp_dc

def dVc_over_dz(myCosmo, z, fSky = 1.0):
    
    dV = myCosmo.differential_comoving_volume(z) # per redshift per steradian
    dVdOmega = dV*fSky*4*np.pi*units.sr
    # astropy has not units Gpc3 yet, but we can "cube" the Gpc unit manually
    dVdOmega = (dVdOmega.value) * (1e-3 * units.Gpc)**3
    return dVdOmega

def z_to_t(cosmo, z):
    return cosmo.age(z).to(units.Gyr)

def dt_over_dz(myCosmo, z):

    result = 1.0/(myCosmo.H0*myCosmo.efunc(z)*(1+z))
    # result above has time unit (Mpc s)/km
    return result.to(units.Gyr)
