import numpy as np
from halotools.empirical_models import NFWPhaseSpace, BiasedNFWPhaseSpace
from halotools.empirical_models.phase_space_models import MonteCarloGalProf
from astropy.utils.misc import NumpyRNGContext

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
    
    
        