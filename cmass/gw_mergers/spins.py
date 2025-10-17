import numpy as np
from scipy.stats import truncnorm
from abc import ABC, abstractmethod

#%%%

# For some of the code below, see 2210.05724 section 2.4 for explanations and equations
class spin_distribution(ABC):
 
    @abstractmethod
    def draw_samples(self) -> (np.ndarray, np.ndarray):
        '''
         Returns samples for (m1,q) or (m1, m2) for binaries
         where q = m1/m2 >= 1

        Returns:
            : length of dataset
        '''
        print("Not implemented for this class yet")
    
    
class zero_spins(spin_distribution):
    '''
    Case for all spin components of the binary to be zero
    '''  
        
    def __init__(self):
        print("Null spins will be considered for BBH mergers")
    
    def draw_samples(self, N, seed): # seed is just there for convenience in writing BBH_mock.py for now
        return np.zeros((N,), dtype = float), np.zeros((N,), dtype = float)


class aligned_spins_old(spin_distribution):
    '''
    Case for aligned spins
    theta1 = theta2 = 0
    or theta1 = theta2 = pi
    '''  
        
    def __init__(self, model_str):
        print("Aligned spins will be considered for BBH mergers")
        self.chieff_name = model_str
        #self.q = m2_s/m1_s
        if self.chieff_name == "gaussian":
            self.mu_eff = mu_eff
            self.std_eff = std_eff
            self.clip_l = clip_l
            self.clip_u = clip_u
    
    def draw_samples(self, N, seed, q_arr):
        '''

        Parameters
        ----------
        q_arr : Careful, the convention here is q <=1 = m2/m1
        seed : TYPE, optional
            DESCRIPTION. The default is 48.

        '''
        chi1 = np.zeros((N,))
        chi2 = np.zeros((N,))
        tilt = np.zeros((N,)) # theta1 = theta2 for aligned-spins case
        np.random.seed(seed)
        
        # Draw values for the effective spin parameter
        if self.chieff_name == "uniform":
            np.random.seed(seed)
            chieff = np.random.uniform(low = -1.0, high = 1.0, size = N)
        elif self.chieff_name == "gaussian":
            low, up = (self.clip_l - self.mu_eff) / self.std_eff, (self.clip_u - self.mu_eff) / self.std_eff
            chieff = truncnorm.rvs(low, up, loc = self.mu_eff, 
                                   scale = self.std_eff, random_state = seed, size = N)
            
        tilt[np.where(chieff <= 0.0)] = np.pi
        tilt[np.where(chieff > 0.0)] = 0.0
              
        for n in range(N):
            a = max([0.0,((1. + 1./q_arr[n])*np.abs(chieff[n])-1.)*q_arr[n]])
            b = min([1.0,((1. + 1./q_arr[n])*np.abs(chieff[n])+1.)*q_arr[n]])
            chi2[n] = np.random.uniform(low = a, high = b)
            chi1[n] = (1. + 1./q_arr[n]) * np.abs(chieff[n]) - 1./q_arr[n] * chi2[n]
        return chi1, chi2, tilt, tilt
    
class aligned_spins(spin_distribution):
    '''
    Case for aligned spins
    theta1 = theta2 = 0
    or theta1 = theta2 = pi
    '''  
        
    def __init__(self, chieff = "uniform", mu_eff = 0.0, std_eff = 1.0, clip_l = 1.0, clip_u = 1.0):
        print("Aligned spins will be considered for BBH mergers")
        self.chieff_name = chieff
        if self.chieff_name == "gaussian":
            self.mu_eff = mu_eff
            self.std_eff = std_eff
            #self.clip_l = clip_l
            #self.clip_u = clip_u
    
    def draw_samples(self, N, seed, q_arr):
        '''

        Parameters
        ----------
        q_arr : Careful, the convention here is q <=1 = m2/m1
        seed : TYPE, optional
            DESCRIPTION. The default is 48.

        '''
        chi1 = np.zeros((N,))
        chi2 = np.zeros((N,))
        tilt1 = np.zeros((N,)) # theta1 = theta2 for aligned-spins case
        tilt2 = np.zeros((N,))
        
        # Draw values for the effective spin parameter
        if self.chieff_name == "uniform":
            np.random.seed(seed)
            chieff = np.random.uniform(low = -1.0, high = 1.0, size = N)
        elif self.chieff_name == "gaussian":
            np.random.normal(seed)
            chieff = np.random.normal(self.mu_eff, self.std_eff, N)
            # low, up = (self.clip_l - self.mu_eff) / self.std_eff, (self.clip_u - self.mu_eff) / self.std_eff
            # chieff = truncnorm.rvs(low, up, loc = self.mu_eff, 
            #                        scale = self.std_eff, random_state = seed, size = N)

        # Chi eff to chi
        inds = np.where(chieff < 0.0)[0] # case theta = pi
        if len(inds) != 0:
            tilt1[inds] = np.pi + 0*chieff[inds]
            tilt2[inds] = np.pi + 0*chieff[inds]

        qchi = (1+q_arr)*np.abs(chieff) # is in [0;1+q], use the absolute value, the sign will be from the cos\theta_i
        chi2 = [np.random.uniform(np.max([0,(qchi[i]-1)/q_arr[i]]),np.min([1,(1+qchi[i])/q_arr[i]]),1)[0] for i in range(N)]
        chi1 = qchi-q_arr*chi2 # is in [0;1]
        return np.array(chi1), np.array(chi2), np.array(tilt1), np.array(tilt2)
              