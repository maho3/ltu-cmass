import numpy as np
from abc import ABC, abstractmethod
from .probabilities import PowerLawGaussian_math, PowerLaw_math, SmoothedProb, S_factor
from scipy.interpolate import interp1d

#%%%
class mass_distribution(ABC):
 
    @abstractmethod
    def draw_samples(self) -> (np.ndarray, np.ndarray):
        '''
         Returns samples for (m1,q) or (m1, m2) for binaries
         where q = m1/m2 >= 1

        Returns:
            : length of dataset
        '''
        print("Not implemented for this class yet")

# Implementation from GWSim
class masses_TP(mass_distribution):

    def __init__(self, params_dict):

        alpha = params_dict["alpha"]
        beta = params_dict["beta"]
        mmin = params_dict['mmin']
        mmax = params_dict['mmax']
    
        self.P_m1 = PowerLaw_math(alpha=-alpha,min_pl=mmin,max_pl=mmax)
        self.P_m2 = PowerLaw_math(alpha=beta, min_pl=mmin, max_pl=mmax)
        self.mmin = mmin
        self.mmax = mmax

    def joint_prob(self, m1, m2):
        """
        This method returns the joint probability :p(m_1,m_2), with masses in solar masses

        """

        P_joint =self.P_m1.prob(m1)*self.P_m2.conditioned_prob(m2,self.mmin*np.ones_like(m1),ms)

        return P_joint

    def draw_samples(self, Nsample, seed = 94):
        """
        This method samples from the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        input: vector of solar masses
        return: m1_samples, m2_sample
        """
        np.random.seed(seed)
        vals_m1 = np.random.rand(Nsample)
        vals_m2 = np.random.rand(Nsample)

        m1_trials = np.logspace(np.log10(self.P_m1.minimum),np.log10(self.P_m1.maximum),10000)
        m2_trials = np.logspace(np.log10(self.P_m2.minimum),np.log10(self.P_m2.maximum),10000)

        cdf_m1_trials = self.P_m1.cdf(m1_trials)
        cdf_m2_trials = self.P_m2.cdf(m2_trials)

        m1_trials = np.log10(m1_trials)
        m2_trials = np.log10(m2_trials)

        _,indxm1 = np.unique(cdf_m1_trials,return_index=True)
        _,indxm2 = np.unique(cdf_m2_trials,return_index=True)

        interpo_icdf_m1 = interp1d(cdf_m1_trials[indxm1],m1_trials[indxm1],bounds_error=False,fill_value=(m1_trials[0],m1_trials[-1]))
        interpo_icdf_m2 = interp1d(cdf_m2_trials[indxm2],m2_trials[indxm2],bounds_error=False,fill_value=(m2_trials[0],m2_trials[-1]))

        mass_1_samples = 10**interpo_icdf_m1(vals_m1)
        mass_2_samples = 10**interpo_icdf_m2(vals_m2*self.P_m2.cdf(mass_1_samples))

        return mass_1_samples, mass_2_samples

    def CheckSmoothing(self,mmin,mmax,delta_m,m1pr,m2pr):
        """
        This function checks the values of the smoothing function in the mass interval
        of interest. If the max is nan or 0, we skip the smoothing and return a 
        dictionnary of m1, m2 pdf without smoothing
        """
        xmass = np.linspace(mmin,mmax,100)
        sfactor = S_factor(xmass,mmin,delta_m)
        if not np.isnan(np.max(sfactor)) and (np.max(sfactor)>0):
            P_dict = {'mass_1': SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                    'mass_2': SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}
        else:
            P_dict = {'mass_1': m1pr,'mass_2': m2pr}
        return P_dict["mass_1"], P_dict["mass_2"]


# Fiducial PowerLaw+Peak (PP) from GWTC-2/3/4 LVK population paper
class masses_PP(mass_distribution):

    def __init__(self, params_dict):

        alpha = params_dict["alpha"]
        beta = params_dict["beta"] 
        # 12/11/2024 the constraints on Beta (VI-A of  https://arxiv.org/pdf/2111.03634) are on q^{beta}, not m_2^{beta}!!
        mmin = params_dict['mmin']
        mmax = params_dict['mmax']

        mu_g = params_dict['mu_g']
        sigma_g = params_dict['sigma_g']
        lambda_g = params_dict['lambda_g']

        delta_m = params_dict['delta_m']

        # Careful, alpha is passed as -alpha in our convention
        P_m1 = PowerLawGaussian_math(alpha=-alpha,min_pl=mmin,max_pl=mmax,lambda_g=lambda_g,
                                          mean_g=mu_g,sigma_g=sigma_g,
                                          min_g=mmin,max_g=mu_g+5*sigma_g)


        P_m2 = PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=np.max([mu_g+5*sigma_g,mmax]))

        P_m1_sm, P_m2_sm = self.CheckSmoothing(mmin, mmax, delta_m ,P_m1,P_m2)
        self.P_m1 = P_m1_sm
        self.P_m2 = P_m2_sm
        self.mmin = mmin
        self.mmax = mmax

    def joint_prob(self, ms1, ms2):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        ms1: np.array(matrix)
            mass one in solar masses
        ms2: dict
            mass two in solar masses
        """

        to_ret =self.P_m1.prob(ms1)*self.P_m2.conditioned_prob(ms2,self.mmin*np.ones_like(ms1),ms1)

        return to_ret

    def draw_samples(self, n_s,seed = 94, n_cdf = 10000):
        """
        This method samples from the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        input: vector of solar masses
        return: m1_samples, m2_samples
        """
        np.random.seed(seed)
        U_1 = np.random.uniform(low = 0., high = 1., size = n_s)
        U_2 = np.random.uniform(low = 0., high = 1., size = n_s)

        m1_arr = np.logspace(np.log10(self.P_m1.minimum),np.log10(self.P_m1.maximum),n_cdf)
        m2_arr = np.logspace(np.log10(self.P_m2.minimum),np.log10(self.P_m2.maximum),n_cdf)

        cdf_m1_arr = self.P_m1.cdf(m1_arr)
        cdf_m2_arr = self.P_m2.cdf(m2_arr)

        _,indxm1 = np.unique(cdf_m1_arr,return_index=True)
        _,indxm2 = np.unique(cdf_m2_arr,return_index=True)

        interpo_icdf_m1 = interp1d(cdf_m1_arr[indxm1],m1_arr[indxm1],bounds_error=False,fill_value=(m1_arr[0],m1_arr[-1]))
        interpo_icdf_m2 = interp1d(cdf_m2_arr[indxm2],m2_arr[indxm2],bounds_error=False,fill_value=(m2_arr[0],m2_arr[-1]))

        mass_1_samples = interpo_icdf_m1(U_1)

        # We must have m_2 < m_1 with our convention on mass ratio
        mass_2_samples = interpo_icdf_m2(U_2*self.P_m2.cdf(mass_1_samples))

        return mass_1_samples, mass_2_samples


    def CheckSmoothing(self,mmin,mmax,delta_m,m1pr,m2pr):
        """
        This function checks the values of the smoothing function in the mass interval
        of interest. If the max is nan or 0, we skip the smoothing and return a 
        dictionnary of m1, m2 pdf without smooting
        """
        xmass = np.linspace(mmin,mmax,100)
        sfactor = S_factor(xmass,mmin,delta_m)
        if not np.isnan(np.max(sfactor)) and (np.max(sfactor)>0):
            dist = {'mass_1': SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                    'mass_2': SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}
        else:
            dist = {'mass_1': m1pr,
                    'mass_2': m2pr}
        return dist["mass_1"], dist["mass_2"]


class mass_truncated_pareto(mass_distribution):
    '''
    Truncated Pareto distribution for the primary mass m_1
    
    The conditional distribution of m_2|m_1 is also a truncated Pareto distribution
    upper-bounded by m1 for each draw
    
    - On the truncated Pareto distribution with applications, Lorenzo Zaninetti1∗, Mario Ferraro2†
    - Parameter Estimation for the Truncated Pareto Distribution Inmaculada B. A BAN , Mark M. M EERSCHAERT , and Anna K. PANORSKA
    '''
    def __init__(self, a, b, c1,c2):
        self.a = a # corresponds to m_min
        self.b = b # corresponds to m_max (only applies for m1)
        self.c1 = c1 # exponential decay P(m1)
        self.c2 = c2 # exponential decay P(m2|m1)
        
        if self.a <= 0. or self.b <= self.a  or self.c1 <= 0. or self.c2 <= 0.:
            raise Exception(
                "Truncated Pareto distribution must have a,b,c > 0 with b>a")
    
    def draw_samples(self, N, seed = 49):
        
        # m1 masses
        np.random.seed(seed)
        U1 = np.random.uniform(0. ,1. ,N)
        base1 = 1.0 - U1 *(1. - np.power(self.a/self.b, self.c1))
        m1_samples = self.a * np.power(base1, -1.0/self.c1)
        
        # m2 masses
        np.random.seed(seed+5)
        U2 = np.random.uniform(0. ,1. ,N)
        base2 = 1.0 - U2 *(1. - np.power(self.a/m1_samples, self.c2)) # each draw bounded by corresponding m1
        m2_samples = self.a * np.power(base2, -1.0/self.c2)
        
        
        return m1_samples, m2_samples