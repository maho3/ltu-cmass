# ACKNOWLEDGEMENTS: some of this code started from the "random" module of GWSim at https://git.ligo.org/benoit.revenu/gwsim

import numpy as _np
from scipy.stats import truncnorm as _truncnorm
import copy as _copy
from scipy.interpolate import interp1d as _interp1d
from scipy.special import erf as _erf
from scipy.special import logsumexp as _logsumexp
from scipy.integrate import cumulative_trapezoid as _cumtrapz

# See "On the truncated Pareto distribution with applications", L. Zaninetti and M. Ferraro
def truncatedPareto_sampling(a,b,c,N, seed = 49):
    
    _np.random.seed(seed)
    U = _np.random.uniform(0. ,1. ,N)
    base = 1.0 - U *(1. - _np.power(a/b, c))
    return a * _np.power(base, -1.0/c)

def Inverse_Cumulative_Sampling_PDF(myPDF, x, N, seed = 49):

    cdf = _np.cumsum(myPDF(x))
    cdf /= _np.max(cdf)
    cdf[0] = 0 # start from 0
    diff = _np.diff(cdf)
    if not _np.all(diff>0.):
        raise ValueError("The CDF is not strictly increasing, distribution has not unique inverse")
    p = _np.linspace(0.0,1.0,len(cdf))
    icdf = _interp1d(cdf,p)
    _np.random.seed(seed)
    return icdf(_np.random.uniform(0,1,N))

def Inverse_Cumulative_Sampling_CDF(myCDF, x, N, seed = 49):

    cdf = myCDF(x)
    cdf /= _np.max(cdf)
    cdf[0] = 0 # start from 0
    diff = _np.diff(cdf)
    if not _np.all(diff > 0.):
        raise ValueError("The CDF is not strictly increasing, distribution has not unique inverse")
    p = _np.linspace(0.0, 1.0, len(cdf))
    icdf = _interp1d(cdf, p)
    _np.random.seed(seed)
    return icdf(_np.random.uniform(0,1,N))

## FROM GWSIM/random/custom_math_priors

# power law normalization
def get_PL_norm(alpha,minv,maxv):
    '''
    This function returns the powerlaw normalization factor

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    '''

    # Get the PL norm as in Eq. 24 on the tex document
    if alpha == -1:
        return _np.log(maxv/minv)
    else:
        return (_np.power(maxv,alpha+1) - _np.power(minv,alpha+1))/(alpha+1)

def get_gaussian_norm(mu,sigma,min,max):
    '''
    This function returns the gaussian normalization factor

    Parameters
    ----------
    mu: float
        mean of the gaussian
    sigma: float
        standard deviation of the gaussian
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    '''

    # Get the gaussian norm as in Eq. 28 on the tex document
    max_point = (max-mu)/(sigma*_np.sqrt(2.))
    min_point = (min-mu)/(sigma*_np.sqrt(2.))
    return 0.5*_erf(max_point)-0.5*_erf(min_point)

class PowerLaw_math:
    """
    Class for a powerlaw probability :math:`p(x) \\propto x^{\\alpha}` defined in
    [a,b]

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    """

    def __init__(self,alpha,min_pl,max_pl):

        self.minimum = min_pl
        self.maximum = max_pl
        self.min_pl = min_pl
        self.max_pl = max_pl
        self.alpha = alpha

        # Get the PL norm and as Eq. 24 on the paper
        self.norm = get_PL_norm(alpha,min_pl,max_pl)

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return _np.exp(self.log_prob(x))

    def log_prob(self,x):
        """
        Returns the logarithm of the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        to_ret = self.alpha*_np.log(x)-_np.log(self.norm)
        to_ret[(x<self.min_pl) | (x>self.max_pl)] = -_np.inf

        return to_ret

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        norms = get_PL_norm(self.alpha,a,b)
        to_ret = self.alpha*_np.log(x)-_np.log(norms)
        to_ret[(x<a) | (x>b)] = -_np.inf

        return to_ret

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        # Define the cumulative density function, see  Eq. 24 to see the integral form

        if self.alpha == -1:
            to_ret = _np.log(x/self.min_pl)/self.norm
        else:
            to_ret =((_np.power(x,self.alpha+1)-_np.power(self.min_pl,self.alpha+1))/(self.alpha+1))/self.norm

        to_ret *= (x>=self.min_pl)

        if hasattr(x, "__len__"):
            to_ret[x>self.max_pl]=1.
        else:
            if x>self.max_pl : to_ret=1.

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))


class Truncated_Gaussian_math(object):
    """
    Class for a truncated gaussian in
    [a,b]

    Parameters
    ----------
    mu: float
        mean of the gaussian
    sigma: float
        standard deviation of the gaussian
    min_g: float
        lower cutoff
    max_g: float
        upper cutoff
    """

    def __init__(self,mu,sigma,min_g,max_g):

        self.minimum = min_g
        self.maximum = max_g
        self.max_g=max_g
        self.min_g=min_g
        self.mu = mu
        self.sigma=sigma

        # Find the gaussian normalization as in Eq. 28 in the tex document
        self.norm = get_gaussian_norm(mu,sigma,min_g,max_g)

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return _np.exp(self.log_prob(x))


    def log_prob(self,x):
        """
        Returns the logarithm of the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        to_ret = -_np.log(self.sigma)-0.5*_np.log(2*_np.pi)-0.5*_np.power((x-self.mu)/self.sigma,2.)-_np.log(self.norm)
        to_ret[(x<self.min_g) | (x>self.max_g)] = -_np.inf

        return to_ret

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        norms = get_gaussian_norm(self.mu,self.sigma,a,b)
        to_ret = -_np.log(self.sigma)-0.5*_np.log(2*_np.pi)-0.5*_np.power((x-self.mu)/self.sigma,2.)-_np.log(norms)
        to_ret[(x<a) | (x>b)] = -_np.inf

        return to_ret


    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        # Define the cumulative density function as in Eq. 28 on the paper to see the integral form

        max_point = (x-self.mu)/(self.sigma*_np.sqrt(2.))
        min_point = (self.min_g-self.mu)/(self.sigma*_np.sqrt(2.))

        to_ret = (0.5*_erf(max_point)-0.5*_erf(min_point))/self.norm

        to_ret *= (x>=self.min_g)

        if hasattr(x, "__len__"):
            to_ret[x>self.max_g]=1.
        else:
            if x>self.max_g : to_ret=1.

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))

class PowerLawGaussian_math(object):
    """
    Class for a powerlaw probability plus gausian peak
    :math:`p(x) \\propto (1-\\lambda)x^{\\alpha}+\\lambda \\mathcal{N}(\\mu,\\sigma)`. Each component is defined in
    a different interval

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    lambda_g: float
        fraction of prob coming from gaussian peak
    mean_g: float
        mean for the gaussian
    sigma_g: float
        standard deviation for the gaussian
    min_g: float
        minimum for the gaussian component
    max_g: float
        maximim for the gaussian component
    """

    def __init__(self,alpha,min_pl,max_pl,lambda_g,mean_g,sigma_g,min_g,max_g):

        self.minimum = _np.min([min_pl,min_g])
        self.maximum = _np.max([max_pl,max_g])

        self.lambda_g=lambda_g

        self.pl= PowerLaw_math(alpha,min_pl,max_pl)
        self.gg = Truncated_Gaussian_math(mean_g,sigma_g,min_g,max_g)


    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 36-37-38 on on the tex document
        return _np.exp(self.log_prob(x))

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        return (1-self.lambda_g)*self.pl.cdf(x)+self.lambda_g*self.gg.cdf(x)

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))

    def log_prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Define the PDF as in Eq. 36-37-38 on on the tex document
        return _np.logaddexp(_np.log1p(-self.lambda_g)+self.pl.log_prob(x),_np.log(self.lambda_g)+self.gg.log_prob(x))

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return  _np.logaddexp(_np.log1p(-self.lambda_g)+self.pl.log_conditioned_prob(x,a,b),_np.log(self.lambda_g)+self.gg.log_conditioned_prob(x,a,b))

def S_factor(mass, mmin,delta_m):
    '''
    This function return the value of the window function defined as Eqs B6 and B7 of https://arxiv.org/pdf/2010.14533.pdf

    Parameters
    ----------
    mass: np.array or float
        array of x or masses values
    mmin: float or np.array (in this case len(mmin) == len(mass))
        minimum value of window function
    delta_m: float or np.array (in this case len(delta_m) == len(mass))
        width of the window function

    Returns
    -------
    Values of the window function
    '''
    if not isinstance(mass,_np.ndarray):
        mass = _np.array([mass])

    to_ret = _np.ones_like(mass)
    if delta_m == 0:
        return to_ret

    mprime = mass-mmin

    # Defines the different regions of the window function ad in Eq. B6 of  https://arxiv.org/pdf/2010.14533.pdf
    select_window = (mass>mmin) & (mass<(delta_m+mmin))
    select_one = mass>=(delta_m+mmin)
    select_zero = mass<=mmin
    effe_prime = _np.ones_like(mass)

    # Definethe f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
    effe_prime[select_window] = _np.exp(_np.nan_to_num((delta_m/mprime[select_window])+(delta_m/(mprime[select_window]-delta_m))))
    to_ret = 1./(effe_prime+1)
    to_ret[select_zero]=0.
    to_ret[select_one]=1.
    return to_ret

class SmoothedProb(object):
    '''
    Class for smoothing the low part of a PDF. The smoothing follows Eq. B7 of
    2010.14533.

    Parameters
    ----------
    origin_prob: class
        Original prior class to smooth from this module
    bottom: float
        minimum cut-off. Below this, the window is 0.
    bottom_smooth: float
        smooth factor. The smoothing acts between bottom and bottom+bottom_smooth
    '''

    def __init__(self,origin_prob,bottom,bottom_smooth):

        self.origin_prob = _copy.deepcopy(origin_prob)
        self.bottom_smooth = bottom_smooth
        self.bottom = bottom
        self.maximum=self.origin_prob.maximum
        self.minimum=self.origin_prob.minimum

        # Find the values of the integrals in the region of the window function before and after the smoothing
        xmax = _np.min([self.maximum,bottom+bottom_smooth])
        int_array = _np.linspace(bottom,xmax,1000)
        integral_before = _np.trapz(self.origin_prob.prob(int_array),int_array)
        if integral_before > 1: integral_before = 1 # approx of the trapz function
        integral_now = _np.trapz(self.prob(int_array),int_array)

        self.integral_before = integral_before
        self.integral_now = integral_now
        # Renormalize the the smoother function.
        self.norm = 1 - integral_before + integral_now - self.origin_prob.cdf(bottom)
        #        print("init norm: {}, intbefore: {}, intnow: {}, cdfbottom: {}".format(self.norm,integral_before,integral_now,self.origin_prob.cdf(bottom)))

        x_eval = _np.logspace(_np.log10(bottom),_np.log10(bottom+bottom_smooth),1000)
        cdf_numeric = _cumtrapz(self.prob(x_eval),x_eval)
        #        print("prob vec={}, int={}".format(self.prob(x_eval),cdf_numeric))
        self.cached_cdf_window = _interp1d(x_eval[:-1:],cdf_numeric,fill_value='extrapolate',bounds_error=False,kind='cubic')

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return _np.exp(self.log_prob(x))

    def log_prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Return the window function
        window = S_factor(x, self.bottom,self.bottom_smooth)

        if hasattr(self,'norm'):
            prob_ret =self.origin_prob.log_prob(x)+_np.log(window)-_np.log(self.norm)
        else:
            prob_ret =self.origin_prob.log_prob(x)+_np.log(window)

        return prob_ret

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array
            Value at which compute the probability
        a: np.array
            New lower boundary
        b: np.array
            New upper boundary
        """

        to_ret = self.log_prob(x)
        # Find the new normalization in the new interval
        new_norm = self.cdf(b)-self.cdf(a)
        # Apply the new normalization and put to zero all the values above/below the interval
        to_ret-=_np.log(new_norm)
        to_ret[(x<a) | (x>b)] = -_np.inf

        return to_ret

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        to_ret = _np.ones_like(x)
        to_ret[x<self.bottom] = 0.
        to_ret[(x>=self.bottom) & (x<=(self.bottom+self.bottom_smooth))] = self.cached_cdf_window(x[(x>=self.bottom) & (x<=(self.bottom+self.bottom_smooth))])
        to_ret[x>=(self.bottom+self.bottom_smooth)]=(self.integral_now+self.origin_prob.cdf(
        x[x>=(self.bottom+self.bottom_smooth)])-self.origin_prob.cdf(self.bottom+self.bottom_smooth))/self.norm

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array
            Value at which compute the probability
        a: np.array
            New lower boundary
        b: np.array
            New upper boundary
        """

        return _np.exp(self.log_conditioned_prob(x,a,b))
