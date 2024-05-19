__all__ = ['construct_sYlm', 'calc_YpYc']

import numpy as np
import jax.numpy as jnp
from scipy.special import factorial as fac

def binom_coeff(n, k):
    """Returns binomial coefficient, `n choose k`.
    
    Arguments
    ---------
    n: int
      number of possibilities
    k: int
      number of unordered outcomes to choose
    """

    num = fac(n)
    denom = fac(k) * fac(n - k)

    # binomial coefficient is zero if k>n, or generally if above formula returns an inf
    num[denom == 0] = 0
    denom[denom == 0] = 1
    return num / denom

def sin_th_2(cosi):
    # sin(iota/2)
    return jnp.sqrt((1 - cosi)/2)

def cos_th_2(cosi):
    return jnp.sqrt((1+cosi)/2)

def cot_th_2(cosi):
    return cos_th_2(cosi)/sin_th_2(cosi)

def construct_sYlm(s : int,
                   l : int | np.ndarray | jnp.ndarray,
                   m : int | np.ndarray | jnp.ndarray):
    """Creates spin-weighted (l, m) spherical harmonic functions with spin
    weight s,  which will be a function of the cosine of the inclination `cosi`.
    
    These are computed following the closed form expression derived by Goldberg
    et al (1967), with Mathematica's convention for the Condon-Shortley phase.
    
    See also: https://en.wikipedia.org/wiki/Spin-weighted_spherical_harmonics
    
    Arguments
    ---------
    s: int
        spin weight of the sYlm, -2 for GWs
    l: JAX array
        l index of each QNM to be included in the model
    m: JAX array
        l index of each QNM to be included in the model
    """
    
    ## with phi = 0
    ## th == np.arccos(cosi)
    
    l = np.atleast_1d(l)
    m = np.atleast_1d(m)
    
    def _get_rs(r):
        shape = len(r)
        leng = np.max(r) + 1
        coll = np.broadcast_to(np.arange(leng), (shape, leng)).swapaxes(0,1)
        return coll
    
    rs = _get_rs(l - s)

    ## normalization constant is \sqrt{1/0.159)} ##
    
    prefactor = (-1)**(l + m - s)*np.sqrt(fac(l + m)*fac(l - m)*(2*l + 1) /
                                          (4*np.pi*fac(l + s)*fac(l-s)))
    prefactor *= jnp.sqrt(5/jnp.pi) # sqrt(5/pi) to match normalization in Isi & Farr (2021)
    
    def ylm(cosi):
        cosi = jnp.clip(cosi, -0.99999, 0.99999)
        
        ylm = np.sqrt(1/0.159) * prefactor * sin_th_2(cosi)**(2*l) 

        summands = [(-1)**r * binom_coeff(l - s, r) \
                            * binom_coeff(l + s, r + s - m) \
                            * cot_th_2(cosi)**(2*r + s - m)
                    for r in rs]
        ylm *= jnp.sum(jnp.array(summands), axis=0)
        return ylm
    return ylm

def calc_YpYc(cosi, swsh):
    """Returns + and x angular factors for aligned model"""
    ylm_p = swsh(cosi)
    ylm_m = swsh(-cosi)
    Yp = ylm_p + ylm_m
    Yc = ylm_p - ylm_m
    return Yp, Yc