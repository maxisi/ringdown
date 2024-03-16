__all__ = ['construct_mode_list', 'KerrMode', 'get_ftau']

from pylab import *
import qnm
import lal
from collections import namedtuple

T_MSUN = lal.MSUN_SI * lal.G_SI / lal.C_SI**3

ModeIndexBase = namedtuple('ModeIndex', ['p', 's', 'l', 'm', 'n'])

class ModeIndex(ModeIndexBase):
    def to_bytestring(self):
        s = f"p={self.p}, s={self.s}, l={self.l}, m={self.m}, n={self.n}"
        return bytes(s, 'utf-8')

def construct_mode_list(modes):
    if modes is None:
        modes = []
    elif isinstance(modes, str):
        # assume modes is a string like "(p0,s0,l0,m0,n0),(p1,s1,l1,m1,n1)"
        from ast import literal_eval
        modes = literal_eval(modes)
    mode_list = []
    for (p, s, l, m, n) in modes:
        mode_list.append(ModeIndex(p, s, l, m, n))
    return mode_list

# TODO: is it even worth caching these here?

_f_coeffs = [
     [-3.90543324e-03,  5.95090414e-02, -2.35344242e-02, 3.24149319e-04,  4.96912520e-04,  3.62247860e-05],
     [-3.05794712e-03,  5.52300575e-02, -2.54841972e-02,-1.53819704e-05,  4.84745438e-04,  3.72002303e-05],
     [ 1.41448042e-03,  4.79764455e-02, -2.55184851e-02, 7.09998492e-04,  6.83309264e-04,  5.19604646e-05],
     [ 1.47012112e-02,  4.00265199e-02, -1.64570303e-02, 5.11637192e-03,  1.50553871e-03,  1.04018967e-04],
     [ 5.65635128e-02,  3.25894918e-02,  1.97538933e-02, 1.90805137e-02,  3.78391970e-03,  2.36244161e-04],
     [-2.05642550e-01,  2.73402853e-02, -2.31960947e-01,-7.84295476e-02, -1.15783541e-02, -6.23617413e-04],
     [ 3.06051317e-02,  1.96570571e-02, -1.85900176e-02, 3.41348041e-03,  9.61814922e-04,  5.54477972e-05],
     [ 5.05519552e-02,  1.35255773e-02, -1.11834989e-02, 4.90643382e-03,  1.12709047e-03,  6.49052859e-05]
     ]

_g_coeffs = [
    [-2.09655358e-02,  8.89718149e-02, -1.87089632e-02,-1.35405686e-02, -2.19235034e-03, -1.15746228e-04],
    [-7.15676358e-02,  2.74034569e-01, -5.70096259e-02,-4.07670432e-02, -6.57569321e-03, -3.45939611e-04],
    [-1.57167550e-01,  4.78774540e-01, -1.06281090e-01,-7.08486898e-02, -1.12607728e-02, -5.86348790e-04],
    [-2.75449425e-01,  7.06668465e-01, -1.57838034e-01,-9.89600927e-02, -1.52116665e-02, -7.63621476e-04],
    [-2.90068279e-01,  9.50179282e-01, -8.67886243e-02,-7.63502027e-02, -1.00565215e-02, -3.62218607e-04],
    [ 1.05251627e+00,  1.18592622e+00,  1.21692088e+00, 3.67901676e-01,  4.90171297e-02,  2.40342069e-03],
    [-8.31114452e-01,  1.44202697e+00, -4.83400168e-01,-3.02435787e-01, -5.53073710e-02, -3.36696640e-03],
    [-9.55159727e-01,  1.68374775e+00, -5.45943428e-01,-3.32236179e-01, -5.82868428e-02, -3.38484116e-03]
]

_COEFF_CACHE = {}
for n, (fc, gc) in enumerate(zip(_f_coeffs, _g_coeffs)):
    _COEFF_CACHE[ModeIndex(1, -2, 2, 2, n)] = (fc, gc)


class KerrMode(object):

    _cache = {}

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]
        self.index = ModeIndex(*args, **kwargs)

    @property
    def coefficients(self):
        i = self.index
        if i not in self._cache:
            self._cache[i] = _COEFF_CACHE.get(i, None) or self.compute_coefficients(i)
        return self._cache[i]

    @staticmethod
    def compute_coefficients(mode, n_chi=1000, **kws):
        p, s, l, m, n = mode
        chis = linspace(0, 1, n_chi)[:-1]
        logchis = log1p(-chis)
        M = column_stack((chis, ones_like(chis), logchis, logchis**2,
                          logchis**3, logchis**4))

        q = qnm.modes_cache(s, l, p*abs(m), n)
        sgn = 1 if m == 0 else sign(m)
        f = sgn*array([q(c)[0].real for c in chis])/(2*pi)
        g = array([abs(q(c)[0].imag) for c in chis])

        coeff_f = np.linalg.lstsq(M, f, rcond=None, **kws)[0]
        coeff_g = np.linalg.lstsq(M, g, rcond=None, **kws)[0]
        return coeff_f, coeff_g

    def __call__(self, *args, **kwargs):
        f, tau = self.ftau(*args, **kwargs)
        return 2*pi*f - 1j/tau

    def ftau(self, chi, m_msun=None, approx=False):
        if approx:
            logchi = log1p(-chi)
            c = (chi, ones_like(chi), logchi, logchi**2, logchi**3, logchi**4)
            f, g = [dot(coeff, c) for coeff in self.coefficients]
        else:
            p, s, l, m, n = self.index
            q = qnm.modes_cache(s, l, p*abs(m), n)
            def omega(c):
                return q(c)[0]
            f = sign(m)*vectorize(omega)(chi).real/(2*pi)
            g = abs(vectorize(omega)(chi).imag)
        if m_msun:
           f /= (m_msun * T_MSUN)
           g /= (m_msun * T_MSUN)
            
        return f, 1/g


def get_ftau(M, chi, n, l=2, m=2):
    q22 = qnm.modes_cache(-2, l, m, n)
    omega, _, _ = q22(a=chi)
    f = real(omega)/(2*pi) / (T_MSUN*M)
    gamma = abs(imag(omega)) / (T_MSUN*M)
    return f, 1./gamma

