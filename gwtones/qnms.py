from pylab import *
import qnm
import lal
from collections import namedtuple

T_MSUN = lal.MSUN_SI * lal.G_SI / lal.C_SI**3

ModeIndex = namedtuple('ModeIndex', ['p', 's', 'l', 'm', 'n'])

def construct_mode_list(modes):
    if modes is None:
        modes = []
    mode_list = []
    for (p, s, l, m, n) in modes:
        mode_list.append(ModeIndex(p, s, l, m, n))
    return mode_list

# TODO: is it even worth caching these here?

_f_coeffs = [
    [-0.00823557, 0.05994978, -0.00106621,  0.08354181, -0.15165638,  0.11021346],  
    [-0.00817443, 0.05566216,  0.00174642,  0.08531863, -0.15465231,  0.11326354],  
    [-0.00803510, 0.04842476,  0.00545740,  0.09300492, -0.16959796,  0.12487077],  
    [-0.00779142, 0.04067346,  0.00491459,  0.12084976, -0.22269851,  0.15991054],  
    [-0.00770094, 0.03418526, -0.00823914,  0.20643478, -0.37685018,  0.24917989],  
    [ 0.00303002, 0.02558406,  0.06756237, -0.15655673,  0.36731757, -0.20880323],  
    [-0.00948223, 0.02209137, -0.00671374,  0.22389539, -0.36335472,  0.21967326],  
    [-0.00931548, 0.01429318,  0.03356735,  0.11195758, -0.20533169,  0.14109002]   
    ]

_g_coeffs = [
    [ 0.01180702,  0.08838127,  0.02528302, -0.09002286,  0.18245511, -0.12162592],
    [ 0.03360470,  0.27188580,  0.07460669, -0.31374292,  0.62499252, -0.4116911 ],
    [ 0.05754774,  0.47487353,  0.10275994, -0.52484007,  1.03658076, -0.67299196],
    [ 0.08300547,  0.70003289,  0.11521228, -0.77083409,  1.48332672, -0.93350403],
    [ 0.11438483,  0.94036953,  0.10326999, -0.89912932,  1.62233654, -0.96391884],
    [-0.01888617,  1.20407042, -0.49651606,  1.04793870, -2.02319930,  0.88102107],
    [ 0.10530775,  1.43868390, -0.05621762, -1.38317450,  3.05769954, -2.25940348],
    [ 0.14280084,  1.69019137, -0.25210715, -0.67029321,  2.09513036, -1.8255968 ]
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
        M = column_stack((log1p(-chis), ones_like(chis), chis, chis**2,
                          chis**3, chis**4))

        q = qnm.modes_cache(s, l, p*abs(m), n)
        f = sign(m)*array([q(c)[0].real for c in chis])/(2*pi)
        g = array([abs(q(c)[0].imag) for c in chis])

        coeff_f = np.linalg.lstsq(M, f, rcond=None, **kws)[0]
        coeff_g = np.linalg.lstsq(M, g, rcond=None, **kws)[0]
        return coeff_f, coeff_g

    def __call__(self, *args, **kwargs):
        f, tau = self.ftau(*args, **kwargs)
        return 2*pi*f - 1j/tau

    def ftau(self, chi, m_msun=None, approx=False):
        if approx:
            c = (log1p(-chi), ones_like(chi), chi, chi**2, chi**3, chi**4)
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

