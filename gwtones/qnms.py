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

_COEFF_CACHE = {}
for n, (fc, gc) in enumerate(zip(_f_coeffs, _g_coeffs)):
    _COEFF_CACHE[ModeIndex(1, -2, 2, 2, n)] = (fg, gc)


# TODO: maybe this should be an attribute of a Mode object
class KerrMode(object):

    _cache = {}

    def __init__(self, *args):
        if len(args) == 1:
            args = args[0]
        self.index = ModeIndex(*args)

    @property
    def coefficients(self):
        i = self.index
        if i not in self._cache:
            self._cache[i] = _COEFF_CACHE.get(i, self.compute_coefficients(i))
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


# ##################################################################
# TODO: go through following functions and see what's worth keeping

# reference values

def get_ftau(M, chi, n, l=2, m=2):
    q22 = qnm.modes_cache(-2, l, m, n)
    omega, _, _ = q22(a=chi)
    f = real(omega)/(2*pi) / (T_MSUN*M)
    gamma = abs(imag(omega)) / (T_MSUN*M)
    return f, 1./gamma

def ellip_template(time, A=None, phi0=None, ellip=None, theta=None, f=None,
                   tau=None, t0=None, fpfc=None):
    dt = time.reshape(len(time), 1) - t0
    phat = A * exp(-dt/tau) * cos(2*pi*f*dt - phi0)
    chat = A * ellip * exp(-dt/tau) * sin(2*pi*f*dt - phi0)
    hp = sum(cos(theta)*phat - sin(theta)*chat, axis=1).flatten()
    hc = sum(sin(theta)*phat + cos(theta)*chat, axis=1).flatten()
    if fpfc is None:
        return hp, hc
    else:
        Fp, Fc = fpfc
        return Fp*hp + Fc*hc
    
def linear_template(time, Ap=None, Ac=None, phip=None, phic=None, f=None,
                    tau=None, t0=None, fpfc=None, **kws):
    if Ap is None:
        Ascale = kws.get('Ascale', 1)
        Ap = 0.5*Ascale*sqrt(kws['Ap_x']**2 + kws['Ap_y']**2)
        Ac = 0.5*Ascale*sqrt(kws['Ac_x']**2 + kws['Ac_y']**2)
        phip = arctan2(kws['Ap_y'], kws['Ap_x'])
        phic = arctan2(kws['Ac_y'], kws['Ac_x'])
    dt = time.reshape(len(time), 1) - t0
    hp = sum(Ap * exp(-dt/tau) * cos(2*pi*f*dt - phip), axis=1).flatten()
    hc = sum(Ac * exp(-dt/tau) * cos(2*pi*f*dt - phic), axis=1).flatten()
    if fpfc is None:
        return hp, hc
    else:
        Fp, Fc = fpfc
        return Fp*hp + Fc*hc

def circ_template(time, A=None, phi0=None, cosi=None, f=None,
                  tau=None, t0=None, fpfc=None):
    dt = time.reshape(len(time), 1) - t0
    hp = (1 + cosi**2) * A * exp(-dt/tau) * cos(2*pi*f*dt - phi0)
    hc = (2*cosi) * A * exp(-dt/tau) * sin(2*pi*f*dt - phi0)
    hp = sum(hp, axis=1).flatten()
    hc = sum(hc, axis=1).flatten()
    if fpfc is None:
        return hp, hc
    else:
        Fp, Fc = fpfc
        return Fp*hp + Fc*hc
    
def reflected_template(time, A=None, phi0=None, f=None, gamma=None, t0=None,
                       A_pre=1, df_pre=0, dtau_pre=0, gamma_pre=None, window=inf):
    ''' Create injection: a sinusoid up to t0, then a damped sinusoiud. The
    (A_pre, df_pre, dtau_pre) parameters can turn the initial sinusoid into a
    "sinegaussian", to produce a ring-up.
    
    Can incorporate several modes, if (A, phi0, f, gamma) are 1D.
    '''
    # reshape arrays (to handle multiple modes)
    signal = np.zeros(len(time))
    t = time.reshape(len(time), 1)
    
    A = np.array([A], ndmin=2)
    phi0 = np.array([phi0], ndmin=2)
    f = np.array([f], ndmin=2)
    gamma = np.array([gamma], ndmin=2)
    
    # define some masks (pre and post t0)
    mpre = (time < t0) & (abs(time-t0) < 0.5*window)
    mpost = (time >= t0) & (abs(time-t0) < 0.5*window)
    
    # signal will be a sinusoid up to t0, then a damped sinusoiud
    t_t0 = t - t0
    f_pre = f*(1 + df_pre)
    signal[mpre]  = np.sum(A*np.cos(2*np.pi*f_pre*t_t0[mpre] - phi0), axis=1).flatten()
    signal[mpost] = np.sum(A*np.cos(2*np.pi*f*t_t0[mpost] - phi0)*np.exp(-t_t0[mpost]*gamma), axis=1).flatten()
    
    # add a damping to the amplitude near t0 for t < t0
    if dtau_pre is not None or gamma_pre is not None:
        A_pre = np.array([A_pre], ndmin=2)
        gamma_pre = gamma_pre if gamma_pre is not None else gamma/(1 + dtau_pre)
        signal[mpre] *= np.sum(A_pre*np.exp(-abs(t_t0[mpre])*gamma_pre), axis=1).flatten()
    return signal
    
def well_ordered(taus):
    return all([taus[n] >= taus[n+1] for n in range(len(taus)-1)])

def good_srate(taus, delta_t):
    return all([tau >= delta_t for tau in taus])

def get_minmax(prior, delta_t, Tanalyze):
    # the minimum good dtau1 is given by the lightest
    # non-spinning BH allowed
    f, tau = get_ftau(prior['M'][0],  prior['chi'][0], 1)
    dtau_min = (delta_t - tau)/tau

    # the maximum good dtau1 is given by the heaviest
    # fastest-spinning BH allowed
    f, tau = get_ftau(prior['M'][1], prior['chi'][1], 1)
    dtau_max = (Tanalyze - tau)/tau

    # the minimum good df1 is given by the heaviest BH allowed 
    # (roughly irrespective of spin)
    f, tau = get_ftau(prior['M'][1], 0, 1)
    df_min = (1/Tanalyze - f)/f

    # the minimum good df1 is given by the heaviest BH allowed 
    # (roughly irrespective of spin)
    f, tau = get_ftau(prior['M'][0], 0, 1)
    df_max = (0.5/delta_t - f)/f

    return df_min, df_max, dtau_min, dtau_max
