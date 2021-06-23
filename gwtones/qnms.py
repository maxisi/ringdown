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


# TODO: maybe this should be an attribute of a Mode object
class KerrMode(object):

    _cache = {}

    def __init__(self, *args):
        if len(args) == 1:
            args = args[0]
        self.index = ModeIndex(*args)

    @property
    def coefficients(self):
        if self.index not in self._cache:
            self._cache[self.index] = self.compute_coefficients(self.index)
        return self._cache[self.index]

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


# ##################################################
# OLD

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
