from pylab import *
import qnm
import lal

T_MSUN = lal.MSUN_SI * lal.G_SI / lal.C_SI**3

# reference values
REFLSTSQR = {
    0: {
        'A': array([1.11029226])*1E-21,
        'phi0': [0.04992418],
    },
    1: {
        'A': array([2.21960765, 2.54166205])*1E-21,
        'phi0': [-0.74096361, 1.83572294],
    },
    2: {
        # 'A': [2.29276816, 4.74034966, 2.48456137],
        'A': array([2.5, 5, 2.5])*1E-21,
        'phi0': [-1.1400821, 1.22669085, -2.31598923]
    }
}

REF150914 = {
    'A': [4.16155615e-21, 5.66277487e-21],
    'phi0': [-0.93908037,  1.79189372],
}

REFDEF = {
    'A': array([2.29276816, 4.74034966, 2.48456137])*1E-21,
    'phi0': array([-0.93908037, 1.79189372, -2.31598923]),
}


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