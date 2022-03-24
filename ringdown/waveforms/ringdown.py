__all__ = ['Ringdown', 'simulated_template']

from pylab import *
import lal
from .core import *
from ..data import TimeSeries
from .. import qnms
from inspect import getfullargspec
import warnings

class Ringdown(Signal):
    _metadata = ['modes']

    def __init__(self, *args, modes=None, **kwargs):
        super(Ringdown, self).__init__(*args, **kwargs)
        self.modes = qnms.construct_mode_list(modes)

    @property
    def _constructor(self):
        return Ringdown

    @property
    def n_modes(self):
        return len(self.get_parameter('A', []))

    @staticmethod
    def _theta_phi_from_phip_phim(phip, phim):
        return -0.5*(phip + phim), 0.5*(phip - phim)

    @staticmethod
    def _phip_phim_from_theta_phi(theta, phi):
        return phi - theta, -(phi + theta)

    @staticmethod
    def _construct_parameters(ndmin=0, **kws):
        kws = {k.lower(): v for k,v in kws.items()}
        # define parameters that will take precedence for storage
        keys = ['a', 'ellip', 'theta', 'phi', 'omega', 'gamma']
        pars = {k: array(kws.pop(k), ndmin=ndmin) for k in list(kws.keys())
                if k in keys}
        # possibly interpret parameters as coming from a specific stan model
        if 'model' in kws:
            if kws['model'] == 'mchi_aligned':
                # this assumes a template for the oscillatory part like
                #     hp = (1 + cosi**2) * A * cos(wt - phi)
                #     hc = 2*cosi * A * sin(wt - phi)
                # in the parameterization adopted here, this corresponds to 
                #     A = A*(1 + cosi**2), e = 2*cosi / (1 + cosi**2)
                #     phi = phi, theta = 0
                # the ellipticity is already computed within stan, but we need
                # to readjust the definition of the amplitude and add theta
                pars['cosi'] = kws.pop('cosi')*ones_like(pars['a'])
                pars['a'] = pars['a']*(1 + pars['cosi']**2)
                kws['theta'] = zeros_like(pars['a'])
                if 'ellip' not in kws:
                    kws['ellip'] = 2*pars['cosi'] / (1 + pars['cosi']**2)
            elif kws['model'] == 'ftau':
                # this assumes a template for the oscillatory part like
                #     hp = Ax*cos(wt) + Ay*sin(wt) = A*cos(wt - phi)
                #     hc = 0
                # with phi = atan2(Ay, Ax) and A = sqrt(Ax**2 + Ay**2)
                # in the parameterization adopted here, this corresponds to
                #     A = A, ellip = 0, theta = 0, phi = phi
                kws['theta'] = zeros_like(pars['a'])
                kws['ellip'] = zeros_like(pars['a'])
        # obtain frequencies from remnant parameters if necessary
        freq_keys = ['omega', 'gamma', 'f', 'tau']
        if 'modes' in kws and not any([k in kws for k in freq_keys]):
            if 'M' in kws:
                kws['m'] = kws.pop('M')
            kws['approx'] = kws.get('approx', False)
            kws['f'], kws['tau'] = [], []
            for m in kws['modes']:
                f, tau = qnms.KerrMode(m).ftau(kws['chi'], kws['m'],
                                               kws['approx'])
                kws['f'].append(f)
                kws['tau'].append(tau)
        # frequency parameters
        if 'f' in kws and 'omega' not in kws:
            pars['omega'] = 2*pi*array(kws.pop('f'), ndmin=ndmin)
        if 'tau' in kws and 'gamma' not in kws:
            pars['gamma'] = 1/array(kws.pop('tau'), ndmin=ndmin)
        # phase parameters
        if 'phip' in kws and 'phim' in kws:
            theta, phi = Ringdown._theta_phi_from_phip_phim(kws['phip'],
                                                            kws['phim'])
            pars['theta'] = array(theta, ndmin=ndmin)
            pars['phi'] = array(phi, ndmin=ndmin)
        pars.update(kws)
        return pars

    @staticmethod
    def complex_mode(time, omega, gamma, A, ellip, theta, phi):
        """Eq. (8) in Isi & Farr (2021).
        """
        phi_p = phi - theta
        phi_m = - (phi + theta)
        iwt = 1j*omega*time
        vt = gamma*time
        # h = 0.5*A*exp(-time*gamma)*((1 + ellip)*exp(-1j*(wt - phi_p)) +
        #                             (1 - ellip)*exp(1j*(wt + phi_m)))
        h = ((0.5*A)*(1 + ellip))*exp(1j*phi_p - iwt - vt) + \
            ((0.5*A)*(1 - ellip))*exp(1j*phi_m + iwt - vt)
        return h

    #_MODE_PARS = [k.lower() for k in getfullargspec(Ringdown.complex_mode)[0][1:]]
    _MODE_PARS = ['omega', 'gamma', 'a', 'ellip', 'theta', 'phi']

    @classmethod
    def from_parameters(cls, time, t0=0, window=inf, two_sided=True, df_pre=0,
                        dtau_pre=0, mode_isel=None, **kws):
        """Create injection: a sinusoid up to t0, then a damped sinusoiud. The
        (A_pre, df_pre, dtau_pre) parameters can turn the initial sinusoid into
        a sinegaussian, to produce a ring-up.  Can incorporate several modes,
        if (A, phi0, f, tau) are 1D.

        Arguments
        ---------
        mode_isel : list, int
            index or indices of modes to include in template; if `None` 
            includes all modes (default).
        """
        # parse arguments
        all_kws = {k: v for k,v in locals().items() if k not in ['cls','time']}
        all_kws.update(all_kws.pop('kws'))
        modes = all_kws.get('modes', None)

        # reshape arrays (to handle multiple modes)
        t = reshape(time, (len(time), 1))
        signal = zeros(len(time), dtype=complex)

        pars = cls._construct_parameters(ndmin=1, **all_kws)

        # define some masks (pre and post t0) to avoid evaluating the waveform
        # over extremely long time arrays [optional]
        mpost = (time >= t0) & (time < t0 + 0.5*window) 

        # each mode will be a sinusoid up to t0, then a damped sinusoid
        if mode_isel == None:
            mode_isel = slice(None)
        margs = {k: array(pars[k][mode_isel], ndmin=2) for k in cls._MODE_PARS}
        if modes:
            if len(modes) > len(margs[0][0]):
                raise ValueError("insufficient parameters provided")
        signal[mpost] = sum(cls.complex_mode(t[mpost]-t0, *margs.values()),
                            axis=1)

        # add a damping to the amplitude near t0 for t < t0
        if two_sided:
            margs['omega'] = margs['omega']*exp(df_pre)
            margs['gamma'] = -margs['gamma']*exp(-dtau_pre)
            mpre = (time < t0) & (time > t0 - 0.5*window)
            signal[mpre] = sum(cls.complex_mode(t[mpre]-t0, *margs.values()),
                               axis=1)
        return cls(signal, index=time, parameters=pars, modes=modes)
    
    def get_parameter(self, k, *args, **kwargs):
        k = k.lower()
        if k == 'f':
            return self.get_parameter('omega', *args, **kwargs)/ (2*pi)
        elif k == 'tau':
            return 1/self.get_parameter('gamma', *args, **kwargs)
        elif k == 'phip' or k == 'phim':
            th = self.get_parameter('theta', *args, **kwargs)
            ph = self.get_parameter('phi', *args, **kwargs)
            d = dict(zip(['phip', 'phim'],
                         self._phip_phim_from_theta_phi(th, ph)))
            return d[k]
        elif k == 'quality':
            # Q = pi*f*tau
            w = self.get_parameter('omega', *args, **kwargs)
            v = self.get_parameter('gamma', *args, **kwargs)
            return 0.5 * w / v
        else:
            return super().get_parameter(k, *args, **kwargs)

    def get_mode_parameters(self, mode):
        try:
            n = int(mode)
        except (TypeError, ValueError):
            n = self.modes.index(qnms.ModeIndex(*mode))
        pars = {k: self.parameters[k][n] for k in self._MODE_PARS}
        return pars


# #############################################################################

def simulated_template(freq, tau, smprate, duration, theta, phi, amplitude, ra,
                       dec, ifos, tgps, ellip, psi=0):
    """Function to make a simulated signal, as in Isi & Farr (2021)
    [`arXiv:2107.05609 <https://arxiv.org/abs/2107.05609>`_] Eqns. (11)-(13),
    here using a "ring-up" followed by a ring-down as modeled by a time decay
    of :math:`\exp(-|t-t_\mathrm{gps}-\Delta t_\mathrm{ifo}|\gamma)`.
            
    Note that in general this function works best for durations much longer
    than the time delay. No noise is overlaid on this template.

    Arguments
    ---------
    freq : list
        list of frequencies of each tone in Hz.
    tau : list
        damping time of each tone in seconds.
    theta : list
        angle of ellipse in :math:`h_+,h_\\times` space, in radians.
    phi : list
        phase offset of sinusoids, in radians.
    ellip : list
        ellipticity of each tone, :math:`-1 \leq \epsilon \leq 1`.
    smprate : float
        sample rate of signal in Hz.
    duration : float
        length of time of the full signal in seconds.
    ra : float
        source right ascension in radians.
    dec : float
        source declination in radians.
    psi : float
        polarization angles for which to evaluate antenna patterns (def. 0).
    ifos : list
        list of ifos as strings, e.g., `["H1", "L1"]`.
    tgps : float
        GPS time for peak of signal.
    amplitude : list
        amplitude of each tone.
        
    Returns
    -------
    sig_dict : dict
        Dict of signal :class:`ringdown.data.TimeSeries` for each IFO.
    modes_dict : dict
        Dict of mode :class:`ringdown.data.TimeSeries` for each ifo.
    time_delay_dict : dict
        Dict of `lal.TimeDelayFromEarthCenter` for each ifo.
    """
    warnings.warn("simulated_template() is deprecated; use Ringdown class", warnings.DeprecationWarning)
    N = int((duration*smprate))
    t = arange(N)/smprate+tgps-duration/2.0 #list of times that will be used as fake data input
    s = TimeSeries(zeros_like(t), index=t) #array for template signal
    hplus = TimeSeries(zeros_like(t), index=t) #array for h_plus polarization, h_plus = hcos*cos(theta) - epsilon*hsin*sin(theta)
    hcross = TimeSeries(zeros_like(t), index=t) #array for h_cross polarization, h_cross = hcos*sin(theta) + epsilon*hsin*cos(theta)
    hsin = TimeSeries(zeros_like(t), index=t) #sine quadrature
    hsin = TimeSeries(zeros_like(t), index=t) #cosine quadrature
    sig_dict = {} #dicts used for template output
    lal_det = {} #ifo information
    modes_dict = {} #individual mode information
    antenna_patterns = {} #antenna patterns projected onto modes
    time_delay_dict = {} #time delays from Earth center
    omega = 2*pi*array(freq) #frequencies
    gamma = 1./array(tau) #damping
    gmst = lal.GreenwichMeanSiderealTime(tgps)
    for ifo in ifos:
        s = s-s #hacky way to zero out the arrays for fresh signal in each ifo
        lal_det[ifo] = lal.cached_detector_by_prefix[ifo]
        antenna_patterns[ifo] = lal.ComputeDetAMResponse(lal_det[ifo].response, ra, dec, psi, gmst)
        time_delay_dict[ifo] = lal.TimeDelayFromEarthCenter(lal_det[ifo].location, ra, dec, tgps)
        modes_dict[ifo]=[]
        for (w,v,A,E,n,th,ph) in zip(omega,gamma,amplitude,ellip,arange(len(omega)),theta,phi):
            hsin = TimeSeries(A*exp(-abs(t-tgps-time_delay_dict[ifo])*v)*sin(w*(t-tgps-time_delay_dict[ifo])-ph), index=t)
            hcos = TimeSeries(A*exp(-abs(t-tgps-time_delay_dict[ifo])*v)*cos(w*(t-tgps-time_delay_dict[ifo])-ph), index=t)
            hplus= hcos*cos(th)-E*hsin*sin(th)
            hcross= hcos*sin(th)+E*hsin*cos(th)
            m = antenna_patterns[ifo][0]*hplus+antenna_patterns[ifo][1]*hcross
            s += m
            modes_dict[ifo].append(m)
        sig_dict[ifo]=s    
    return sig_dict, modes_dict, time_delay_dict
