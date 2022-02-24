__all__ = ['simulated_template']

from pylab import *
import lal
from .data import *
from scipy.interpolate import interp1d

class Signal(TimeSeries):
    _metadata = ['parameters']

    def __init__(self, *args, parameters=None, **kwargs):
        super(Signal, self).__init__(*args, **kwargs)
        self.parameters = parameters

    @property
    def _constructor(self):
        return Signal

    @property
    def t0(self):
        return self.parameters.get('t0', None)

    @property
    def _hp(self):
        return np.real(self) 

    @property
    def hp(self):
        hp = self.copy()
        hp.iloc[:] = self._hp
        return hp

    @property
    def _hc(self):
        return -np.imag(self) 

    @property
    def hc(self):
        hc = self.copy()
        hc.iloc[:] = self._hc
        return hc

    def project(self, ifo=None, t0=None, antenna_patterns=None, delay=None,
                ra=None, dec=None, psi=None, fd_shift=False, interpolate=False):
        if antenna_patterns is None and ifo:
            tgps = lal.LIGOTimeGPS(t0 or self.t0)
            gmst = lal.GreenwichMeanSiderealTime(tgps)
            det = lal.cached_detector_by_prefix[ifo]
            antenna_patterns = lal.ComputeDetAMResponse(det.response, ra, dec,
                                                        psi, gmst)
        Fp, Fc = antenna_patterns
        h = Fp*self._hp + Fc*self._hc
        if isinstance(delay, str):
            if delay.lower() == 'from_geo':
                tgps = lal.LIGOTimeGPS(t0 or self.t0)
                det = lal.cached_detector_by_prefix[ifo]
                delay = lal.TimeDelayFromEarthCenter(det.location,ra,dec,tgps)
            else:
                raise ValueError("invalid delay reference: {}".format(delay))
        else:
            delay = delay or 0
        if fd_shift:
            h_fd = np.fft.rfft(h)
            frequencies = np.fft.rfftfreq(len(h), d=self.delta_t)
            timeshift_vector = np.exp(-2.*1j*np.pi*delay*frequencies)
            h = np.fft.irfft(h_fd * timeshift_vector)
        else:
            idt = int(round(delay * self.fsamp))
            if interpolate:
                hint = interp1d(self.time, h, kind='cubic', fill_value=0,
                                bounds_error=False)
                dt = (idt - delay*self.fsamp)*self.delta_t
                h = hint(self.time + dt)
            h = np.roll(h, idt)
        return Data(h, ifo=ifo, index=self.time)


class Ringdown(Signal):
    @property
    def _constructor(self):
        return Ringdown

    @classmethod
    def from_pars(cls, time, f=None, tau=None, A=None, ellip=None, theta=None,
                  phi=None, t0=0, A_pre=1, df_pre=0, dtau_pre=None,
                  window=np.inf):
        """Create injection: a sinusoid up to t0, then a damped sinusoiud. The
        (A_pre, df_pre, dtau_pre) parameters can turn the initial sinusoid into
        a sinegaussian, to produce a ring-up.  Can incorporate several modes,
        if (A, phi0, f, tau) are 1D.
        """
        # reshape arrays (to handle multiple modes)
        signal_cos = np.zeros(len(time))
        t = time.reshape(len(time), 1)

        A = np.array([A], ndmin=2)
        phi0 = np.array([phi0], ndmin=2)
        f = np.array([f], ndmin=2)
        tau = np.array([tau], ndmin=2)

        # define some masks (pre and post t0)
        mpre = (time < t0) & (abs(time-t0) < 0.5*window)
        mpost = (time >= t0) & (abs(time-t0) < 0.5*window)

        # signal will be a sinusoid up to t0, then a damped sinusoiud
        t_t0 = t - t0
        f_pre = f*(1 + df_pre)
        signal_cos[mpre] = np.sum(A*np.cos(2*np.pi*f_pre*t_t0[mpre] - phi0), axis=1).flatten()
        signal_cos[mpost] = np.sum(A*np.cos(2*np.pi*f*t_t0[mpost] - phi0)*np.exp(-t_t0[mpost]/tau), axis=1).flatten()

        # add a damping to the amplitude near t0 for t < t0
        if dtau_pre is not None:
            tau_pre = tau * (1 + dtau_pre)
            signal[mpre] *= A_pre*np.exp(-abs(t_t0[mpre])/tau_pre).flatten()
        return cls(signal, index=time)



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
