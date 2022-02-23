__all__ = ['simulated_template']

from pylab import *
import lal
from .data import *

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
