from pylab import *
import lal
from .data import *

def simulated_template(freq,tau,smprate,duration,theta,phi,amplitude,alpha,declination,ifolist,tgps,ellip,psi=0):
    #Function to make a simulated signal, as in "Analyzing black-hole ringdowns" Eqns. 9-13, here using a 
    #"ring-up" followed by a ring-down as modeled by a time decay of exp(-abs(t-tgps-time_delay_dict[ifo])*v)
    #
    #The various paramters are defined as follows
    #freq: frequency of each tone, Hz
    #tau: damping time of each tone, seconds
    #phase: phasor offset for complex amplitude, note 2 independent phases phi_l,+m,n and phi_l,-m,n are expected,
    #       thus for each tone the list must be entered as [[phi0plus,phi0minus],[phi1plus,ph1minus],....], in radians
    #ellip: ellipticity of each tone, -1<=ellip<=1
    #smprate: sample rate of signal, Hz
    #duration: length of time of the full signal, seconds
    #alpha: source right ascension
    #declination: source declination
    #psi: degenerate with alpha and declination, therefore evaluated at arbitrary fixed value of zero
    #ifolist: List of ifos as strings, i.e. ["H1","L1"]
    #tgps: gps time for peak of signal
    #amplitude: amplitude of each tone
    #
    #Note freq,tau,phase,amplitude,epsilon must all be entered as a list ordered by overtones,
    #and in general this function works best for durations much longer than the time delay
    #
    #No noise is overlaid on this template - noise should be generated separately as another ringdown.TimeSeries object
    #and overlaid on this template
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
    for ifo in ifolist:
        s = s-s #hacky way to zero out the arrays for fresh signal in each ifo
        lal_det[ifo] = lal.cached_detector_by_prefix[ifo]
        antenna_patterns[ifo] = lal.ComputeDetAMResponse(lal_det[ifo].response, alpha, declination, psi, gmst)
        time_delay_dict[ifo] = lal.TimeDelayFromEarthCenter(lal_det[ifo].location, alpha, declination, tgps)
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
