from pylab import *
import h5py
import lal
import lalsimulation
import scipy.signal
import os
os.environ["LAL_DATA_PATH"] = os.path.join(os.environ['HOME'], "lscsoft/src/lalsuite-extra/data/lalsimulation/")

# utilities from https://git.ligo.org/waveforms/reviews/nrsur7dq4/blob/master/utils.py
# and from Carl
lalsim = lalsimulation
from lalsimulation.nrfits import eval_nrfit

lalinf_intr_keys = ['m1', 'm2', 'a1', 'a2', 'theta_jn', 'tilt1', 'tilt2', 'phi_jl', 'phi12', 'phase', 'f_ref']


def change_spin_convention(theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1,
                           m2, f_ref, phi_orb=0.):
    iota, S1x, S1y, S1z, S2x, S2y, S2z = lalsim.SimInspiralTransformPrecessingNewInitialConditions(
                                             theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2,
                                             m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_ref, phi_orb)
    spin1 = [S1x, S1y, S1z]
    spin2 = [S2x, S2y, S2z]
    return spin1, spin2, iota

def set_single_mode(params, l, m):
    """ Sets modes in params dict.
        Only adds (l,m) and (l,-m) modes.
    """
    # First, create the 'empty' mode array
    ma = lalsimulation.SimInspiralCreateModeArray()
    # add (l,m) and (l,-m) modes
    lalsimulation.SimInspiralModeArrayActivateMode(ma, l, m)
    lalsimulation.SimInspiralModeArrayActivateMode(ma, l, -m)    
    # then insert the ModeArray into the LALDict params
    lalsimulation.SimInspiralWaveformParamsInsertModeArray(params, ma)
    return params

def get_remnant_from_lalinf(*args, **kwargs):
    model_name = kwargs.pop('model_name', 'NRSur7dq4Remnant')
    fit_type_list = ['FinalMass', 'FinalSpin']#, 'RecoilKick']
    
    phase = kwargs.pop('phase')
    theta_jn = kwargs.pop('theta_jn')
    m1_msun = kwargs.pop('m1')
    m2_msun = kwargs.pop('m2')
    m1_kg = m1_msun*lal.MSUN_SI
    m2_kg = m2_msun*lal.MSUN_SI

    phi_jl = kwargs.pop('phi_jl')
    tilt1 = kwargs.pop('tilt1')
    tilt2 = kwargs.pop('tilt2')
    phi12 = kwargs.pop('phi12')
    a1 = kwargs.pop('a1')
    a2 = kwargs.pop('a2')
    
    f_ref = kwargs.pop('f_ref')
    
    chi1, chi2, _ = change_spin_convention(theta_jn, phi_jl, tilt1, tilt2,
                                           phi12, a1, a2, m1_msun, m2_msun,
                                           f_ref, phase)

    data = eval_nrfit(m1_kg, m2_kg, chi1, chi2, model_name, fit_type_list, f_ref=f_ref)
    return data['FinalMass'], data['FinalSpin']#, data['RecoilKick']


def generate_lal_hphc_from_lalinf(*args, **kwargs):
    phase = kwargs.pop('phase')
    theta_jn = kwargs.pop('theta_jn')
    m1_msun = kwargs.pop('m1')
    m2_msun = kwargs.pop('m2')

    phi_jl = kwargs.pop('phi_jl')
    tilt1 = kwargs.pop('tilt1')
    tilt2 = kwargs.pop('tilt2')
    phi12 = kwargs.pop('phi12')
    a1 = kwargs.pop('a1')
    a2 = kwargs.pop('a2')
    
    f_ref = kwargs.pop('f_ref')

    c1, c2, iota = change_spin_convention(theta_jn, phi_jl, tilt1, tilt2,
                                          phi12, a1, a2, m1_msun, m2_msun,
                                          f_ref, phase)
    kwargs.update({
        'm1_msun': m1_msun,
        'm2_msun': m2_msun,
        'chi1': c1,
        'chi2': c2,
        'inclination': iota,
        'phi_ref': phase,
    })
    return generate_lal_hphc(*args, **kwargs)

def generate_lal_hphc(approximant_key, m1_msun=None, m2_msun=None, chi1=None,
                      chi2=None, dist_mpc=None, dt=None, f_low=20, f_ref=20,
                      inclination=None, phi_ref=None, ell_max=None,
                      single_mode=None, epoch=None, mtot_msun=None, 
                      nr_path=None):

    approximant = lalsim.SimInspiralGetApproximantFromString(approximant_key)

    param_dict = lal.CreateDict()

    # NR handling based on https://arxiv.org/abs/1703.01076
    if approximant_key == 'NR_hdf5':
        # get masses
        mtot_msun = mtot_msun or m1_msun + m2_msun
        with h5py.File(nr_path, 'r') as f:
            m1 = f.attrs['mass1']
            m2 = f.attrs['mass2']
            m1_msun = m1 * mtot_msun/(m1 + m2)
            m2_msun = m2 * mtot_msun/(m1 + m2)
        # Compute spins in the LAL frame
        s1x, s1y, s1z, s2x, s2y, s2z = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(f_ref, mtot_msun, nr_path)
        chi1 = [s1x, s1y, s1z]
        chi2 = [s2x, s2y, s2z]
        # Create a dictionary and pass /PATH/TO/H5File
        params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertNumRelData(param_dict, nr_path)
        longAscNodes = np.pi / 2
    else:
        longAscNodes = 0.

    m1_kg = m1_msun*lal.MSUN_SI
    m2_kg = m2_msun*lal.MSUN_SI
    
    distance = dist_mpc*1e6*lal.PC_SI

    if single_mode is not None and ell_max is not None:
        raise Exception("Specify only one of single_mode or ell_max")

    if ell_max is not None:
        # If ell_max, load all modes with ell <= ell_max
        ma = lalsim.SimInspiralCreateModeArray()
        for ell in range(2, ell_max+1):
            lalsim.SimInspiralModeArrayActivateAllModesAtL(ma, ell)
        lalsim.SimInspiralWaveformParamsInsertModeArray(param_dict, ma)
    elif single_mode is not None:
        # If a single_mode is given, load only that mode (l,m) and (l,-m)
        param_dict = set_single_mode(param_dict, single_mode[0], single_mode[1])

    hp, hc = lalsim.SimInspiralChooseTDWaveform(m1_kg, m2_kg,
                                                chi1[0], chi1[1], chi1[2],
                                                chi2[0], chi2[1], chi2[2],
                                                distance, inclination,
                                                phi_ref, longAscNodes,
                                                0., 0., dt, f_low, f_ref,
                                                param_dict, approximant)
    return hp, hc

def generate_lal_waveform(*args, **kwargs):
    times = kwargs.pop('times')
    triggertime = kwargs.pop('triggertime')
    
    bufLength = len(times)
    delta_t = times[1] - times[0]
    tStart = times[0]
    tEnd = tStart + delta_t * bufLength

    kwargs['dt'] = delta_t

    hplus = kwargs.pop('hplus', None)
    hcross = kwargs.pop('hcross', None)
    if (hplus is None) or (hcross is None):
        hplus, hcross = generate_lal_hphc(*args, **kwargs)
    
    # align waveform, based on LALInferenceTemplate
    # https://git.ligo.org/lscsoft/lalsuite/blob/master/lalinference/lib/LALInferenceTemplate.c#L1124

    # /* The nearest sample in model buffer to the desired tc. */
    tcSample = round((triggertime - tStart)/delta_t)

    # /* The actual coalescence time that corresponds to the buffer
    #    sample on which the waveform's tC lands. */
    # i.e. the nearest time in the buffer
    injTc = tStart + tcSample*delta_t

    # /* The sample at which the waveform reaches tc. */
    if kwargs.pop('manual_epoch', False):
        # manually find peak of the waveform envelope
        habs = np.sqrt(hplus.data.data**2 + hcross.data.data**2)
        waveTcSample = np.argmax(habs)
    else:
        hplus_epoch = hplus.epoch.gpsSeconds + hplus.epoch.gpsNanoSeconds*1E-9
        waveTcSample = round(-hplus_epoch/delta_t)

    # /* 1 + (number of samples post-tc in waveform) */
    wavePostTc = hplus.data.length - waveTcSample

    # bufStartIndex = (tcSample >= waveTcSample ? tcSample - waveTcSample : 0);
    bufStartIndex = int(tcSample - waveTcSample if tcSample >= waveTcSample else 0)
    # size_t bufEndIndex = (wavePostTc + tcSample <= bufLength ? wavePostTc + tcSample : bufLength);
    bufEndIndex = int(tcSample + wavePostTc if tcSample + wavePostTc <= bufLength else bufLength)
    bufWaveLength = bufEndIndex - bufStartIndex
    waveStartIndex = int(0 if tcSample >= waveTcSample else waveTcSample - tcSample)

    if kwargs.get('window', True) and tcSample >= waveTcSample:
        # smoothly turn on waveform
        window = scipy.signal.tukey(bufWaveLength)
        window[int(0.5*bufWaveLength):] = 1.
    else:
        window = 1
    h_td = np.zeros(bufLength, dtype=complex)
    h_td[bufStartIndex:bufEndIndex] = window*hplus.data.data[waveStartIndex:waveStartIndex+bufWaveLength] -\
                                      1j*window*hcross.data.data[waveStartIndex:waveStartIndex+bufWaveLength]
    return h_td

def project_fd(hp_fd, hc_fd, frequencies, Fp=None, Fc=None, time_delay=None, ifo=None, **kws):
    
    if ifo is not None:
        triggertime_geo = kws['triggertime']
        geo_gps_time = lal.LIGOTimeGPS(triggertime_geo)
        gmst = lal.GreenwichMeanSiderealTime(geo_gps_time)

        detector = lal.cached_detector_by_prefix[ifo]
        # get antenna patterns
        Fp, Fc = lal.ComputeDetAMResponse(detector.response, kws['ra'], 
                                          kws['dec'], kws['psi'], gmst)
        # get time delay and align waveform
        # assume reference time corresponds to envelope peak
        time_delay = lal.TimeDelayFromEarthCenter(detector.location, 
                                                  kws['ra'], kws['dec'],
                                                  geo_gps_time)
    
    fancy_timedelay = lal.LIGOTimeGPS(time_delay)
    timeshift = fancy_timedelay.gpsSeconds + 1e-9*fancy_timedelay.gpsNanoSeconds
    
    timeshift_vector = np.exp(-2.*1j*np.pi*timeshift*frequencies)
    
    h_fd = (Fp*hp_fd + Fc*hc_fd)*timeshift_vector
    return h_fd

def project_td(h_td, times, **kwargs):
    hp_td = h_td.real
    hc_td = -h_td.imag
    
    fft_norm = times[1] - times[0]
    hp_fd = np.fft.rfft(hp_td) * fft_norm
    hc_fd = np.fft.rfft(hc_td) * fft_norm
    frequencies = np.fft.rfftfreq(len(times)) / fft_norm

    h_fd = project_fd(hp_fd, hc_fd, frequencies, **kwargs)
    return np.fft.irfft(h_fd) / fft_norm

def get_peak_times(*args, **kwargs):
    p = kwargs.pop('parameters')
    times = kwargs.pop('times')
    ifos = kwargs.pop('ifos', ['H1', 'L1', 'V1'])
    approx = kwargs.pop('approx', 'NRSur7dq4')
    
    delta_t = times[1] - times[0]
    tlen = len(times)
    
    fp = {k: kwargs[k] if k in kwargs else p[k] for k in ['f_ref', 'flow', 'lal_amporder']}

    chi1, chi2, iota = change_spin_convention(p['theta_jn'], p['phi_jl'], p['tilt1'], p['tilt2'],
                                          p['phi12'], p['a1'], p['a2'], p['m1'], p['m2'],
                                          fp['f_ref'], p['phase'])
    
    f_start = fp['flow']*2/(fp['lal_amporder'] + 2.)
    # get strain
    h_td = generate_lal_waveform(approx, p['m1'], p['m2'], chi1, chi2, dist_mpc=p['dist'],
                                  dt=delta_t, f_low=f_start, f_ref=fp['f_ref'], inclination=iota,
                                  phi_ref=p['phase'], ell_max=None, times=times, triggertime=p['time'])
    # FFT
    hp_td = h_td.real
    hc_td = -h_td.imag

    fft_norm = delta_t
    hp_fd = np.fft.rfft(hp_td) * fft_norm
    hc_fd = np.fft.rfft(hc_td) * fft_norm
    frequencies = np.fft.rfftfreq(tlen) / fft_norm
    
    # get peak time
    tp_geo_loc = np.argmax(np.abs(h_td))
    tp_geo = times[tp_geo_loc]
    
    geo_gps_time = lal.LIGOTimeGPS(p['time'])
    gmst = lal.GreenwichMeanSiderealTime(geo_gps_time)

    tp_dict = {'geo': tp_geo}
    for ifo in ifos:
        detector = lal.cached_detector_by_prefix[ifo]
        # get antenna patterns
        Fp, Fc = lal.ComputeDetAMResponse(detector.response, p['ra'], p['dec'], p['psi'], gmst)
        # get time delay and align waveform
        # assume reference time corresponds to envelope peak
        timedelay = lal.TimeDelayFromEarthCenter(detector.location,  p['ra'], p['dec'], geo_gps_time)

        fancy_timedelay = lal.LIGOTimeGPS(timedelay)
        timeshift = fancy_timedelay.gpsSeconds + 1e-9*fancy_timedelay.gpsNanoSeconds

        timeshift_vector = np.exp(-2.*1j*np.pi*timeshift*frequencies)
    
        tp_dict[ifo] = tp_geo + timedelay
    return tp_dict

def get_fd_waveforms(*args, **kwargs):
    p = kwargs.pop('parameters')
    times = kwargs.pop('times')
    ifos = kwargs.pop('ifos', ['H1', 'L1', 'V1'])
    approx = kwargs.pop('approx', 'NRSur7dq4')
    
    delta_t = times[1] - times[0]
    tlen = len(times)
    
    fp = {k: kwargs[k] if k in kwargs else p[k] for k in ['f_ref', 'flow', 'lal_amporder']}

    chi1, chi2, iota = change_spin_convention(p['theta_jn'], p['phi_jl'], p['tilt1'], p['tilt2'],
                                              p['phi12'], p['a1'], p['a2'], p['m1'], p['m2'],
                                              fp['f_ref'], p['phase'])
    
    f_start = fp['flow']*2/(fp['lal_amporder'] + 2.)
    # get strain
    h_td = generate_lal_waveform(approx, p['m1'], p['m2'], chi1, chi2, dist_mpc=p['dist'],
                                  dt=delta_t, f_low=f_start, f_ref=fp['f_ref'], inclination=iota,
                                  phi_ref=p['phase'], ell_max=None, times=times, triggertime=p['time'])
    # FFT
    hp_td = h_td.real
    hc_td = -h_td.imag

    fft_norm = delta_t
    hp_fd = np.fft.rfft(hp_td) * fft_norm
    hc_fd = np.fft.rfft(hc_td) * fft_norm
    frequencies = np.fft.rfftfreq(tlen) / fft_norm
    
    # get peak time
    tp_geo_loc = np.argmax(np.abs(h_td))
    tp_geo = times[tp_geo_loc]
    
    geo_gps_time = lal.LIGOTimeGPS(p['time'])
    gmst = lal.GreenwichMeanSiderealTime(geo_gps_time)

    h_fd_dict = {}
    for ifo in ifos:
        detector = lal.cached_detector_by_prefix[ifo]
        # get antenna patterns
        Fp, Fc = lal.ComputeDetAMResponse(detector.response, p['ra'], p['dec'], p['psi'], gmst)
        # get time delay and align waveform
        # assume reference time corresponds to envelope peak
        timedelay = lal.TimeDelayFromEarthCenter(detector.location,  p['ra'], p['dec'], geo_gps_time)

        fancy_timedelay = lal.LIGOTimeGPS(timedelay)
        timeshift = fancy_timedelay.gpsSeconds + 1e-9*fancy_timedelay.gpsNanoSeconds

        timeshift_vector = np.exp(-2.*1j*np.pi*timeshift*frequencies)
        h_fd_dict[ifo] = (Fp*hp_fd + Fc*hc_fd)*timeshift_vector
    
    return h_fd_dict

NR_PATH = '/Users/maxisi/lscsoft/src/lvcnr-lfs/SXS/SXS_BBH_0305_Res6.h5'
def get_signal_dict(time_dict, dist, phase=pi, inclination=0, **kwargs): 
    # get antenna patterns and trigger times
    tgps_dict = kwargs.pop('tgps_dict', None)
    ap_dict = kwargs.pop('ap_dict', None)
    if not tgps_dict or not ap_dict:
        ra, dec, psi = [kwargs.pop(k) for k in ['ra', 'dec', 'psi']]
        ifos = time_dict.keys()
        tgps_dict, ap_dict = utils.get_tgps_aps(tgps_geocent, ra, dec, psi, ifos)
        
    # get complex strain
    time = list(time_dict.values())[0]
    delta_t = time[1] - time[0]
    approx = kwargs.pop('approx')
    hp, hc = generate_lal_hphc(approx, dist_mpc=dist, inclination=inclination,
                               dt=delta_t, phi_ref=phase, **kwargs)


    # project signal onto detector
    raw_signal_dict = {}
    for ifo, time in time_dict.items():
        h = generate_lal_waveform(hplus=hp, hcross=hc, times=time,
                                  triggertime=tgps_dict[ifo],
                                  manual_epoch=True)
        Fp, Fc = ap_dict[ifo]
        h_ifo = Fp*h.real - Fc*h.imag

        raw_signal_dict[ifo] = h_ifo
    return raw_signal_dict, tgps_dict, ap_dict