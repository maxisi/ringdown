import numpy as np
import lal
import lalsimulation as ls

def _ishift(hp_t, hc_t):
    hmag = np.sqrt(hp_t*hp_t + hc_t*hc_t)

    ib = np.argmax(hmag)
    if ib == len(hmag) - 1:
        ic = 0
        ia = ib-1
    elif ib == 0:
        ia = len(hmag)-1
        ic = 1
    else:
        ia = ib-1
        ic = ib+1

    a = hmag[ia]
    b = hmag[ib]
    c = hmag[ic]

    return (len(hmag) - (ib + (3*a - 4*b + c)/(2*(a-2*b+c)) - 1))%len(hmag)

def complex_strain_peak_time_fd(sample, wf=ls.IMRPhenomPv2, f_high=1024, df=0.5, f_low=20., f_ref=100.):
    """Return the time associated with the peak complex strain in detector
    `prefix` and associated parameter values.

    :param sample: A numpy record of the posterior sample of interest.

    :param wf: The FD waveform to use to estimate the peak complex strain
      (default: IMRPhenomPv2).

    :param f_high: The highest frequency for which to compute the waveform.

    :param df: Frequency spacing (should be <= 1/T).

    :param f_low: Low frequency limit for waveform generation (if `None`, will
      try to find in samples)

    :param f_ref: Reference frequency at which the time-dependent quantities
      like spins are defined (if `None` will try to find in samples).

    :return: `(t_peak_geocent, t_peak_dict, hp, hc)`: the GPS time of the peak
      at geocenter, a dict mapping 'geocent' or IFO name to GPS of peak arrival,
      plus frequency series, cross frequency series.
    """
    tname = 'geocent_time'
    samp = sample
    geocent_GPS = lal.LIGOTimeGPS(samp[tname])

    if f_low is None:
        f_low = samp['flow']

    if f_ref is None:
        f_ref = samp['f_ref']

    hp, hc = ls.SimInspiralChooseFDWaveform(samp['mass_1']*lal.MSUN_SI,
                                            samp['mass_2']*lal.MSUN_SI,
                                            samp['spin_1x'], samp['spin_1y'], samp['spin_1z'],
                                            samp['spin_2x'], samp['spin_2y'], samp['spin_2z'],
                                            samp['luminosity_distance']*1e6*lal.PC_SI,
                                            samp['iota'],
                                            samp['phase'],
                                            0.0, 0.0, 0.0,
                                            df, f_low, f_high, f_ref,
                                            None,
                                            wf)

    hp_t = np.fft.irfft(hp.data.data)
    hc_t = np.fft.irfft(hc.data.data)

    ishift = _ishift(hp_t, hc_t)

    dt = 1/(2*f_high)
    ipeak = len(hp_t) - ishift
    tpeak = hp.epoch + ipeak*dt

    t_dict = {'geocent': samp[tname] + tpeak}
    for ifo in ['H1', 'L1', 'V1']:
        tdelay = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location, samp['ra'], samp['dec'], geocent_GPS)
        t_dict[ifo] = t_dict['geocent'] + tdelay

    return samp[tname] + tpeak, t_dict, hp, hc

def complex_strain_peak_time_td(sample, wf=ls.NRSur7dq4, dt=1.0/1024.0, f_low=20., f_ref=100.):
    """Return the time associated with the peak complex strain in detector
    `prefix` and associated parameter values.

    :param sample: A numpy record with the posterior sample describing the
      waveform of interest.

    :param wf: The TD waveform to use to estimate the peak complex strain
      (default: NRSur7dq4).

    :param dt: Time spacing (default 1.0/1024.0)

    :param f_low: Low frequency limit for waveform generation (if `None`, will
      try to find in samples)

    :param f_ref: Reference frequency at which the time-dependent quantities
      like spins are defined (if `None` will try to find in samples).

    :return: `(t_peak_geocent, t_peak_dict, hp, hc)`: the GPS time of the peak
      at geocenter, a dict mapping 'geocent' or IFO name to GPS of peak arrival,
      plus frequency series, cross frequency series.
    """
    tname = 'geocent_time'
    samp = sample
    geocent_GPS = lal.LIGOTimeGPS(samp[tname])

    if f_low is None:
        f_low = samp['flow']

    if f_ref is None:
        f_ref = samp['f_ref']

    hp, hc = ls.SimInspiralChooseTDWaveform(samp['mass_1']*lal.MSUN_SI,
                                            samp['mass_2']*lal.MSUN_SI,
                                            samp['spin_1x'], samp['spin_1y'], samp['spin_1z'],
                                            samp['spin_2x'], samp['spin_2y'], samp['spin_2z'],
                                            samp['luminosity_distance']*1e6*lal.PC_SI,
                                            samp['iota'],
                                            samp['phase'],
                                            0.0, 0.0, 0.0,
                                            dt, f_low, f_ref,
                                            None,
                                            wf)

    ishift = _ishift(hp.data.data, hc.data.data)

    # Peak happens at
    ipeak = hp.data.data.shape[0] - ishift
    tpeak = dt*ipeak + float(hp.epoch)

    t_dict = {'geocent': samp[tname] + tpeak}
    for ifo in ['H1', 'L1', 'V1']:
        tdelay = lal.TimeDelayFromEarthCenter(lal.cached_detector_by_prefix[ifo].location, samp['ra'], samp['dec'], geocent_GPS)
        t_dict[ifo] = t_dict['geocent'] + tdelay

    return t_dict['geocent'], t_dict, hp, hc
