__all__ = ['Signal', '_ishift', 'get_detector_signals', 'get_delay']

from pylab import *
import lal
from ..data import Data, TimeSeries
from scipy.interpolate import interp1d
from inspect import getfullargspec

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


class Signal(TimeSeries):
    _metadata = ['parameters']
    _T0_ALIASES = ['t0', 'geocent_time', 'trigger_time', 'triggertime',
                   'tc', 'tgps_geo', 'tgps_geocent']
    _FROM_GEO_KEY = 'from_geo'

    _MODEL_REGISTER = {}

    def __init__(self, *args, parameters=None, **kwargs):
        super(Signal, self).__init__(*args, **kwargs)
        self.parameters = parameters or {}

    @property
    def _constructor(self):
        return Signal

    @property
    def t0(self):
        """Reference time `t0`, or alias, from :attr:`Signal.parameters`.
        Valid aliases are: `{}`.
        """
        for k in self._T0_ALIASES:
            if k in self.parameters:
                return self.get_parameter(k)
    t0.__doc__ = t0.__doc__.format(_T0_ALIASES)

    @staticmethod
    def _register_model(obj):
        for m in obj._MODELS:
            Signal._MODEL_REGISTER[m] = obj.from_parameters

    def get_parameter(self, k, *args):
        return self.parameters.get(k.lower(), *args) 

    @property
    def _hp(self):
        return np.real(self) 

    @property
    def hp(self):
        return Signal(self._hp, index=self.index, parameters=self.parameters)

    @property
    def _hc(self):
        return -np.imag(self) 

    @property
    def hc(self):
        return Signal(self._hc, index=self.index, parameters=self.parameters)

    @property
    def envelope(self):
        return Signal(sqrt(self._hp**2 + self._hc**2), index=self.index,
                      parameters=self.parameters)

    @property
    def peak_time(self):
        """Empirical estimate of the time of waveform envelope peak,
        :math:`\max\left(h_+^2 + h_\\times^2\\right)`, obtained through
        quadratic sub-sample interpolation.
        """
        ipeak = len(self) - _ishift(self._hp, self._hc)
        tpeak = self.delta_t*ipeak + float(self.time[0])
        return tpeak

    @classmethod
    def from_parameters(cls, *args, **kwargs):
        """Produce a gravitational wave from parameters, for either a full
        compact binary coalescence, or the ringdown alone.

        Input is passed to the corresponding constructor method, depending on
        the `model` specified in arguments. Assumes signal is ringdown by default.

        Arguments
        ---------
        \*args :
            arguments passed to constructor.
        \*\*kwargs:
            keyword arguments passed to constructor.

        Returns
        -------
        signal : Signal
            subclass of signal.
        """
        if not cls._MODEL_REGISTER:
            raise ValueError("no models registered: reload or reinstall ringdown")
        m = kwargs.get('model', 'default')
        if m in cls._MODEL_REGISTER:
            return cls._MODEL_REGISTER[m](*args, **kwargs)
        else:
            raise ValueError("unrecognized model: {}".format(m))

    def project(self, ifo=None, t0=None, antenna_patterns=None, delay=None,
                ra=None, dec=None, psi=None, fd_shift=False, interpolate=False):
        """Project waveform onto detector, multiplying by antenna patterns and 
        shifting by time-of-flight delay (if required).

        Arguments
        ---------
        ifo : str
            name of detector onto which to project signal, used to compute
            `antenna_patterns` and `delay`, if not provided explicitly.
        t0 : float
            trigger time used to compute `antenna_patterns` and `delay` if not
            provided explicitly; if projecting from `ifo` name, assumes this is
            GPS time. Defaults to :attr:`Signal.t0`.
        antenna_patterns : tuple
            tuple of floats with plus and cross antenna pattern values;
            computed based on `ifo` and sky location if not given.
        delay : float, str
            numerical delay by which to shift signal, or ``'from_geo'`` to
            indicate delay is to be computed relative to geocenter for detector
            specified by `ifo`.  Defaults to 0, which results in no time shift.
        ra : float
            source right ascension (rad) if projecting based on detector name.
        dec : float
            source declination (rad) if projecting based on detector name.
        psi : float
            source polarization angle (rad) if projecting based on detector
            name.
        fd_shift : bool
            time shift signal in the Fourier domain. Defaults to False.
        interpolate : bool
            use cubic subsample signal interpolation when shifting in the time
            domain (`fd_shift = False`); otherwise, rolls by number of samples
            closest to `delay`. Defaults to False.

        Returns
        -------
        data : Data
            signal projected onto detector.
        """
        if antenna_patterns is None and ifo:
            tgps = lal.LIGOTimeGPS(t0 or self.t0)
            gmst = lal.GreenwichMeanSiderealTime(tgps)
            det = lal.cached_detector_by_prefix[ifo]
            antenna_patterns = lal.ComputeDetAMResponse(det.response, ra, dec,
                                                        psi, gmst)
        else:
            tgps = None
            det = None

        Fp, Fc = antenna_patterns
        h = Fp*self._hp + Fc*self._hc
        if isinstance(delay, str):
            delay = get_delay(ifo, tgps or t0 or self.t0, ra, dec, delay)
        else:
            delay = delay or 0
        if fd_shift:
            h_fd = np.fft.rfft(h)
            frequencies = np.fft.rfftfreq(len(h), d=self.delta_t)
            timeshift_vector = np.exp(-2.*1j*np.pi*delay*frequencies)
            h = np.fft.irfft(h_fd * timeshift_vector, n=len(h))
        else:
            idt = int(round(delay * self.fsamp))
            if interpolate:
                hint = interp1d(self.time, h, kind='cubic', fill_value=0,
                                bounds_error=False)
                dt = (idt - delay*self.fsamp)*self.delta_t
                h = hint(self.time + dt)
            h = np.roll(h, idt)
        # record projection information
        info = self.parameters.copy()
        pars = locals()
        info.update({k: pars[k] for k in ['tgps', 'antenna_patterns', 'delay',
                                          'ra', 'dec', 'psi', 'fd_shift',
                                          'interpolate']})
        return Data(h, ifo=ifo, index=self.time, info=info)
    
    def plot(self, ax=None, envelope=False):
        """Plot the series' plus and cross components.
        Remember that the value of this timeseries is ``h = hp - 1j*hc``.
        """
        if ax is None:
            fig, ax = subplots(1)
        ax.plot(self.time, self._hp, label="$h_+$")
        ax.plot(self.time, self._hc, label=r"$h_\times$")
        if envelope:
            ax.plot(self.time, abs(self), label="$h$", ls='--', c='k')
        legend(loc='best')
        return ax


def get_detector_signals(times=None, ifos=None, antenna_patterns=None,
                         trigger_times=None, t0_default=None,
                         fast_projection=False, **kws):
    """Produce templates at each detector for a given set of parameters. Can be
    used to generate waveforms from model samples, or to obtain IMR injections.

    Additional keyword arguments are passed to :meth:`Signal.from_parameters`
    and :meth:`Signal.project`.

    Arguments
    ---------
    times : dict, array
        a dictionary with time stamps for each detector, or a single array of
        time stamps to be shared by all detectors
    ifos : list
        list of detector names, only required if `times` is a single array
    antenna_patterns : dict, None
        optional dictionary of tuples with pluss and cross antenna patterns for
        each detector, `(Fp, Fc)`; computed from sky location if not given.
    trigger_times : dict, None
        dictionary of arrival-times at each detector; computed from sky
        location and reference trigger time if not given.
    t0_default : float
        optional default trigger time, to be used if no other valid trigger time
        argument is provided.
    fast_projection : bool
        if true, evaluates polarization functions only once using the time
        array of the first interferometer and then projects onto each
        detector by time shifiting; otherwise, evaluates polarizations for
        each detector, ensuring that there are no off-by-one alignment
        errors. (Def. False)
    \*\*kws :
        arguments passed to :meth:`Signal.from_parameters` and/or
        :meth:`Signal.project`.
    
    Returns
    -------
    sdict : dict
        dictionary of :class:`Data` waveforms for each detector.
    """
    # parse GW and projection arguments
    all_kws = {k: v for k,v in locals().items() if k not in ['times', 'ifos']}
    all_kws.update(all_kws.pop('kws'))

    # arguments for signal generation
    s_kws = all_kws.copy()

    # check if a trigger time was provided
    t0 = None
    for k in Signal._T0_ALIASES:
        t0 = t0 if t0 is not None else s_kws.pop(k, None)
    if t0 is None:
        t0 = t0_default
    # get other arguments for signal projection
    p_kws ={k: s_kws.pop(k) for k in kws.keys() if k in 
            getfullargspec(Signal.project)[0][1:] and k != 't0'}

    # parse detectors and time arrays
    if isinstance(times, dict):
        # assume `times` is a dictionary with entries for each detector
        # (define `time` anyway, to be used in case of `fast_projection`)
        ifos = ifos or list(times.keys())
        time = times[ifos[0]]
    elif ifos is not None:
        # assume `times` is a single time array to be used for all detectors
        time = times
        times = {ifo: time for ifo in ifos}
    else:
        raise ValueError("must provide interferometers")

    # check if antenna patterns were provided
    if antenna_patterns is None:
        antenna_patterns = {}
    sky_provided = all([k in p_kws for k in ['ra', 'dec']])
    if not antenna_patterns and not sky_provided:
        raise ValueError("must provide antenna patterns or sky location")

    # parse time options: first, could provide individual start times explicitly
    # if so, this will take precedence over all other options
    trigger_times = trigger_times or {}

    if all([i in trigger_times for i in ifos]):
        # all start times given, compute relative delays
        if t0 is None:
            # set reference time to the first start time
            t0 = trigger_times[ifos[0]]
    elif trigger_times:
        # some start times given, but not all: fail
        x = [i for i in ifos if i not in trigger_times]
        raise ValueError(f"missing trigger times for some detectors: {x!r}")
    elif sky_provided:
        # no trigger times given, compute from sky location
        if t0 is None:
            raise ValueError("missing reference time for sky location")
        else:
            trigger_times = {i: t0 + get_delay(i, t0, p_kws['ra'],
                                               p_kws['dec']) for i in ifos}
    else:
        raise ValueError("must provide trigger times or sky location")
        
    # some models allow for a `dts` parameter which shift t0 for detectors
    # other than the first (i.e., a relative shift wrt the first detector)
    # Note: this is not to be confused with the time-of-flight delay!
    dts = dict(zip(ifos[1:], kws.get('dts', zeros(len(ifos)-1))))

    sdict = {}
    if fast_projection:
        # evaluate GW polarizations once and timeshift for each detector:
        # first, get the trigger time and assume it refers to geocenter
        # generate signal, evaluating at geocenter timestamps
        h = Signal.from_parameters(time, t0=t0, **s_kws)
        # project onto each detector: we provide a target geocenter time from
        # which delays to each individual detectors will be computed
        # (potentially plus an additional `dt` defined above); we also provide
        # antenna patterns for the projection, or let them be computed from the
        # sky location and time.
        for i in ifos:
            sdict[i] = h.project(antenna_patterns=antenna_patterns.get(i, None),
                                 delay=trigger_times[i] + dts.get(i, 0) - t0,
                                 ifo=i, **p_kws)
    else:
        # revaluate the template from scratch for each detector first,
        # check if a trigger time was provided: if so, assume this
        # reference time refers to geocenter
        for i, time in times.items():
            # target time will be the start time at this detector
            # (potentially plus an arbitrary `dt` shift as above)
            s_kws['t0'] = trigger_times[i] + dts.get(i, 0)
            h = Signal.from_parameters(time, **s_kws)
            sdict[i] = h.project(antenna_patterns=antenna_patterns.get(i, None),
                                 delay=0, ifo=i, **p_kws)
    return sdict

def get_delay(ifo, t0, ra, dec, reference=Signal._FROM_GEO_KEY):
    """ Establish truncation target, stored to `self.target`.
    """
    if not isinstance(t0, lal.LIGOTimeGPS):
        t0 = lal.LIGOTimeGPS(t0)
    d = lal.cached_detector_by_prefix[ifo]
    if reference.lower() == Signal._FROM_GEO_KEY:
        dt = lal.TimeDelayFromEarthCenter(d.location, ra, dec, t0)
    else:
        # TODO: implement delay from given detector
        raise ValueError(f"unrecognized time reference {reference!r}")
    return dt

