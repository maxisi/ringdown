__all__ = ['Signal', '_ishift']

from pylab import *
import lal
from ..data import Data, TimeSeries
from scipy.interpolate import interp1d

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
    _MODEL_REGISTER = {}

    def __init__(self, *args, parameters=None, **kwargs):
        super(Signal, self).__init__(*args, **kwargs)
        self.parameters = parameters or {}

    @property
    def _constructor(self):
        return Signal

    @property
    def t0(self):
        """Reference time `t0`, or otherwise `geocent_time`, from
        :attr:`Signal.parameters`.
        """
        return self.get_parameter('t0',
                                  self.get_parameter('geocent_time', None))

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
    def peak_time(self):
        """Empirical estimate of the time of waveform envelope peak,
        :math:`\max\left(h_+^2 + h_\\times^2\\right)`, obtained through
        quadratic sub-sample interpolation.
        """
        ipeak = len(self) - _ishift(self._hp, self._hc)
        tpeak = self.delta_t*ipeak + float(self.time[0])
        return tpeak

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
            if delay.lower() == 'from_geo':
                tgps = tgps or lal.LIGOTimeGPS(t0 or self.t0)
                det = det or lal.cached_detector_by_prefix[ifo]
                delay = lal.TimeDelayFromEarthCenter(det.location,ra,dec,tgps)
            else:
                raise ValueError("invalid delay reference: {}".format(delay))
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
        return Data(h, ifo=ifo, index=self.time)

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


