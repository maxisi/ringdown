__all__ = ['condition', 'TimeSeries', 'FrequencySeries', 'Data',
           'AutoCovariance', 'PowerSpectrum']

from pylab import *
import scipy.signal as sig
import lal
import scipy.linalg as sl
import pandas as pd

# def get_raw_time_ifo(tgps, raw_time, duration=None, ds=None):
#     ds = ds or 1
#     duration = inf if duration is None else duration
#     m = abs(raw_time - tgps) < 0.5*duration
#     i = argmin(abs(raw_time - tgps))
#     return roll(raw_time, -(i % ds))[m]

def condition(raw_data, raw_time=None, flow=None, fhigh=None, ds=None,
              scipy_dec=True, remove_mean=True, t0=None, decimate_kws=None, trim=0.25):
    decimate_kws = decimate_kws or {}

    if t0 is not None:
        ds = ds or 1
        i = argmin(abs(raw_time - t0))
        raw_time = roll(raw_time, -(i % ds))
        raw_data = roll(raw_data, -(i % ds))

    fny = 0.5/(raw_time[1] - raw_time[0])
    # Filter
    if flow and not fhigh:
        b, a = sig.butter(4, flow/fny, btype='highpass', output='ba')
    elif fhigh and not flow:
        b, a = sig.butter(4, fhigh/fny, btype='lowpass', output='ba')
    elif flow and fhigh:
        b, a = sig.butter(4, (flow/fny, fhigh/fny), btype='bandpass',
                          output='ba')

    if flow or fhigh:
        cond_data = sig.filtfilt(b, a, raw_data)
    else:
        cond_data = raw_data

    # Decimate
    if ds and ds > 1:
        if scipy_dec:
            cond_data = sig.decimate(cond_data, ds, zero_phase=True, **decimate_kws)
        else:
            cond_data = cond_data[::ds]
        if raw_time is not None:
            cond_time = raw_time[::ds]
    elif raw_time is not None:
        cond_time = raw_time

    N = len(cond_data)
    istart = int(round(trim*N))
    iend = int(round((1-trim)*N))

    cond_time = cond_time[istart:iend]
    cond_data = cond_data[istart:iend]

    if remove_mean:
        cond_data -= mean(cond_data)

    return cond_time, cond_data

class TimeSeries(pd.Series):
    """ A container for time series data based on `pandas.Series`;
    the index should contain time stamps for uniformly-sampled data.
    """

    @property
    def _constructor(self):
        return TimeSeries

    @property
    def delta_t(self) -> float:
        """Sampling time interval."""
        return self.index[1] - self.index[0]

    @property
    def fsamp(self) -> float:
        """Sampling frequency (`1/delta_t`)."""
        return 1/self.delta_t

    @property
    def duration(self) -> float:
        """Time series duration (time spanned between first and last samples).
        """
        return self.delta_t * len(self)

    @property
    def delta_f(self) -> float:
        """Fourier frequency spacing."""
        return 1/self.duration

    @property
    def time(self) -> pd.Index:
        """Time stamps."""
        return self.index


class FrequencySeries(pd.Series):
    """ A container for frequency domain data based on `pandas.Series`;
    the index should contain frequency stamps for uniformly-sampled data.
    """

    @property
    def _constructor(self):
        return FrequencySeries

    @property
    def delta_f(self) -> float:
        """Fourier frequency spacing."""
        return self.index[1] - self.index[0]

    # WARNING: I think this breaks for odd N
    @property
    def delta_t(self) -> float:
        """Sampling time interval."""
        return 0.5/((len(self)-1)*self.delta_f)

    @property
    def fsamp(self) -> float:
        """Sampling frequency (`1/delta_t`)."""
        return 1/self.delta_t

    @property
    def duration(self) -> float:
        """Time series duration (time spanned between first and last samples).
        """
        return self.delta_t * len(self)

    @property
    def freq(self) -> pd.Index:
        """Frequency stamps."""
        return self.index


class Data(TimeSeries):
    """Container for time-domain strain data from a given GW detector.

    Attributes
    ----------
    ifo : str
        detector identifier (e.g., 'H1' for LIGO Hanford).
    """

    _metadata = ['ifo']

    def __init__(self, *args, ifo=None, **kwargs):
        super(Data, self).__init__(*args, **kwargs)
        self.ifo = ifo.upper() if ifo is not None else ifo

    @property
    def _constructor(self):
        return Data

    @property
    def detector(self) -> lal.Detector:
        """ :cls:`LALSuite` object containing detector information.
        """
        if self.ifo:
            d = lal.cached_detector_by_prefix[self.ifo]
        else:
            d = None
        return d

    def condition(self, flow=None, fhigh=None, ds=None, scipy_dec=True, t0=None,
                  remove_mean=True, decimate_kws=None, trim=0.25):
        """ Condition data.

        Arguments
        ---------
        flow : `float`
            lower frequency for high passing.
        fhigh : float
            higher frequency for low passing.
        ds : int
            decimation factor for downsampling.
        scipy_dec : bool
            use scipy to decimate.
        t0 : float
            target time to be preserved after downsampling.
        remove_mean : bool
            explicitly remove mean from time series after conditioning.
        decimate_kws : dict
            options for decimation function.
        trim : float
            fraction of data to trim from edges after conditioning, to avoid
            spectral issues if filtering.

        Returns
        -------
        cond_data : `Data`
            conditioned data object.
        """
        raw_data = self
        raw_time = self.index

        decimate_kws = decimate_kws or {}

        if t0 is not None:
            ds = ds or 1
            i = argmin(abs(raw_time - t0))
            raw_time = roll(raw_time, -(i % ds))
            raw_data = roll(raw_data, -(i % ds))

        fny = 0.5/(raw_time[1] - raw_time[0])
        # Filter
        if flow and not fhigh:
            b, a = sig.butter(4, flow/fny, btype='highpass', output='ba')
        elif fhigh and not flow:
            b, a = sig.butter(4, fhigh/fny, btype='lowpass', output='ba')
        elif flow and fhigh:
            b, a = sig.butter(4, (flow/fny, fhigh/fny), btype='bandpass',
                              output='ba')

        if flow or fhigh:
            cond_data = sig.filtfilt(b, a, raw_data)
        else:
            cond_data = raw_data

        # Decimate
        if ds and ds > 1:
            if scipy_dec:
                cond_data = sig.decimate(cond_data, ds, zero_phase=True,
                                         **decimate_kws)
            else:
                cond_data = cond_data[::ds]
            if raw_time is not None:
                cond_time = raw_time[::ds]
        elif raw_time is not None:
            cond_time = raw_time

        N = len(cond_data)
        istart = int(round(trim*N))
        iend = int(round((1-trim)*N))

        cond_time = cond_time[istart:iend]
        cond_data = cond_data[istart:iend]

        if remove_mean:
            cond_data -= mean(cond_data)

        return Data(cond_data, index=cond_time, ifo=self.ifo)


    def get_acf(self, **kws):
        return AutoCovariance.from_data(self, **kws)

    def get_psd(self, **kws):
        return PowerSpectrum.from_data(self, **kws)


class PowerSpectrum(FrequencySeries):

    def __init__(self, *args, delta_f=None, **kwargs):
        super(PowerSpectrum, self).__init__(*args, **kwargs)
        if delta_f is not None:
            self.index = arange(len(self))*delta_f

    @property
    def _constructor(self):
        return PowerSpectrum

    @classmethod
    def from_data(self, data, f_low=None, **kws):
        fs = kws.pop('fs', 1/getattr(data, 'delta_t', 1))
        freq, psd = sig.welch(data, fs=fs, **kws)
        p = PowerSpectrum(psd, index=freq)
        if f_low:
            p.flatten(f_low, inplace=True)
        return p

    def flatten(self, f_low, inplace=False):
        freq = self.freq
        if not inplace:
            psd = self.copy()
        fref = freq[freq >= f_low][0]
        psd_ref = self[argmin(abs(freq - fref))]
        def get_low_freqs(f):
            return psd_ref + psd_ref*(fref-f)*np.exp(-(fref-f))/3
        psd[freq < f_low] = get_low_freqs(freq[freq < f_low])
        if not inplace:
            return psd

    def to_acf(self):
        rho = 0.5*np.fft.irfft(self) / self.delta_t
        return AutoCovariance(rho, delta_t=self.delta_t)


class AutoCovariance(TimeSeries):

    def __init__(self, *args, delta_t=None, **kwargs):
        super(AutoCovariance, self).__init__(*args, **kwargs)
        if delta_t is not None:
            self.index = arange(len(self))*delta_t

    @property
    def _constructor(self):
        return AutoCovariance

    @classmethod
    def from_data(self, d, n=None, dt=1, nperseg=None, f_low=None,
                  method='td'):
        dt = getattr(d, 'delta_t', dt)
        n = n or len(d)
        if method.lower() == 'td':
            rho = sig.correlate(d, d)
            rho = ifftshift(rho)
            rho = rho[:n] / len(d)
        elif method.lower() == 'fd':
            nperseg = nperseg or 3*len(d)
            freq, psd = sig.welch(d, fs=1/dt, nperseg=nperseg)
            rho = 0.5*np.fft.irfft(psd)[:n] / dt
        else:
            raise ValueError("method must be 'td' or 'fd' not %r" % method)
        return AutoCovariance(rho, delta_t=dt)

    def to_psd(self):
        # acf = 0.5*np.fft.irfft(psd) / delta_t
        psd = 2 * self.delta_t * abs(np.fft.rfft(self))
        freq = np.fft.rfftfreq(len(self), d=self.delta_t)
        return PowerSpectrum(psd, index=freq)

    @property
    def matrix(self):
        return sl.toeplitz(self)

    @property
    def cholesky(self):
        if getattr(self, '_cholesky', None) is None:
            self._cholesky = linalg.cholesky(self.matrix)
        return self._cholesky

    def compute_snr(self, x, y=None):
        if y is None: y = x
        ow_x = sl.solve_toeplitz(self.iloc[:len(x)], x)
        return dot(ow_x, y)/sqrt(dot(x, ow_x))

    def whiten(self, data):
        """Whiten stretch of data using ACF.

        Arguments
        ---------
        data : array, TimeSeries
            unwhitened data.

        Returns
        -------
        w_data : Data
            whitened data.
        """
        if isinstance(data, TimeSeries):
            assert (data.delta_t == self.delta_t)
        # whiten stretch of data using Cholesky factor
        L = self.iloc[:len(data)].cholesky
        w_data = np.linalg.solve(L, data)
        # return same type as input
        if isinstance(data, Data):
            w_data = Data(w_data, index=data.index, ifo=data.ifo)
        elif isinstance(data, TimeSeries):
            w_data = TimeSeries(w_data, index=data.index)
        return w_data

