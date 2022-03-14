__all__ = ['TimeSeries', 'FrequencySeries', 'Data',
           'AutoCovariance', 'PowerSpectrum']

from pylab import *
import scipy.signal as sig
import lal
import scipy.linalg as sl
from scipy.interpolate import interp1d
import pandas as pd
import h5py
import os
import logging

# def get_raw_time_ifo(tgps, raw_time, duration=None, ds=None):
#     ds = ds or 1
#     duration = inf if duration is None else duration
#     m = abs(raw_time - tgps) < 0.5*duration
#     i = argmin(abs(raw_time - tgps))
#     return roll(raw_time, -(i % ds))[m]

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

    def interpolate(self, times=None, t0=None, duration=None, fsamp=None):
        """Reinterpolate the timeseries to contain the new indices
        Arguments
        ---------
        times : list or numpy array or pd.Series
            array of GPS times (seconds) to label the times
        t0 : float
            instead of an array of times, one can provide the start time t0
            the duration and sample rate. If both this and
            times is provided then this takes the duration and fsamp from the times
            array and sets the t0 to be the initial one
        duration: float
            duration of the new interpolated signal
        fsamp: float
            sample rate of the new interpolated signal
        """
        if times is None:
            t0 = t0 or self.time.min()
            duration = duration or (self.time.max() - t0)
            fsamp = fsamp or self.fsamp

            # Create the timing array
            times = np.arange(0.0, duration, 1/fsamp) + t0
            print(f"the new timing array sample rate {fsamp} is {1/(times[1]-times[0])}")

            # Make sure we don't include points outside of the index
            if times.max() > self.time.max():
                times = times[times <= self.time.max()]
        elif t0 is not None:
            # Use the times array for the delta_t and duration, but set 
            # the t0 provided
            times = times - times[0] + t0
            
        # Interpolate to the new times
        interp_func = interp1d(self.time, self.values, kind='cubic', fill_value=0, bounds_error=False);
        interp = interp_func(times)

        kwargs = {}
        for attr in getattr(self, '_metadata', []):
            kwargs[attr] = getattr(self, attr)

        return self._constructor(interp, index=times, **kwargs)

    @classmethod
    def read(cls, path, kind=None, **kws):
        kind = (kind or '').lower()
        if not kind:
            # attempt to guess filetype
            ext = os.path.splitext(path)[1].lower().strip('.')
            if ext in ['h5', 'hdf5', 'hdf']:
                kind = 'hdf'
            elif ext in ['txt', 'gz', 'dat', 'csv']:
                kind = 'csv'
            else:
                raise ValueError("unrecognized extension: {}".format(ext))

        if kind == 'gwosc':
            with h5py.File(path, 'r') as f:
                t0 = f['meta/GPSstart'][()]
                T = f['meta/Duration'][()]
                h = f['strain/Strain'][:]
                dt = T/len(h)
                time = t0 + dt*arange(len(h))
                return cls(h, index=time, **kws)
        elif kind in ['hdf', 'csv']:
            read_func = getattr(pd, 'read_{}'.format(kind))
            # get list of arguments accepted by pandas read function in order
            # to filter out extraneous arguments that should go to cls
            read_vars = read_func.__code__.co_varnames
            # define some defaults to ensure we get a Series and not a DataFrame
            read_kws = dict(sep=None, index_col=0, squeeze=True)
            if 'sep' in kws:
                # gymnastics to be able to support `sep = \t` (e.g., when
                # reading a config file)
                kws['sep'] = kws['sep'].encode('raw_unicode_escape').decode('unicode_escape')
            read_kws.update({k: v for k,v in kws.items() if k in read_vars})
            cls_kws = {k: v for k,v in kws.items() if k not in read_vars}
            if kind == 'csv' and 'float_precision' not in read_kws:
                logging.warning("specify `float_precision='round_trip'` or risk "
                                "strange errors due to precission loss")
            return cls(read_func(path, **read_kws), **cls_kws)
        else:
            raise ValueError("unrecognized file kind: {}".format(kind))


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

    def interpolate(self, freqs=None, fmin=None, fmax=None, df=None):
        """Reinterpolate the frequency series to contain the new indices
        Arguments
        ---------
        freqs : list or numpy array or pd.Series
            array of frequency bins to label the new frequencies
        fmin : float
            instead of an array of freqs, one can provide the starting frequency fmin,
            the fmax and df. 
        fmax: float
            max frequency of the new interpolated signal
        df: float
            frequency steps of the new interpolated signal
        """
        if freqs is None:
            fmin = fmin or self.freq.min()
            fmax = fmax or self.freq.max()
            df = df or self.delta_f
            N = (1 + (fmax-fmin)/df) or len(self.freq)
            freqs = np.linspace(fmin,fmax,int(N))
            
        # Interpolate to the new times
        interp_func = interp1d(self.freq, self.values, kind='linear', fill_value=0, bounds_error=False);
        interp = interp_func(freqs)

        kwargs = {}
        for attr in getattr(self, '_metadata', []):
            kwargs[attr] = getattr(self, attr)

        return self._constructor(interp, index=freqs, **kwargs)

    def read(cls, path, kind=None, **kws):
        kind = (kind or '').lower()
        if not kind:
            # attempt to guess filetype
            ext = os.path.splitext(path)[1].lower().strip('.')
            if ext in ['h5', 'hdf5', 'hdf']:
                kind = 'hdf'
            elif ext in ['txt', 'gz', 'dat', 'csv']:
                kind = 'csv'
            else:
                raise ValueError("unrecognized extension: {}".format(ext))
        if kind in ['hdf', 'csv']:
            read_func = getattr(pd, 'read_{}'.format(kind))
            # get list of arguments accepted by pandas read function in order
            # to filter out extraneous arguments that should go to cls
            read_vars = read_func.__code__.co_varnames
            # define some defaults to ensure we get a Series and not a DataFrame
            read_kws = dict(sep=None, index_col=0, squeeze=True)
            if 'sep' in kws:
                # gymnastics to be able to support `sep = \t` (e.g., when
                # reading a config file)
                kws['sep'] = kws['sep'].encode('raw_unicode_escape').decode('unicode_escape')
            read_kws.update({k: v for k,v in kws.items() if k in read_vars})
            cls_kws = {k: v for k,v in kws.items() if k not in read_vars}
            if kind == 'csv' and 'float_precision' not in read_kws:
                logging.warning("specify `float_precision='round_trip'` or risk "
                                "strange errors due to precission loss")
            return cls(read_func(path, **read_kws), **cls_kws)
        else:
            raise ValueError("unrecognized file kind: {}".format(kind))



class Data(TimeSeries):
    """Container for time-domain strain data from a given GW detector.

    Attributes
    ----------
    ifo : str
        detector identifier (e.g., 'H1' for LIGO Hanford).
    """

    _metadata = ['ifo']

    def __init__(self, *args, ifo=None, **kwargs):
        if ifo is not None:
            ifo = ifo.upper()
        kwargs['name'] = kwargs.get('name', ifo)
        super(Data, self).__init__(*args, **kwargs)
        self.ifo = ifo

    @property
    def _constructor(self):
        return Data

    @property
    def detector(self) -> lal.Detector:
        """:mod:`lal` object containing detector information.
        """
        if self.ifo:
            d = lal.cached_detector_by_prefix[self.ifo]
        else:
            d = None
        return d

    def condition(self, flow=None, fhigh=None, ds=None, scipy_dec=True, t0=None,
                  remove_mean=True, decimate_kws=None, trim=0.25):
        """Condition data.

        Arguments
        ---------
        flow : float
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
        cond_data : Data
            conditioned data object.
        """
        raw_data = self.values
        raw_time = self.index.values

        decimate_kws = decimate_kws or {}

        if t0 is not None:
            ds = int(ds or 1)
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
    def from_data(self, data, flow=None, **kws):
        simple = kws.pop('simple', True)
        fs = kws.pop('fs', 1/getattr(data, 'delta_t', 1))
        kws['nperseg'] = kws.get('nperseg', fs)  # default to 1s segments
        freq, psd = sig.welch(data, fs=fs, **kws)
        p = PowerSpectrum(psd, index=freq)
        if flow:
            p.flatten(flow, simple=simple, inplace=True)
        return p

    def flatten(self, flow, simple=True, inplace=False):
        freq = self.freq
        if inplace:
            psd = self
        else:
            psd = self.copy()
        fref = freq[freq >= flow][0]
        psd_ref = self[fref]
        def get_low_freqs(f, simple):
            if simple:
                return psd_ref
            else:
                return psd_ref + psd_ref*(fref-f)*np.exp(-(fref-f))/3
        psd[freq < flow] = get_low_freqs(freq[freq < flow], simple)
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
    def from_data(self, d, n=None, delta_t=None, method='fd', **kws):
        dt = getattr(d, 'delta_t', delta_t)
        n = n or len(d)
        if method.lower() == 'td':
            rho = sig.correlate(d, d, **kws)
            rho = ifftshift(rho)
            rho = rho[:n] / len(d)
        elif method.lower() == 'fd':
            kws['fs'] = kws.get('fs', 1/dt)
            rho = PowerSpectrum.from_data(d, **kws).to_acf()
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

    def whiten(self, data, drift=1):
        """Whiten stretch of data using ACF.

        Arguments
        ---------
        data : array, TimeSeries
            unwhitened data.

        drift : float, default=1
            factor to apply to noise amplitude before whitening (accounts for
            short-term noise drift)

        Returns
        -------
        w_data : Data
            whitened data.
        """
        if isinstance(data, TimeSeries):
            assert (data.delta_t == self.delta_t)
        # whiten stretch of data using Cholesky factor
        L = self.iloc[:len(data)].cholesky
        w_data = np.linalg.solve(drift*L, data)
        # return same type as input
        if isinstance(data, Data):
            w_data = Data(w_data, index=data.index, ifo=data.ifo)
        elif isinstance(data, TimeSeries):
            w_data = TimeSeries(w_data, index=data.index)
        return w_data
