"""Utilities for handling and manipulating strain data from gravitational-wave
detectors.
"""

__all__ = ['Series', 'TimeSeries', 'FrequencySeries', 'Data',
           'AutoCovariance', 'PowerSpectrum']

from pylab import *
import scipy.signal as sig
import lal
import scipy.linalg as sl
from scipy.interpolate import interp1d
import scipy.signal as ss
import pandas as pd
import h5py
import os
import logging

class Series(pd.Series):
    """ A wrapper of :class:`pandas.Series` with some additional functionality.
    """

    @property
    def _constructor(self):
        return Series

    @classmethod
    def read(cls, path, kind=None, **kws):
        """Load data from disk.
        
        If ``kind`` is `gwosc` assumes input is an strain HDF5 file downloaded
        from `GWOSC <https://www.gw-openscience.org>`_. Otherwise, it is a
        wrapper around :func:`pandas.read_hdf` or :func:`pandas.read_csv`
        functions, for ``kind = 'hdf'`` or ``kind = 'csv'``.

        If ``kind`` is ``None``, guesses filetype from extension.

        Arguments
        ---------
        path : str
            path to file
        kind : str
            kind of file to load: `gwsoc`, `hdf` or `csv`

        Returns
        -------
        series : Series
            series loaded from disk
        """
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
            read_vars = inspect.signature(read_func).parameters.keys()
            # define some defaults to ensure we get a Series and not a DataFrame
            read_kws = dict(sep=None, index_col=0, squeeze=True)
            if 'sep' in kws:
                # gymnastics to be able to support `sep = \t` (e.g., when
                # reading a config file)
                kws['sep'] = kws['sep'].encode('raw_unicode_escape').decode('unicode_escape')
            read_kws.update({k: v for k,v in kws.items() if k in read_vars})
            squeeze = read_kws.pop('squeeze', False)
            cls_kws = {k: v for k,v in kws.items() if k not in read_vars}
            if kind == 'csv' and 'float_precision' not in read_kws:
                logging.warning("specify `float_precision='round_trip'` or risk "
                                "strange errors due to precission loss")
            # squeeze if needed (since sequeeze argument no longer accepted)
            d = read_func(path, **read_kws)
            if squeeze:
                d = d.squeeze("columns")
            return cls(d, **cls_kws)
        else:
            raise ValueError("unrecognized file kind: {}".format(kind))

    _DEF_INTERP_KWS = dict(kind='cubic', fill_value=0, bounds_error=False)

    def interpolate_to_index(self, new_index, **kwargs):
        """Reinterpolate the :class:`Series` to new index.

        Makes use of :func:`scipy.interpolate.interp1d` to which additional
        arguments are passed (by default ``{}``)

        Arguments
        ---------
        new_index : list or numpy array or pd.Series
            new index over which to interpolate

        Returns
        -------
        new_series : Series
            interpolated :class:`Series`
        """
        kws = self._DEF_INTERP_KWS.copy()
        kws.update(**kwargs)
        if any(iscomplex(self.values)):
            re_interp_func = interp1d(self.index, self.values.real, **kws)
            im_interp_func = interp1d(self.index, self.values.imag, **kws)
            interp = re_interp_func(new_index) + 1j*im_interp_func(new_index)
        else:
            interp_func = interp1d(self.index, self.values, **kws)
            interp = interp_func(new_index)
        info = {a: getattr(self, a) for a in getattr(self, '_metadata', [])}
        return self._constructor(interp, index=new_index, **info)
    interpolate_to_index.__doc__ = interpolate_to_index.__doc__.format(_DEF_INTERP_KWS)


class TimeSeries(Series):
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

    def interpolate_to_index(self, time=None, t0=None, duration=None,
                             fsamp=None, **kws):
        """Reinterpolate the :class:`TimeSeries` to new index. Inherits from
        :func:`Series.interpolate_to_index`.

        Arguments
        ---------
        time : list or numpy array or pd.Series
            new times over which to interpolate
        t0 : float
            instead of an array of times, one can provide the start time ``t0``
            the duration and sample rate. If these and ``time`` are provided
            then the duration and sampling rate are taken from the ``time``
            array, setting ``t0`` to be the original one
        duration: float
            duration of the new interpolated signal
        fsamp: float
            sample rate of the new interpolated signal

        Returns
        -------
        new_series : TimeSeries
            interpolated series
        """
        if time is None:
            t0 = t0 or self.time.min()
            duration = duration or (self.time.max() - t0)
            fsamp = fsamp or self.fsamp

            # Create the timing array
            time = np.arange(0.0, duration, 1/fsamp) + t0

            # Make sure we don't include points outside of the index
            if time.max() > self.time.max():
                time = time[time <= self.time.max()]
        elif t0 is not None:
            # Use the time array for the delta_t and duration, but set 
            # the t0 provided
            time = time - time[0] + t0
        return super(TimeSeries, self).interpolate_to_index(time, **kws)


class FrequencySeries(Series):
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

    def interpolate_to_index(self, freq=None, fmin=None, fmax=None,
                             delta_f=None, **kws):
        """Reinterpolate the :class:`FrequencySeries` to new index. Inherits
        from :func:`Series.interpolate_to_index`.

        Arguments
        ---------
        freq : list or numpy array or pd.Series
            array of frequency bins to label the new frequencies
        fmin : float
            instead of an array of frequencies, one can provide the starting
            frequency ``fmin``, the highest frequency ``fmax`` and the
            frequency spacing ``delta_f``. 
        fmax: float
            max frequency of the new interpolated signal
        delta_f: float
            frequency steps of the new interpolated signal

        Returns
        -------
        new_series : FrequencySeries
            interpolated series
        """
        if freq is None:
            fmin = fmin or self.freq.min()
            fmax = fmax or self.freq.max()
            delta_f = delta_f or self.delta_f
            N = (1 + (fmax-fmin)/delta_f) or len(self.freq)
            freq = np.linspace(fmin, fmax, int(N))
        return super(FrequencySeries, self).interpolate_to_index(freq, **kws)

class Data(TimeSeries):
    """Container for time-domain strain data from a given GW detector.

    Attributes
    ----------
    ifo : str
        detector identifier (e.g., 'H1' for LIGO Hanford).
    info : dict
        optional additional information, e.g., to identify data provenance.
    """

    _metadata = ['ifo', 'info']

    def __init__(self, *args, ifo=None, info=None,  **kwargs):
        if ifo is not None:
            ifo = ifo.upper()
        kwargs['name'] = kwargs.get('name', ifo)
        super(Data, self).__init__(*args, **kwargs)
        self.ifo = ifo
        self.info = info or {}

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

    def condition(self, t0=None, ds=None, flow=None, fhigh=None, trim=0.25,
                  digital_filter=False, remove_mean=True, decimate_kws=None,
                  scipy_dec=None, slice_left = None, slice_right = None):
        """Condition data.

        Arguments
        ---------
        flow : float
            lower frequency for high passing.
        fhigh : float
            higher frequency for low passing.
        ds : int
            decimation factor for downsampling.
        digital_filter : bool
            apply digital antialiasing filter by discarding Fourier components
            higher than Nyquist; otherwise, filter through
            :func:`scipy.signal.decimate`. Defaults to False.
        t0 : float
            target time to be preserved after downsampling.
        remove_mean : bool
            explicitly remove mean from time series after conditioning.
        decimate_kws : dict
            options for decimation function.
        trim : float
            fraction of data to trim from edges after conditioning, to avoid
            spectral issues if filtering.
        slice_left : float
            number of seconds before t0 to slice the strain data, e.g. to avoid NaNs
        slice_right : float
            number of seconds after t0 to slice the strain data, e.g. to avoid NaNs

        Returns
        -------
        cond_data : Data
            conditioned data object.
        """
        if slice_left is not None and slice_right is None:
            self = self[t0-slice_left:]
        elif slice_left is None and slice_right is not None:
            self = self[:t0+slice_right]
        elif slice_left is not None and slice_right is not None:
            self = self[t0-slice_left:t0+slice_right]

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
        if scipy_dec is not None:
            logging.warn("scipy_dec arg is deprecated, use digital_filter")
            digital_filter = not scipy_dec

        if ds and ds > 1:
            if digital_filter:
                # fft data
                w = ss.windows.tukey(len(cond_data), trim)
                cond_data_fd = np.fft.rfft(cond_data*w)
                freq = np.fft.rfftfreq(len(cond_data), self.delta_t)
                # throw away frequencies
                cond_data_fd[freq > fny/ds] = 0
                # ifft and downsample
                cond_data = np.fft.irfft(cond_data_fd)
                cond_data = cond_data[::ds]
            else:
                cond_data = sig.decimate(cond_data, ds, zero_phase=True,
                                         **decimate_kws)
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
        """Estimate ACF from data, see :meth:`AutoCovariance.from_data`.
        """
        return AutoCovariance.from_data(self, **kws)

    def get_psd(self, **kws):
        """Estimate PSD from data, see :meth:`PowerSpectrum.from_data`.
        """
        return PowerSpectrum.from_data(self, **kws)


class PowerSpectrum(FrequencySeries):
    """Contains and manipulates power spectral densities, a special kind of
    :class:`FrequencySeries`.
    """

    def __init__(self, *args, delta_f=None, **kwargs):
        super(PowerSpectrum, self).__init__(*args, **kwargs)
        if delta_f is not None:
            self.index = arange(len(self))*delta_f

    @property
    def _constructor(self):
        return PowerSpectrum

    @classmethod
    def from_data(cls, data, flow=None, fhigh=None, patch_level=None, **kws):
        """Estimate :class:`PowerSpectrum` from time domain data using Welch's
        method.

        Arguments
        ---------
        data : Data, array
            data time series.
        flow : float, None
            optional lower frequency at which to taper PSD via
            :meth:`PowerSpectrum.flatten`. Defaults to None (i.e., no
            flattening).
        fhigh : float, None
            optional higher frequency at which to taper PSD via
            :meth:`PowerSpectrum.flatten`. Defaults to None (i.e., no
            flattening).
        smooth : bool
            if flattening PSD, whether to do so smoothly. Defaults to False.
        \*\*kws :
            additional keyword arguments passed to :func:`scipy.signal.Welch`.

        Returns
        -------
        psd : PowerSpectrum
            power specturm estimate.
        """
        fs = kws.pop('fs', 1/getattr(data, 'delta_t', 1))
        kws['nperseg'] = kws.get('nperseg', fs)  # default to 1s segments
        kws['average'] = kws.get('average', 'median') # default to median-averaged, not mean-averaged to handle outliers.
        freq, psd = sig.welch(data, fs=fs, **kws)
        p = cls(psd, index=freq)
        if flow is not None or fhigh is not None:
            p.flatten(flow=flow, fhigh=fhigh, patch_level=patch_level, inplace=True)
        return p

    @classmethod
    def from_lalsimulation(cls, func, freq, flow=0, **kws):
        """Obtain :class:`PowerSpectrum` from LALSimulation function.

        Arguments
        ---------
        func : str, builtin_function_or_method
            LALSimulation PSD function, or name thereof (e.g.,
            ``SimNoisePSDaLIGOZeroDetHighPower``).
        freq : array
            frequencies over which to evaluate PSD.
        flow : float
            lower frequency threshold for padding: PSD will be flattened below
            this value.
        \*\*kw : 
            additional arguments passed to padding function
            :meth:`PowerSpectrum.flatten`.

        Returns
        -------
        psd : PowerSpectrum
            power spectrum frequency series.
        """
        if isinstance(func, str):
            import lalsimulation as lalsim
            func = getattr(lalsim, func)
        f_ref = freq[argmin(abs(freq - flow))]
        p_ref = func(f_ref)
        def get_psd_bin(f):
            if f > flow:
                return func(f)
            else:
                return cls._pad_low_freqs(f, f_ref, p_ref)
        psd = cls(vectorize(get_psd_bin)(freq), index=freq)
        return psd
        
    @staticmethod
    def _pad_low_freqs(f, f_ref, psd_ref):
        # made up function to taper smoothly
        return psd_ref + psd_ref*(f_ref-f)*np.exp(-(f_ref-f))/3

    def flatten(self, flow, fhigh=None, patch_level=None, inplace=False):
        """Modify PSD at low or high frequencies so that it flattens to a
        constant.

        Arguments
        ---------
        flow : float
            low frequency threshold.
        fhigh : float
            high frequency threshold (def., `None`).
        patch_level : float,tuple
            value with which to patch PSD; if a tuple, then these ``(psd_low,
            psd_high)`` values are used at the low and high ends respectively;
            if a float, the same value is used in both ends; if `None`, will
            patch with 10x the maximum PSD value in the respective patched
            region.
        inplace : bool
            modify PSD in place; otherwise, returns copy. Defaults to `False`.

        Returns
        -------
        psd : PowerSpectrum, None
            returns PSD only if not ``inplace``.
        """
        # copy array or operate in place
        psd = self if inplace else self.copy()
        # determine highest frequency
        f = psd.freq
        flow = max(flow or min(f), min(f))
        fhigh = min(fhigh or max(f), max(f))
        # create tuple (patch_level_low, patch_level_high)
        if patch_level is None:
            patch_level = (10*max(psd[f < flow]), 10*max(psd[f >= fhigh]))
        else:
            try:
                patch_level[1]
            except Exception:
                patch_level = (patch_level, patch_level)
        # patch low frequencies
        psd[f < flow] = patch_level[0]
        # patch high frequencies
        psd[f > fhigh] = patch_level[1]
        if not inplace:
            return psd

    def to_acf(self):
        """Return cyclic ACF corresponding to PSD obtained by inverse Fourier
        transforming.

        Returns
        -------
        acf : AutoCovariance
            autocovariance function.
        """
        rho = 0.5*np.fft.irfft(self) / self.delta_t
        return AutoCovariance(rho, delta_t=self.delta_t)


class AutoCovariance(TimeSeries):
    """Contains and manipulates autocovariance functions, a special kind of
    :class:`TimeSeries`.
    """

    def __init__(self, *args, delta_t=None, **kwargs):
        super(AutoCovariance, self).__init__(*args, **kwargs)
        if delta_t is not None:
            self.index = arange(len(self))*delta_t

    @property
    def _constructor(self):
        return AutoCovariance

    @classmethod
    def from_data(cls, d, n=None, delta_t=None, method='fd', **kws):
        """Estimate :class:`AutoCovariance` from time domain data using Welch's
        method by default.

        Arguments
        ---------
        d : Data, array
            data time series.
        n : int
            length of output ACF. Defaults to ``len(d)``.
        delta_t : float
            time-sample spacing, necessary if `d` is a simple array with no
            associated timing information. Defaults to `d.delta_t` if `d` is
            :class:`Data`.
        method : str
            whether to use Welch's method (``'fd'``), or simply auto-correlate
            the data (``'td'``). The latter is highly discouraged and will
            result in a warning. Defaults to `fd`.
        \*\*kws :
            additional keyword arguments passed to :meth:`PowerSpectrum.fom_data`.

        Returns
        -------
        acf : AutoCovariance
            estimate of the autocovariance function.
        """
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
        return cls(rho, delta_t=dt)

    def to_psd(self) -> PowerSpectrum:
        """Returns corresponding :class:`PowerSpectrum`, obtained by Fourier
        transforming ACF.

        Returns
        -------
        psd : PowerSpectrum
            power spectral density.
        """
        # acf = 0.5*np.fft.irfft(psd) / delta_t
        psd = 2 * self.delta_t * abs(np.fft.rfft(self))
        freq = np.fft.rfftfreq(len(self), d=self.delta_t)
        return PowerSpectrum(psd, index=freq)

    @property
    def matrix(self):
        """Covariance matrix built from ACF, :math:`C_{ij} = \\rho(|i-j|)`.
        """
        return sl.toeplitz(self)

    @property
    def cholesky(self):
        """Cholesky factor :math:`L` of covariance matrix :math:`C = L^TL`.
        """
        if getattr(self, '_cholesky', None) is None:
            self._cholesky = linalg.cholesky(self.matrix)
        return self._cholesky

    def compute_snr(self, x, y=None):
        """Efficiently compute the signal-to_noise ratio
        :math:`\\mathrm{SNR} = \left\langle x \mid y \\right\\rangle / \\sqrt{\\left\langle x \mid x \\right\\rangle}`,
        where the inner product is defined by :math:`\left\langle x \mid y \\right\\rangle \\equiv x_i C^{-1}_{ij} y_j`.
        This is internally computed using Cholesky factors to speed up the
        computation.

        If `x` is a signal and `y` is noisy data, then this is the matched
        filter SNR; if both of them are a template, then this is the optimal
        SNR.

        Arguments
        ---------
        x : array
            target time series.
        y : array
            reference time series. Defaults to `x`.

        Returns
        -------
        snr : float
            signal-to-noise ratio
        """

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
        w_data = sl.solve_triangular(L, data, lower=True)
        # return same type as input
        if isinstance(data, Data):
            w_data = Data(w_data, index=data.index, ifo=data.ifo)
        elif isinstance(data, TimeSeries):
            w_data = TimeSeries(w_data, index=data.index)
        return w_data
