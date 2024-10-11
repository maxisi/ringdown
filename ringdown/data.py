"""Utilities for handling and manipulating strain data from gravitational-wave
detectors.
"""

__all__ = ['Series', 'TimeSeries', 'FrequencySeries', 'Data',
           'AutoCovariance', 'PowerSpectrum']

import numpy as np
import scipy.signal as sig
import lal
import scipy.linalg as sl
from scipy.interpolate import interp1d
import scipy.signal as ss
import pandas as pd
import h5py
import os
import logging
import inspect
from . import utils
from typing import Callable


class Series(pd.Series):
    """ A wrapper of :class:`pandas.Series` with some additional functionality.
    """

    @property
    def _constructor(self):
        return Series

    @classmethod
    def read(cls, path: str, kind: str | None = None,
             channel: str | None = None, **kws):
        """Load data from disk.

        If ``kind`` is `gwosc` assumes input is an strain HDF5 file downloaded
        from `GWOSC <https://www.gw-openscience.org>`_.  If ``kind`` is `frame`
        and the keyword `channel` is given, then attempt to load the given
        path(s) using gwpy's frame file reading.  Otherwise, it is a wrapper
        around :func:`pandas.read_hdf` or :func:`pandas.read_csv` functions,
        for ``kind = 'hdf'`` or ``kind = 'csv'``.

        If ``kind`` is ``None``, guesses filetype from extension.

        Arguments
        ---------
        path : str
            path to file, or None if fetching remote data.
        kind : str
            kind of file to load: `gwsoc`, `hdf` or `csv`.
        channel : str
            channel name to use when reading GW frame files (not required
            otherwise).

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
            if path is not None and os.path.exists(path):
                logging.info("loading GWOSC file from disk")
                # read GWOSC HDF5 file from disk
                with h5py.File(path, 'r') as f:
                    t0 = f['meta/GPSstart'][()]
                    T = f['meta/Duration'][()]
                    h = f['strain/Strain'][:]
                dt = T/len(h)
                time = t0 + dt*np.arange(len(h))
                # get valid arguments and form class
                meta = {a: kws.get(a, None) for a in getattr(cls, '_meta', [])}
                return cls(h, index=time, **meta)
            else:
                raise FileNotFoundError("file not found: {}".format(path))

        elif kind == 'frame':
            if channel is None:
                raise KeyError('channel must be specified for frame files')
            try:
                from gwpy.timeseries import TimeSeries
            except ModuleNotFoundError:
                raise ImportError("missing optional dependency 'gwpy'; "
                                  "use pip or conda to install it")
            ts = TimeSeries.read(path, channel)
            # GWpy puts units on its times, we remove them by redefining the
            # index
            return cls(ts.value, index=np.array(ts.times), **kws)

        elif kind in ['hdf', 'csv']:
            read_func = getattr(pd, 'read_{}'.format(kind))
            # get list of arguments accepted by pandas read function in order
            # to filter out extraneous arguments that should go to cls
            read_vars = inspect.signature(read_func).parameters.keys()
            # define defaults to ensure we get a Series and not a DataFrame
            read_kws = dict(sep=None, index_col=0, squeeze=True)
            if 'sep' in kws:
                # gymnastics to be able to support `sep = \t` (e.g., when
                # reading a config file)
                kws['sep'] = kws['sep'].encode(
                    'raw_unicode_escape').decode('unicode_escape')
            read_kws.update({k: v for k, v in kws.items() if k in read_vars})
            squeeze = read_kws.pop('squeeze', False)
            cls_kws = {k: v for k, v in kws.items() if k not in read_vars}
            if kind == 'csv' and 'float_precision' not in read_kws:
                logging.warning("specify `float_precision='round_trip'` or "
                                "risk strange errors due to precision loss")
            if kind == 'csv':
                read_kws['header'] = kws.get('header', None)
            # squeeze if needed (since squeeze argument no longer accepted)
            d = read_func(path, **read_kws)
            if squeeze and not isinstance(d, pd.Series):
                d = d.squeeze("columns")
            return cls(d, **cls_kws)
        else:
            raise ValueError("unrecognized file kind: {}".format(kind))

    @classmethod
    def fetch(cls, channel: str, start: float | None = None,
              end: float | None = None, t0: float | None = None,
              seglen: float | None = None, frametype: str | None = None,
              **kws):
        """Download open data or discover data using NDS2 (requires GWpy).

        Uses GWpy's :meth:`gwpy.timeseries.TimeSeries.fetch_open_data` or
        :meth:`gwpy.timeseries.TimeSeries.get` to download data. If `channel`
        is 'GWOSC', then it will download open data from GWOSC; otherwise, it
        will attempt to discover remote or local data using NDS2.

        Arguments
        ---------
        channel : str
            channel name or 'GWOSC' for public data.
        start : float
            start GPS time.
        end : float
            end GPS time.
        t0 : float
            center time of segment to fetch (alternative to start and end).
        seglen : float
            length of segment to fetch (alternative to start and end).
        frametype : str
            specify frame type to facilitate NDS frame discovery.
        **kws :
            additional keyword arguments passed to GWpy's fetch function.

        Returns
        -------
        series : Series
            series downloaded from GWOSC or NDS.
        """
        # first check that we have GWpy, which is an optional dependency
        try:
            from gwpy.timeseries import TimeSeries
        except ModuleNotFoundError:
            raise ImportError("missing optional dependency 'gwpy'; "
                              "use pip or conda to install it")

        # validate time input
        if start is None and end is None and \
           t0 is not None and seglen is not None:
            start = t0 - seglen/2
            end = t0 + seglen/2
            logging.info(f"fetching {seglen} s long segment centered on {t0}"
                         f" [{start}, {end}]")
        elif start is None or end is None:
            raise ValueError("must provide start and end times,"
                             " or t0 and seglen")

        # download data
        if channel.lower() == 'gwosc':
            logging.info("fetching open data from GWOSC")
            ifo = kws.get('ifo', None)
            if ifo is None:
                raise ValueError("must provide ifo to fetch from GWOSC")
            attrs = {a: kws.pop(a, None) for a in getattr(cls, '_meta', [])}
            d = TimeSeries.fetch_open_data(ifo, start, end, **kws)
            # GWpy puts units on its times, we remove them
            return cls(d.value, index=np.array(d.times), **attrs)
        else:
            logging.info("fetching remote or local data using GWpy")
            attrs = {a: kws.pop(a, None) for a in getattr(cls, '_meta', [])}
            d = TimeSeries.get(channel, start, end, frametype=frametype, **kws)
            # GWpy puts units on its times, we remove them
            return cls(d.value, index=np.array(d.times), **attrs)

    @classmethod
    def load(cls, path: str | None = None, channel: str | None = None, **kws):
        """Universal load function to read data from disk or discover GWOSC/NDS
        data using GWpy.

        Only one of `path` or `channel` can be provided. If a `path` is
        provided, it will attempt to read the data from disk using
        :meth:`Series.read`; if `channel` is provided, it will attempt to
        discover and fetch the data using :meth:`Series.fetch`, which leverages
        GWpy's NDS2 interface.

        A special case of the latter is when `channel` is 'GWOSC' (case
        insensitive); if so, it will download open data from GWOSC.

        Arguments
        ---------
        path : str
            path to file, or None if fetching remote data
        channel : str
            channel name or 'GWOSC' for public data (case insensitive)
        **kws :
            additional keyword arguments passed to :meth:`Series.read` or
            :meth:`Series.fetch`.
        """
        if path is None and channel is not None:
            ts = cls.fetch(channel, **kws)
        elif path is not None:
            ts = cls.read(path, channel=channel, **kws)
        else:
            raise ValueError("must provide either path or channel")
        # record data provenance in series attrs (note that attrs is a property
        # of pandas.Series, we don't need to re-create it)
        info = dict(path=path, channel=channel)
        info.update(kws)
        ts.attrs.update(dict(load=info))
        return ts

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
        if any(np.iscomplex(self.values)):
            re_interp_func = interp1d(self.index, self.values.real, **kws)
            im_interp_func = interp1d(self.index, self.values.imag, **kws)
            interp = re_interp_func(new_index) + 1j*im_interp_func(new_index)
        else:
            interp_func = interp1d(self.index, self.values, **kws)
            interp = interp_func(new_index)
        attrs = {a: getattr(self, a, None) for a in getattr(self, '_meta', [])}
        return self._constructor(interp, index=new_index, **attrs)
    interpolate_to_index.__doc__ = \
        interpolate_to_index.__doc__.format(_DEF_INTERP_KWS)


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
    def f_samp(self) -> float:
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

    @property
    def epoch(self) -> float:
        """Time of first sample."""
        return self.index[0]

    def to_frequency_series(self):
        """Fourier transform time series to frequency series.

        Returns
        -------
        freq_series : FrequencySeries
            frequency series.
        """
        freq = np.fft.rfftfreq(len(self), self.delta_t)
        data = np.fft.rfft(self) * self.delta_t
        return FrequencySeries(data, index=freq, name=self.name)

    def interpolate_to_index(self, time=None, t0=None, duration=None,
                             delta_t=None, fsamp=None, **kws):
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
            if fsamp is None and delta_t is not None:
                fsamp = 1/delta_t
            elif fsamp is None:
                raise ValueError("must provide only one of delta_t or fsamp")

            t0 = t0 or self.time.min()
            duration = duration or (self.time.max() - t0)
            fsamp = fsamp or self.f_samp

            # Create the timing array
            time = np.arange(0.0, duration, 1/fsamp) + t0

            # Make sure we don't include points outside of the index
            if time.max() > self.time.max():
                time = time[time <= self.time.max()]
        elif t0 is not None:
            # Use the time array for the delta_t and duration, but set
            # the t0 provided
            time = time - time[0] + t0
        return super().interpolate_to_index(time, **kws)


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
        # sampling rate is the inverse of the sampling frequency
        return 1/self.f_samp

    @property
    def f_samp(self) -> float:
        """Sampling frequency (`1/delta_t`)."""
        # sampling frequency is twice the Nyquist frequency
        return 2*self.freq[-1]

    @property
    def duration(self) -> float:
        """Time series duration (time spanned between first and last samples).
        """
        return 1/self.delta_f

    @property
    def freq(self) -> pd.Index:
        """Frequency stamps."""
        return self.index

    def to_time_series(self, epoch=0.):
        """Inverse Fourier transform frequency series to time series.

        Returns
        -------
        time_series : TimeSeries
            time series.
        """
        data = np.fft.irfft(self.mask(self.isna(), 0.)) / self.delta_t
        time = np.arange(len(data)) * self.delta_t + epoch
        return TimeSeries(data, index=time, name=self.name)

    def interpolate_to_index(self, freq: np.ndarray | None = None,
                             delta_f: float | None = None,
                             f_min: float | None = None,
                             f_max: float | None = None,
                             log: bool = False, **kws):
        """Interpolate the :class:`FrequencySeries` to new index. Inherits
        from :func:`Series.interpolate_to_index`.

        If no frequency arguments are specified, it will interpolate to a a
        frequency array with the same span as the original, but enforcing
        uniform spacing based on the difference between the first two frequency
        samples.

        Arguments
        ---------
        freq : list or numpy array or pd.Series
            array of frequency bins to label the new frequencies
        delta_f: float
            frequency steps of the new interpolated signal
        f_min : float
            instead of an array of frequencies, one can provide the starting
            frequency ``f_min``, the highest frequency ``f_max`` and the
            frequency spacing ``delta_f``.
        f_max : float
            max frequency of the new interpolated signal
        log : bool
            interpolate in log-log space [EXPERIMENTAL] (default False)

        Returns
        -------
        new_series : FrequencySeries
            interpolated series
        """
        f_min_orig = self.freq.min()
        f_max_orig = self.freq.max()
        if freq is None:
            # construct frequency index with uniform spacing
            f_min = f_min_orig if f_min is None else f_min
            f_max = f_max_orig if f_max is None else f_max
            delta_f = delta_f or self.delta_f
            n = int((f_max - f_min) / delta_f) + 1
            freq = np.arange(n)*delta_f + f_min

        if min(freq) < f_min_orig or max(freq) > f_max_orig:
            logging.warning("extrapolating frequencies "
                            f"[{f_min_orig}, {f_max_orig}] -> "
                            f"[{min(freq)}, {max(freq)}]")

        if log:
            logging.info("log-log interpolation is experimental")
            # construct loglog representation of self
            y = np.log(self)
            y.index = np.log(self.index)
            # interpolate in loglog space
            yinterp = np.exp(y.interpolate_to_index(np.log(freq), **kws))
            yinterp.index = freq
            return yinterp
        else:
            return super().interpolate_to_index(freq, **kws)


class Data(TimeSeries):
    """Container for time-domain strain data from a given GW detector.

    Attributes
    ----------
    ifo : str
        detector identifier (e.g., 'H1' for LIGO Hanford).
    attrs : dict
        optional additional information, e.g., to identify data provenance.
    """

    _meta = ['ifo', 'attrs']

    def __init__(self, *args, ifo=None, attrs=None,  **kwargs):
        if ifo is not None:
            ifo = ifo.upper()
        kwargs['name'] = kwargs.get('name', ifo)
        super().__init__(*args, **kwargs)
        if len(args) == 0:
            args = [None]
        self.ifo = ifo or getattr(args[0], 'ifo', None)
        self.attrs = attrs or getattr(args[0], 'attrs', {}) or {}

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

    def condition(self, t0: float | None = None,
                  ds: int | None = None,
                  f_min: float | None = None,
                  f_max: float | None = None,
                  trim: float = 0.25,
                  digital_filter: bool = True,
                  remove_mean: bool = True,
                  decimate_kws: dict | None = None,
                  slice_left: float | None = None,
                  slice_right: float | None = None):
        """Condition data.

        Arguments
        ---------
        t0 : float
            target time to be preserved after downsampling.
        ds : int
            decimation factor for downsampling.
        f_min : float
            lower frequency for high passing.
        f_max : float
            higher frequency for low passing.
        trim : float
            fraction of data to trim from edges after conditioning, to avoid
            spectral issues if filtering (default 0.25).
        digital_filter : bool
            apply digital antialiasing filter by discarding Fourier components
            higher than Nyquist; otherwise, filter through
            :func:`scipy.signal.decimate`.(default True).
        remove_mean : bool
            explicitly remove mean from time series after conditioning (default
            True).
        decimate_kws : dict
            options for decimation function.
        slice_left : float
            number of seconds before t0 to slice the strain data, e.g. to avoid
            NaNs
        slice_right : float
            number of seconds after t0 to slice the strain data, e.g. to avoid
            NaNs

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
            if t0 < raw_time[0] or t0 > raw_time[-1]:
                raise ValueError(f"t0 must be within the time series: {t0} "
                                 f"not in [{raw_time[0]}, {raw_time[-1]}]")
            ds = int(ds or 1)
            i = np.argmin(abs(raw_time - t0))
            raw_time = np.roll(raw_time, -(i % ds))
            raw_data = np.roll(raw_data, -(i % ds))

        fny = 0.5/(raw_time[1] - raw_time[0])
        # Filter
        if f_min and not f_max:
            b, a = sig.butter(4, f_min/fny, btype='highpass', output='ba')
        elif f_max and not f_min:
            b, a = sig.butter(4, f_max/fny, btype='lowpass', output='ba')
        elif f_min and f_max:
            b, a = sig.butter(4, (f_min/fny, f_max/fny), btype='bandpass',
                              output='ba')

        if f_max == fny:
            logging.warning("f_max is at Nyquist frequency but filter will "
                            "be applied anyway; to prevent this, set f_max to"
                            " None (default)")

        if f_min or f_max:
            cond_data = sig.filtfilt(b, a, raw_data)
        else:
            cond_data = raw_data

        # Decimate
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
            cond_data -= np.mean(cond_data)

        return Data(cond_data, index=cond_time, ifo=self.ifo)

    def get_acf(self, **kws):
        """Estimate ACF from data, see :meth:`AutoCovariance.from_data`.
        """
        return AutoCovariance.from_data(self, **kws)

    def get_psd(self, **kws):
        """Estimate PSD from data, see :meth:`PowerSpectrum.from_data`.
        """
        return PowerSpectrum.from_data(self, **kws)

    @classmethod
    def from_psd(cls, psd: 'PowerSpectrum', **kws):
        """Generate data from a given PSD.

        Arguments
        ---------
        psd : PowerSpectrum
            power spectral density.

        Returns
        -------
        data : Data
            time series data.
        """
        noise_td = PowerSpectrum(psd).draw_noise_td(**kws)
        return Data(noise_td)


class PowerSpectrum(FrequencySeries):
    """Contains and manipulates power spectral densities, a special kind of
    :class:`FrequencySeries`.
    """

    _meta = ['ifo', 'attrs']

    def __init__(self, *args, delta_f=None, ifo=None, attrs=None,
                 fill_power_of_two=True, **kwargs):
        """Initialize power spectral density.

        Arguments
        ---------
        delta_f : float
            frequency spacing.
        ifo : str
            detector identifier (e.g., 'H1' for LIGO Hanford).
        attrs : dict
            optional additional information, e.g., to identify data provenance.
        complete_power_of_two : bool
            ensure that the power spectrum is complete up to the next power of
            two.
        """
        if ifo is not None:
            ifo = ifo.upper()
        kwargs['name'] = kwargs.get('name', ifo)
        if len(args) == 1 and np.ndim(args[0]) == 2:
            if 'index' not in kwargs:
                # interpret input as (freq, psd)
                a = np.array(args[0])
                kwargs['index'] = a[:, 0]
                args = [a[:, 1]]
        super().__init__(*args, **kwargs)
        if delta_f is not None:
            self.index = np.arange(len(self))*delta_f
        if len(args) == 0:
            args = [None]
        self.ifo = ifo or getattr(args[0], 'ifo', None)
        self.attrs = attrs or getattr(args[0], 'attrs', {}) or {}
        if fill_power_of_two and not self.empty:
            logging.info("completing power spectrum to next power of two")
            self.fill_power_of_two()
        self.sort_index(inplace=True, ascending=True)

    @property
    def _constructor(self):
        return PowerSpectrum

    def gate(self, max_dynamic_range=6, inplace=False):
        """Gate PSD to avoid numerical issues.

        Arguments
        ---------
        max_dynamic_range : float
            maximum dynamic range in decades to allow in the PSD.

        Returns
        -------
        psd : PowerSpectrum
            gated power spectrum.
        """
        log_psd = np.log10(self)
        min_log_psd = log_psd.min()
        max_log_psd = max_dynamic_range + min_log_psd
        if inplace:
            self.iloc[log_psd > max_log_psd] = 10**max_log_psd
        else:
            p = self.copy()
            p.iloc[log_psd > max_log_psd] = 10**max_log_psd
            return p

    def fill_power_of_two(self) -> None:
        """Ensure that the power spectrum is complete up to the next power of
        two frequency.
        """
        if self.empty:
            return self
        fmax = self.freq[-1]
        new_fmax = utils.np2(fmax)
        if fmax % 2 and np.isclose(new_fmax - fmax, self.delta_f):
            self[new_fmax] = self.iloc[-1]

    def fill_low_frequencies(self, f_min: float = 0.,
                             fill_value: float | None = None,
                             **kws) -> 'PowerSpectrum':
        """Complete low frequencies in power spectrum, extending all the
        way down to `f_min`, which defaults to 0. If `fill_value` is not
        provided, it will be set to 10 times the maximum PSD value.

        If f_min is not below the lowest frequency in the PSD, nothing
        is done and the PSD is returned as is.

        Additional arguments are passed to
        :meth:`PowerSpectrum.interpolate_to_index`.

        Arguments
        ---------
        f_min : float
            lower frequency threshold.
        fill_value : float
            value with which to patch PSD below `f_min`.
        ** kws :
            additional keyword arguments passed to
            :meth:`PowerSpectrum.interpolate_to_index`.

        Returns
        -------
        psd : PowerSpectrum
            power spectrum with low frequencies completed.
        """
        if f_min >= self.freq[0]:
            logging.info("no need to complete low PSD frequencies")
            return self

        if fill_value is None:
            logging.info("completing low frequencies with 10x max PSD")
            fill_value = 10*self.max()

        f = np.arange(f_min, self.freq[-1] + self.delta_f, self.delta_f)
        return self.interpolate_to_index(f, fill_value=fill_value, **kws)

    @classmethod
    def from_data(cls, data: Data | np.ndarray,
                  f_min: float | None = None, f_max: float | None = None,
                  fill_value: float | tuple | None = None, **kws):
        """Estimate :class:`PowerSpectrum` from time domain data using Welch's
        method.

        Arguments
        ---------
        data : Data, array
            data time series.
        f_min : float, None
            optional lower frequency at which to taper PSD via
            :meth:`PowerSpectrum.patch`. Defaults to None (i.e., no
            patching).
        f_max : float, None
            optional higher frequency at which to taper PSD via
            :meth:`PowerSpectrum.patch`. Defaults to None (i.e., no
            patching).
        fill_value : float, tuple, None
            value with which to patch PSD; if a tuple, then these ``(psd_low,
            psd_high)`` values are used at the low and high ends respectively;
            if a float, the same value is used in both ends; if `None`, will
            patch with 10x the maximum PSD value in the respective patched
            region.
        **kws :
            additional keyword arguments passed to :func:`scipy.signal.Welch`.

        Returns
        -------
        psd : PowerSpectrum
            power specturm estimate.
        """
        # check some deprecated arguments
        patch_level = kws.pop('patch_level', None)
        if patch_level is not None:
            logging.warning("patch_level argument is deprecated;"
                            " use fill_value instead")
            fill_value = patch_level
        flow = kws.pop('flow', None)
        if flow is not None:
            logging.warning("flow argument is deprecated; use f_min instead")
            f_min = flow
        fhigh = kws.pop('fhigh', None)
        if fhigh is not None:
            logging.warning("fhigh argument is deprecated; use f_max instead")
            f_max = fhigh
        # get sampling rate if not provided
        fs = kws.pop('fs', kws.pop('f_samp', 1/getattr(data, 'delta_t', 1)))
        # default to 1s segments
        kws['nperseg'] = kws.get('nperseg', fs)
        # default to median-averaged, not mean-averaged to handle outliers.
        kws['average'] = kws.get('average', 'median')
        freq, psd = sig.welch(data, fs=fs, **kws)
        _meta = {a: getattr(data, a, None) for a in getattr(data, '_meta', [])}
        p = cls(psd, index=freq, **_meta)
        if f_min is not None or f_max is not None:
            p.patch(f_min=f_min, f_max=f_max, fill_value=fill_value,
                    in_place=True)
        return p

    @classmethod
    def from_lalsimulation(cls, func: str | Callable,
                           freq: np.ndarray | None = None,
                           f_min: float = 0, f_max: float | None = None,
                           delta_f: float | None = None,
                           fill_value: float | None = None, **kws):
        """Obtain :class:`PowerSpectrum` from LALSimulation function.

        Arguments
        ---------
        func : str, builtin_function_or_method
            LALSimulation PSD function, or name thereof (e.g.,
            ``SimNoisePSDaLIGOZeroDetHighPower``).
        freq : array
            frequencies over which to evaluate PSD.
        f_min : float
            lower frequency threshold for padding: PSD will be patched below
            this value.
        delta_f : float
            frequency spacing (required if ``freq`` is not provided).
        fill_value : float
            value with which to patch PSD below ``f_min``.
        **kw :
            additional arguments passed to padding function
            :meth:`PowerSpectrum.patch`.

        Returns
        -------
        psd : PowerSpectrum
            power spectrum frequency series.
        """
        if isinstance(func, str):
            import lalsimulation as lalsim
            if not hasattr(lalsim, func):
                raise ValueError(f"unrecognized PSD name: {func}")
            func = getattr(lalsim, func)
        if freq is None:
            if f_max is None:
                raise ValueError("must provide f_max if not freq")
            if delta_f is None:
                raise ValueError("must provide delta_f if not freq")
            freq = np.arange(0, f_max+delta_f, delta_f)
        f_ref = freq[np.argmin(abs(freq - f_min))]
        p_ref = func(f_ref) if fill_value is None else fill_value

        def get_psd_bin(f):
            if f > f_min:
                return func(f)
            else:
                return cls._patch_low_freqs(f, f_ref, p_ref)
        return cls(np.vectorize(get_psd_bin)(freq), index=freq, **kws)

    @staticmethod
    def _patch_low_freqs(f, f_ref, psd_ref):
        # made up function to taper smoothly
        return psd_ref + psd_ref*(f_ref-f)*np.exp(-(f_ref-f))/3

    def patch(self, f_min, f_max=None, patch_level=None, in_place=False,
              fill_value=None):
        """Modify PSD at low or high frequencies so that it patches to a
        constant.

        Arguments
        ---------
        f_min : float
            low frequency threshold.
        f_max : float
            high frequency threshold (def., `None`).
        patch_level : float,tuple
            value with which to patch PSD; if a tuple, then these ``(psd_low,
            psd_high)`` values are used at the low and high ends respectively;
            if a float, the same value is used in both ends; if `None`, will
            patch with 10x the maximum PSD value in the respective patched
            region.
        fill_value : float
            an alias for patch level for consistency with interp1d
        in_place : bool
            modify PSD in place; otherwise, returns copy. Defaults to `False`.

        Returns
        -------
        psd : PowerSpectrum, None
            returns PSD only if not ``in_place``.
        """
        # copy array or operate in place
        psd = self if in_place else self.copy()
        # determine highest frequency
        f = psd.freq
        if f_min < min(f):
            logging.warning("f_min below PSD range; no patching appplied; "
                            "use `interpolate_to_index` or "
                            "`complete_low_frequencies` to extend PSD")
        if f_max is not None and f_max > max(f):
            logging.warning("f_max above PSD range; no patching appplied; "
                            "use `interpolate_to_index` to extend PSD")
        f_min = max(f_min or min(f), min(f))
        f_max = min(f_max or max(f), max(f))
        # create tuple (patch_level_low, patch_level_high)
        if fill_value is not None:
            patch_level = fill_value
        if patch_level is None:
            patch_level = (10*max(psd[f < f_min]), 10*max(psd[f >= f_max]))
        else:
            try:
                patch_level[1]
            except Exception:
                patch_level = (patch_level, patch_level)
        # patch low frequencies
        psd[f < f_min] = patch_level[0]
        # patch high frequencies
        psd[f > f_max] = patch_level[1]
        if not in_place:
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

    def draw_noise_fd(self, freq: np.ndarray | None = None,
                      delta_f: float | None = None,
                      f_min: float | None = None,
                      f_max: float | None = None,
                      prng: int | np.random.Generator | None = None,
                      **kws):
        """Draw Fourier-domain noise from the PSD, with variance consistent
        with the LIGO definition of the PSD (cf. GW likelihood), namely

        .. math::
            \\tilde{h}(f) \\sim \\mathcal{N}(0, \\sqrt{\\frac{S(f)}{4\\Delta f}

        where the covariance matrix is diagonal, and :math:`\\Delta f = 1/T` is
        the PSD.

        Can specify a frequency array or frequency range and spacing to which
        interpolate or extrapolate the PSD.

        Other keyword arguments are passed to the interpolation rutine
        :meth:`PowerSpectrum.interpolate_to_index`.

        Arguments
        ---------
        freq : array
            frequencies over which to draw noise (default to PSD index).
        delta_f : float
            frequency spacing; can be used to create ``freq`` if not provided).
        f_min : float
            minimum frequency to draw noise.
        f_max : float
            maximum frequency to draw noise.
        prng : int, np.random.Generator
            random number generator.
        kws : dict
            additional keyword arguments passed to
            :meth:`PowerSpectrum.interpolate_to_index`.

        Returns
        -------
        noise : FrequencySeries
            noise realization.
        """
        if isinstance(prng, int):
            prng = np.random.default_rng(prng)
        elif isinstance(prng, np.random.Generator):
            pass
        elif prng is None:
            prng = np.random.default_rng()
        else:
            raise ValueError(f"invalid random number generator {prng}")

        psd = self.interpolate_to_index(freq=freq, delta_f=delta_f,
                                        f_min=f_min, f_max=f_max, **kws)

        # draw noise with the correct variance corresponding to the LIGO
        # definition of the PSD (cf. GW likelihood)
        n = len(psd)
        std = np.sqrt(psd / (4*psd.delta_f))
        noise_real = prng.normal(size=n, loc=0, scale=std)
        noise_imag = prng.normal(size=n, loc=0, scale=std)

        return FrequencySeries(noise_real + 1j*noise_imag, index=psd.freq,
                               name=self.name)

    def draw_noise_td(self, duration: float | None = None,
                      f_samp: float | None = None,
                      delta_t: float | None = None,
                      f_min: float | None = None,
                      f_max: float | None = None,
                      delta_f: float | None = None,
                      epoch=0., **kws):
        """Draw time-domain noise from the PSD.

        Duration and sampling rate are determined from the PSD if not provided.
        If a frequency range is provided, the PSD is interpolated/extrapolated
        to that range before drawing the noise.

        Additional keyword arguments are passed to
        :meth:`PowerSpectrum.draw_noise_fd`.

        Arguments
        ---------
        duration : float
            duration of the time series (provide this argument or `delta_f`).
        f_samp : float
            sampling frequency (provide this argument or `f_high` which will be
            half of `f_samp`; alternatively, `f_samp` will be determined from
            `delta_t`).
        delta_t : float
            time step (provide this argument or `f_samp` or `f_high`).
        f_min : float
            minimum frequency to draw noise.
        f_max : float
            maximum frequency to draw noise.
        delta_f : float
            frequency step (provide this argument or `duration`)
        epoch : float
            initial time of the time series.
        kws : dict
            additional keyword arguments passed to
            :meth:`PowerSpectrum.draw_noise_fd`.

        Returns
        -------
        noise : Data
            noise realization.
        """
        if duration is None and delta_f is None:
            duration = self.duration
        elif duration is None and delta_f is not None:
            duration = 1 / delta_f
        elif delta_f is not None and duration is not None:
            if duration != 1 / delta_f:
                raise ValueError("cannot specify both duration and delta_f")

        if f_samp is None and f_max is not None:
            f_samp = 2 * f_max
        elif f_samp is not None and f_max is not None:
            if f_samp != 2 * f_max:
                raise ValueError("cannot specify both f_samp and f_max")
        if delta_t is None and f_samp is None:
            delta_t = self.delta_t
        elif delta_t is None and f_samp is not None:
            delta_t = 1 / f_samp
        elif delta_t is not None and f_samp is not None:
            if delta_t != 1 / f_samp:
                raise ValueError("cannot specify both delta_t and f_samp/max")

        # we must have a well formed frequency array to draw TD noise
        t_len = int(duration / delta_t)
        freq = np.fft.rfftfreq(t_len, delta_t)

        psd = self.copy()
        if f_min is not None or f_max is not None:
            psd.patch(in_place=True, f_min=f_min, f_max=f_max,
                      fill_value=kws.get('fill_value', 0.))
        noise_fd = psd.draw_noise_fd(freq=freq, **kws)
        return noise_fd.to_time_series(epoch=epoch)

    def inner_product(self,
                      x: FrequencySeries,
                      y: FrequencySeries | None = None,
                      f_min: float | None = None,
                      f_max: float | None = None) -> complex:
        """Compute the noise weighterd inner product between `x` and `y`
        defined by :math:`\\left\\langle x \\mid y \\right\\rangle \\equiv
        4 \\Delta f \\Re \\sum x_i y_i / S_i`.

        Arguments
        ---------
        x : array
            target frequency series.
        y : array
            reference frequency series. Defaults to `x`.
        f_min : float
            lower frequency bound (default derived from `x`)
        f_max : float
            upper frequency bound (default derived from `x`)

        Returns
        -------
        snr : float
            signal-to-noise ratio
        """
        if f_min is None:
            f_min = x.freq[0]
        if f_max is None:
            f_max = x.freq[-1]
        x = x.loc[f_min:f_max]
        f = x.freq

        if y is None:
            y = x
        else:
            y = y.interpolate_to_index(f, fill_value=0.)
        s = self.interpolate_to_index(f, fill_value=np.inf)

        return 4*x.delta_f*np.sum(np.conj(x)*y/s)

    def compute_snr(self,
                    x: FrequencySeries,
                    y: FrequencySeries | None = None,
                    f_min: float | None = None,
                    f_max: float | None = None) -> float:
        """Efficiently compute the signal-to_noise ratio
        :math:`\\mathrm{SNR} = \\left\\langle x \\mid y \\right\\rangle /
        \\sqrt{\\left\\langle x \\mid x \\right\\rangle}`, where the inner
        product is defined by
        :math:`\\left\\langle x \\mid y \\right\\rangle \\equiv
        4 \\Delta f \\Re \\sum x_i y_i / S_i`.

        If `x` is a signal and `y` is noisy data, then this is the matched
        filter SNR; if both of them are a template, then this is the optimal
        SNR (default).

        Arguments
        ---------
        x : array
            target frequency series.
        y : array
            reference frequency series. Defaults to `x`.

        Returns
        -------
        snr : float
            signal-to-noise ratio
        """
        x_dot_y = self.inner_product(x, y, f_min, f_max).real
        if y is None:
            return np.sqrt(x_dot_y)
        x_dot_x = self.inner_product(x, x, f_min, f_max).real
        return x_dot_y / np.sqrt(x_dot_x)


class AutoCovariance(TimeSeries):
    """Contains and manipulates autocovariance functions, a special kind of
    :class:`TimeSeries`.
    """

    _meta = ['ifo', 'attrs']

    def __init__(self, *args, delta_t=None, ifo=None, attrs=None,
                 **kwargs):
        if ifo is not None:
            ifo = ifo.upper()
        kwargs['name'] = kwargs.get('name', ifo)
        super().__init__(*args, **kwargs)
        if delta_t is not None:
            self.index = np.arange(len(self))*delta_t
        if len(args) == 0:
            args = [None]
        self.ifo = ifo or getattr(args[0], 'ifo', None)
        self.attrs = attrs or getattr(args[0], 'attrs', {}) or {}

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
        **kws :
            additional keyword arguments passed to
            :meth:`PowerSpectrum.fom_data`.

        Returns
        -------
        acf : AutoCovariance
            estimate of the autocovariance function.
        """
        dt = getattr(d, 'delta_t', delta_t)
        n = n or len(d)
        if method.lower() == 'td':
            rho = sig.correlate(d, d, **kws)
            rho = np.fft.ifftshift(rho)
            rho = rho[:n] / len(d)
        elif method.lower() == 'fd':
            kws['f_samp'] = kws.get('f_samp', 1/dt)
            rho = PowerSpectrum.from_data(d, **kws).to_acf()
        else:
            raise ValueError("method must be 'td' or 'fd' not %r" % method)
        _meta = {a: getattr(d, a, None) for a in getattr(d, '_meta', [])}
        return cls(rho, delta_t=dt, **_meta)

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
    def matrix(self) -> np.ndarray:
        """Covariance matrix built from ACF, :math:`C_{ij} = \\rho(|i-j|)`.
        """
        return sl.toeplitz(self)

    @property
    def cholesky(self) -> np.ndarray:
        """Cholesky factor :math:`L` of covariance matrix :math:`C = L^TL`.
        """
        if getattr(self, '_cholesky', None) is None:
            self._cholesky = np.linalg.cholesky(self.matrix)
        return self._cholesky

    def compute_snr(self, x, y=None) -> float:
        """Efficiently compute the signal-to_noise ratio
        :math:`\\mathrm{SNR} = \\left\\langle x \\mid y \\right\\rangle /
        \\sqrt{\\left\\langle x \\mid x \\right\\rangle}`, where the inner
        product is defined by
        :math:`\\left\\langle x \\mid y \\right\\rangle \\equiv x_i
        C^{-1}_{ij} y_j`. This is internally computed using Cholesky factors to
        speed up the computation.

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
        if y is None:
            y = x
        ow_x = sl.solve_toeplitz(self.iloc[:len(x)], x)
        return np.dot(ow_x, y)/np.sqrt(np.dot(x, ow_x))

    def inner_product(self, x, y=None) -> complex:
        """Compute the noise weighterd inner product between `x` and `y`
        defined by :math:`\\left\\langle x \\mid y \\right\\rangle \\equiv
        x_i C^{-1}_{ij} y_j`.

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
        if y is None:
            y = x
        ow_x = sl.solve_toeplitz(self.iloc[:len(x)], x)
        return np.dot(x, ow_x)

    def whiten(self, data) -> Data | TimeSeries | np.ndarray:
        """Whiten stretch of data using ACF.

        Arguments
        ---------
        data : array, TimeSeries
            unwhitened data.

        Returns
        -------
        w_data : Data | TimeSeries | array
            whitened data.
        """
        if isinstance(data, TimeSeries):
            assert (data.delta_t == self.delta_t)
        # whiten stretch of data using Cholesky factor
        L = self.iloc[:len(data)].cholesky
        w_data = sl.solve_triangular(L, data, lower=True)
        # return same type as input
        if isinstance(data, Data):
            _meta = {a: getattr(data, a, None)
                     for a in getattr(data, '_meta', [])}
            w_data = Data(w_data, index=data.index, **_meta)
        elif isinstance(data, TimeSeries):
            w_data = TimeSeries(w_data, index=data.index)
        return w_data


class StrainStack(np.ndarray):
    """A wrapper around numpy arrays with useful methods for whitening strain
    data and computing SNRs.
    """

    def __new__(cls, input_array, attrs=None, *args, **kwargs):
        obj = np.asarray(input_array).view(cls)
        # Assign custom attribute
        obj.attrs = attrs or {}
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        # Preserve custom attributes
        self.attrs = getattr(obj, 'attrs', {})

    @staticmethod
    def _get_time_axis(h, cholesky):
        ntime = np.shape(cholesky)[1]
        matching_axes = [i for i, n in enumerate(h.shape) if n == ntime]
        if len(matching_axes) == 0:
            raise ValueError("No matching time axis found:"
                             f"strain shape {h.shape}, cholesky shape "
                             f"{np.shape(cholesky)}")
        elif len(matching_axes) > 1:
            raise ValueError("Multiple matching time axes found:"
                             f"strain shape {h.shape}, cholesky shape "
                             f"{np.shape(cholesky)}")
        return matching_axes[0]

    @staticmethod
    def _get_ifo_axis(h, cholesky):
        nifo = len(cholesky)
        matching_axes = [i for i, n in enumerate(h.shape) if n == nifo]
        if len(matching_axes) == 0:
            raise ValueError("No matching detector axis found:"
                             f"strain shape {h.shape}, cholesky shape "
                             f"{np.shape(cholesky)}")
        elif len(matching_axes) > 1:
            raise ValueError("Multiple matching detector axes found:"
                             f"strain shape {h.shape}, cholesky shape "
                             f"{np.shape(cholesky)}")
        return matching_axes[0]

    def whiten(self, cholesky: list | np.ndarray | dict,
               ifo_axis: int | None = None,
               time_axis: int | None = None) -> np.ndarray:
        """Whiten strain data using provided Cholesky factors for the
        covariance matrix of the noise at each detector.

        Arguments
        ---------
        cholesky : array | dict
            Cholesky factor or list of Cholesky factors; if a dict
            the values are serialized assuming entries correspond to
            detectors.
        ifo_axis : int
            axis indexing detectors in the strain array, default is
            guessed from cholesky array shape.
        time_axis : int
            axis indexing time in the strain array, default is guessed
            from cholesky array shape.

        Returns
        -------
        wh : array
            whitened strain data, with same shape as original array.
        """

        # check shape of strain array (nifo, ntime, ...)
        h = np.atleast_2d(self)

        # check shape of cholesky factors (nifo, ntime, ntime)
        if isinstance(cholesky, dict):
            cholesky = list(cholesky.values())
        cholesky = np.atleast_3d(cholesky)

        # identify detector axis
        if ifo_axis is None:
            ifo_axis = self._get_ifo_axis(h, cholesky)

        # identify time axis
        if time_axis is None:
            time_axis = self._get_time_axis(h, cholesky)

        # bring ifo and time axes to first and second locations
        h = np.moveaxis(h, [ifo_axis, time_axis], [0, 1])

        wh = np.array([sl.solve_triangular(L, h_i, lower=True)
                       for L, h_i in zip(cholesky, h)])

        # undo axis swapping
        wh = np.moveaxis(wh, [0, 1], [ifo_axis, time_axis])

        # remove extra dimension if we added it above
        if self.ndim == 1:
            wh = wh[0]

        return wh

    def compute_snr(self, cholesky: list | np.ndarray | dict,
                    data: list | np.ndarray | dict | None = None,
                    network: bool = False, cumulative: bool = False,
                    time_axis: int | None = None,
                    ifo_axis: int | None = None) -> np.ndarray:
        """Compute SNR of strain data using provided Cholesky factors for the
        covariance matrix of the noise at each detector.

        If ``data`` are provided, computes the matched-filter SNR; otherwise,
        computes the optimal SNR.

        Arguments
        ---------
        cholesky : list, array, or dict
            Cholesky factor or list of Cholesky factors; if a dict
            the values are serialized assuming entries correspond to
            detectors.
        data : list, array, or dict, optional
            data array or list of data arrays; if a dict the values are
            serialized assuming entries correspond to detectors; if None,
            computes the optimal SNR, rather than the matched filter SNR.
        network : bool, optional
            compute network SNR by taking norm over detectors.
        cumulative : bool, optional
            compute cumulative SNR over time.
        time_axis : int, optional
            axis indexing time in the strain array, default is guessed
            from ``cholesky`` array shape.
        ifo_axis : int, optional
            axis indexing detectors in the strain array, default is
            guessed from ``cholesky`` array shape.

        Returns
        -------
        snrs : array
            SNR array with the time dimension removed if ``cumulative`` is
            False and the detector dimension removed if ``network`` is True.
        """
        # check shape of cholesky factors (nifo, ntime, ntime)
        if isinstance(cholesky, dict):
            cholesky = list(cholesky.values())
        cholesky = np.atleast_3d(cholesky)

        # check shape of strain array (nifo, ntime, ...)
        h = np.atleast_2d(self)

        # identify detector and time axes
        if ifo_axis is None:
            ifo_axis = self._get_ifo_axis(h, cholesky)
        if time_axis is None:
            time_axis = self._get_time_axis(h, cholesky)

        # get whitened templates
        whs = h.whiten(cholesky, ifo_axis=ifo_axis, time_axis=time_axis)

        if cumulative:
            opt_ifo_snrs = np.sqrt(np.cumsum(whs**2, axis=time_axis))
        else:
            opt_ifo_snrs = np.linalg.norm(whs, axis=time_axis)

        if data is None:
            snrs = opt_ifo_snrs
        else:
            # check shape of data array (nifo, ntime, ...)
            if isinstance(data, dict):
                data = list(data.values())
            data = np.atleast_2d(data)
            if np.ndim(data) > 2:
                raise ValueError("Data array must have shape (nifo, ntime)")

            # whiten data
            wds = np.array([sl.solve_triangular(L, d, lower=True)
                            for L, d in zip(cholesky, data)])

            # make sure ifo axis is first and time axis is second
            if ifo_axis is None:
                ifo_axis = self._get_ifo_axis(h, cholesky)
            whs = np.moveaxis(whs, [ifo_axis, time_axis], [0, 1])

            # extra dimensions
            n = np.ndim(whs) - np.ndim(wds)

            # take inner product
            wdwh = wds[(...,)+(None,)*n] * whs
            wdwh = np.moveaxis(wdwh, [0, 1], [ifo_axis, time_axis])
            if cumulative:
                snrs = np.cumsum(wdwh, axis=time_axis) / opt_ifo_snrs
            else:
                snrs = np.sum(wdwh, axis=time_axis) / opt_ifo_snrs

        if network:
            if ifo_axis is None:
                ifo_axis = self._get_ifo_axis(snrs, cholesky)
            return np.linalg.norm(snrs, axis=ifo_axis)
        else:
            if self.ndim == 1:
                # an extra dimension was prepended to the strain array,
                # remove it from output
                snrs = snrs[0]
            return snrs

    def slice(self, start_indices: list | dict,
              n: int, ifo_axis: int = 0) -> 'StrainStack':
        """Slice data array into segments of length `n` starting at
        `start_indices`.

        Arguments
        ---------
        start_indices : list | dict
            starting indices of each segment; if a dict, the values are
            serialized assuming entries correspond to detectors.
        n : int
            length of each segment.
        ifo_axis : int, optional
            index detector axis in the strain array (default is 0).
        """
        if isinstance(start_indices, dict):
            start_indices = list(start_indices.values())
        h = np.moveaxis(self, ifo_axis, 0)
        new_h = []
        for i0, ifo_wfs in zip(start_indices, h):
            new_h.append(ifo_wfs[i0:i0+n, ...])
        return StrainStack(np.moveaxis(new_h, 0, ifo_axis))
