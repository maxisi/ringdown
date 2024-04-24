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

class Series(pd.Series):
    """ A wrapper of :class:`pandas.Series` with some additional functionality.
    """

    @property
    def _constructor(self):
        return Series

    @classmethod
    def read(cls, path : str | None = None, kind : str | None = None, **kws):
        """Load data from disk.
        
        If ``kind`` is `gwosc` assumes input is an strain HDF5 file downloaded
        from `GWOSC <https://www.gw-openscience.org>`_.  If ``kind`` is `frame`
        and the keyword `channel` is given, then attempt to load the given
        path(s) using gwpy's frame file reading.  Otherwise, it is a wrapper
        around :func:`pandas.read_hdf` or :func:`pandas.read_csv` functions, for
        ``kind = 'hdf'`` or ``kind = 'csv'``.

        If ``kind`` is ``None``, guesses filetype from extension.

        Arguments
        ---------
        path : str
            path to file, or None if fetching remote data
        kind : str
            kind of file to load: `gwsoc`, `hdf` or `csv`, or `discover` to
            automatically discover remote or local data with GWpy.

        Returns
        -------
        series : Series
            series loaded from disk
        """
        if path is None and kind is None:
            raise ValueError("must provide either path or kind")
        
        kind = (kind or '').lower()
        channel = kws.pop('channel', None)
        frametype = kws.pop('frametype', None)
        start = kws.pop('start', None)
        end = kws.pop('end', None)
        t0 = kws.pop('t0', None)
        seglen = kws.pop('seglen', None)
        if start is None and end is None and t0 is not None and path is None:
            if seglen is None:
                raise ValueError("must provide seglen if t0 is provided")
            start = t0 - seglen/2
            end = t0 + seglen/2
            logging.info(f"fetching {seglen} s long segment centered on {t0}"
                         f" [{start}, {end}]")
        if start is not None and end is not None and path is not None:
            logging.warning("seglen options ignored when loading from disk")
        
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
                    return cls(h, index=time, **kws)
            else:
                logging.info("GWOSC file not found, attempting to fetch from server")
                # first check that we have GWpy, which is an optional dependency
                from gwpy.timeseries import TimeSeries
                if start is None or end is None:
                    raise ValueError("need start and end times to fetch from GWOSC")
                ifo = kws.get('ifo', None)
                if ifo is None:
                    raise ValueError("ifo must be provided to fetch from GWOSC")
                # get arguments accepted by TimeSeries.fetch_open_data()
                ts_args = list(inspect.signature(TimeSeries.fetch_open_data).parameters.keys())
                kwargs = list(kws.keys())
                ts_kws = {k: kws.pop(k) for k in ts_args if k in kwargs 
                          and k not in['ifo', 'start', 'end']}
                ts = TimeSeries.fetch_open_data(ifo, start, end, **ts_kws)
                # GWpy puts units on its times, we remove them by redefining the index
                return cls(ts.value, index=np.array(ts.times), **kws)
            
        elif kind == 'discover':
            logging.info("attempting to discover remote or local data using GWpy")
            # first check that we have GWpy, which is an optional dependency
            from gwpy.timeseries import TimeSeries
            logging.info("attempting to discover remote or local data using NDS2/FrameCPP")
            # validate all arguments
            if channel is None:
                raise KeyError('channel must be specified to discover')
            if start is None or end is None:
                raise ValueError("start and end times must be provided to discover data")
            # get arguments accepted by TimeSeries.get(), fetch() and find()
            ts_args = list(inspect.signature(TimeSeries.get).parameters.keys())
            ts_args += list(inspect.signature(TimeSeries.fetch).parameters.keys())
            ts_args += list(inspect.signature(TimeSeries.find).parameters.keys())
            kwargs = list(kws.keys())
            ts_kws = {k: kws.pop(k) for k in ts_args if k in kwargs
                      and k not in ['channel', 'start', 'end']}
            ts_kws['frametype'] = frametype
            # fetch data and form Data object
            ts = TimeSeries.get(channel, start, end, **ts_kws)
            return cls(ts.value, index=np.array(ts.times), **kws)
        
        elif kind == 'frame':
            if channel is None:
                raise KeyError('channel must be specified for frame files')
            from gwpy.timeseries import TimeSeries
            ts = TimeSeries.read(path, channel)
            # GWpy puts units on its times, we remove them by redefining the index
            return cls(ts.value, index=np.array(ts.times), **kws)
        
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
                                "strange errors due to precision loss")
            if kind == 'csv':
                read_kws['header'] = kws.get('header', None)
            # squeeze if needed (since sequeeze argument no longer accepted)
            d = read_func(path, **read_kws)
            if squeeze and not isinstance(d, pd.Series):
                d = d.squeeze("columns")
            return cls(d, **cls_kws)
        else:
            raise ValueError("unrecognized file kind: {}".format(kind))
    
    @classmethod
    def fetch(cls, *args, **kws):
        """Alias to :meth:`Series.read`.
        """
        return cls.read(*args, **kws)

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
        info = {a: getattr(self, a, None) for a in getattr(self, '_metadata', [])}
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
    def f_samp(self) -> float:
        """Sampling frequency (`1/delta_t`)."""
        return 1/self.delta_t

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

    def interpolate_to_index(self, freq : np.ndarray | None = None,
                             delta_f : float | None = None,
                             f_min : float | None = None,
                             f_max : float | None = None,
                             log : bool =False, **kws):
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
            interpolate in log-log space (default False)

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
            # construct loglog representation of self
            y = np.log(self)
            y.index = np.log(self.index)
            # interpolate in loglog space
            yinterp = np.exp(y.interpolate_to_index(np.log(freq), **kws))
            yinterp.index = freq
            return yinterp
        else:
            return super(FrequencySeries, 
                         self).interpolate_to_index(freq, **kws)


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
        if ifo is None and len(args) > 0:
            self.ifo = getattr(args[0], 'ifo', None)
        if info is None and len(args) > 0:
            self.info = getattr(args[0], 'info', None)

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

    def condition(self, t0 : float | None = None,
                  ds : int | None = None,
                  flow : float | None = None,
                  fhigh : float | None = None,
                  trim : float = 0.25,
                  digital_filter : bool = True,
                  remove_mean : bool = True,
                  decimate_kws : dict | None = None,
                  slice_left : float | None = None,
                  slice_right : float | None = None):
        """Condition data.

        Arguments
        ---------
        t0 : float
            target time to be preserved after downsampling.
        ds : int
            decimation factor for downsampling.
        flow : float
            lower frequency for high passing.
        fhigh : float
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
                raise ValueError("t0 must be within the time series")
            ds = int(ds or 1)
            i = np.argmin(abs(raw_time - t0))
            raw_time = np.roll(raw_time, -(i % ds))
            raw_data = np.roll(raw_data, -(i % ds))

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
    def from_psd(cls, psd : 'PowerSpectrum', **kws):
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
        return Data(noise_td, **_meta)
        

class PowerSpectrum(FrequencySeries):
    """Contains and manipulates power spectral densities, a special kind of
    :class:`FrequencySeries`.
    """
    
    _metadata = ['ifo', 'info']
            
    def __init__(self, *args, delta_f=None, ifo=None, info=None,
                 **kwargs):
        if ifo is not None:
            ifo = ifo.upper()
        kwargs['name'] = kwargs.get('name', ifo)
        super(PowerSpectrum, self).__init__(*args, **kwargs)
        if delta_f is not None:
            self.index = np.arange(len(self))*delta_f
        self.ifo = ifo
        self.info = info or {}
        if ifo is None and len(args) > 0:
            self.ifo = getattr(args[0], 'ifo', None)
        if info is None and len(args) > 0:
            self.info = getattr(args[0], 'info', None)

    @property
    def _constructor(self):
        return PowerSpectrum

    @classmethod
    def from_data(cls, data : Data | np.ndarray,
                  f_min : float | None = None, f_max : float | None = None,
                  fill_value : float | tuple | None = None, **kws):
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
        \*\*kws :
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
        _meta = {a: getattr(data, a, None) for a in getattr(cls, '_metadata', [])}
        p = cls(psd, index=freq, **_meta)
        if f_min is not None or f_max is not None:
            p.patch(f_min=f_min, f_max=f_max, fill_value=fill_value,
                    in_place=True)
        return p

    @classmethod
    def from_lalsimulation(cls, func, freq, flow=0, fill_value=None, **kws):
        """Obtain :class:`PowerSpectrum` from LALSimulation function.

        Arguments
        ---------
        func : str, builtin_function_or_method
            LALSimulation PSD function, or name thereof (e.g.,
            ``SimNoisePSDaLIGOZeroDetHighPower``).
        freq : array
            frequencies over which to evaluate PSD.
        flow : float
            lower frequency threshold for padding: PSD will be patched below
            this value.
        \*\*kw : 
            additional arguments passed to padding function
            :meth:`PowerSpectrum.patch`.

        Returns
        -------
        psd : PowerSpectrum
            power spectrum frequency series.
        """
        if isinstance(func, str):
            import lalsimulation as lalsim
            func = getattr(lalsim, func)
        f_ref = freq[np.argmin(abs(freq - flow))]
        p_ref = func(f_ref) if fill_value is None else fill_value
        def get_psd_bin(f):
            if f > flow:
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
        _meta = {a: getattr(self, a, None) 
                 for a in getattr(self, '_metadata', [])}
        return AutoCovariance(rho, delta_t=self.delta_t, **_meta)
    
    def draw_noise_fd(self, freq : np.ndarray | None = None,
                      delta_f : float | None = None,
                      f_min : float | None = None,
                      f_max : float | None = None,  
                      rng : int | np.random.Generator | None = None,
                      **kws):
        """Draw Fourier-domain noise from the PSD, with variance consistent
        with the LIGO definition of the PSD (cf. GW likelihood), namely
        
        .. math::
            \\tilde{h}(f) \\sim \\mathcal{N}(0, \\sqrt{\\frac{S(f)}{4\\Delta f}
            
        where the covariance matrix is diagonal, and :math:`\\Delta f = 1/T` is
        the PSD.
        
        Can specify a frequency array or frequency range and spacing to which
        interpolate or extrapolate the PSD.
        
        All keyword arguments except for `rng` are passed to the interpolation
        routing :meth:`PowerSpectrum.interpolate_to_index`.

        Returns
        -------
        noise : FrequencySeries
            noise realization.
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng()

        psd = self.interpolate_to_index(freq=freq, delta_f=delta_f,
                                        f_min=f_min, f_max=f_max, **kws)
        
        # draw noise with the correct variance corresponding to the LIGO
        # definition of the PSD (cf. GW likelihood)
        n = len(psd)
        std = np.sqrt(psd / (4*psd.delta_f))
        noise_real = np.random.normal(size=n, loc=0, scale=std)
        noise_imag = np.random.normal(size=n, loc=0, scale=std)
        
        return FrequencySeries(noise_real + 1j*noise_imag, index=psd.freq,
                               name=self.name)
        
    def draw_noise_td(self, duration : float | None = None,
                      f_samp : float | None = None,
                      delta_t : float | None = None,
                      f_min : float | None = None,
                      f_max : float | None = None,
                      delta_f : float | None = None,
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
            raise ValueError("cannot specify both duration and delta_f")
        
        if f_samp is None and f_max is not None:
            f_samp = 2 * f_max
        elif f_samp is not None and f_max is not None:
            raise ValueError("cannot specify both f_samp and f_max")
        if delta_t is None and f_samp is None:
            delta_t = self.delta_t
        elif delta_t is None and f_samp is not None:
            delta_t = 1 / f_samp
        elif delta_t is not None and f_samp is not None:
            raise ValueError("cannot specify both delta_t and f_samp/max")
        
        # we must have a well formed frequency array to draw TD noise
        t_len = int(duration / delta_t)
        freq = np.fft.rfftfreq(t_len, delta_t)
        
        psd = self.copy()
        if f_min is not None or f_max is not None:
            psd.patch(inplace=True, f_min=f_min, f_max=f_max,
                      fill_value=kws.get('fill_value', 0.))
        noise_fd = psd.draw_noise_fd(freq=freq, **kws)
        return noise_fd.to_time_series(epoch=epoch)        


class AutoCovariance(TimeSeries):
    """Contains and manipulates autocovariance functions, a special kind of
    :class:`TimeSeries`.
    """
    
    _metadata = ['ifo', 'info']
            
    def __init__(self, *args, delta_t=None, ifo=None, info=None,
                 **kwargs):
        if ifo is not None:
            ifo = ifo.upper()
        kwargs['name'] = kwargs.get('name', ifo)
        super(AutoCovariance, self).__init__(*args, **kwargs)
        if delta_t is not None:
            self.index = np.arange(len(self))*delta_t
        self.ifo = ifo
        self.info = info or {}
        if ifo is None and len(args) > 0:
            self.ifo = getattr(args[0], 'ifo', None)
        if info is None and len(args) > 0:
            self.info = getattr(args[0], 'info', None)
            

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
            kws['fs'] = kws.get('fs', 1/dt)
            rho = PowerSpectrum.from_data(d, **kws).to_acf()
        else:
            raise ValueError("method must be 'td' or 'fd' not %r" % method)
        _meta = {a: getattr(cls, a, None) 
                     for a in getattr(cls, '_metadata', [])}
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
        return PowerSpectrum(psd, index=freq, **_meta)

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
            self._cholesky = np.linalg.cholesky(self.matrix)
        return self._cholesky

    def compute_snr(self, x, y=None):
        """Efficiently compute the signal-to_noise ratio
        :math:`\\mathrm{SNR} = \left\langle x \mid y \\right\\rangle /
        \\sqrt{\\left\langle x \mid x \\right\\rangle}`, where the inner product
        is defined by :math:`\left\langle x \mid y \\right\\rangle \\equiv x_i
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

        if y is None: y = x
        ow_x = sl.solve_toeplitz(self.iloc[:len(x)], x)
        return np.dot(ow_x, y)/np.sqrt(np.dot(x, ow_x))

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
            _meta = {a: getattr(data, a, None) 
                     for a in getattr(data, '_metadata', [])}
            w_data = Data(w_data, index=data.index, **_meta)
        elif isinstance(data, TimeSeries):
            w_data = TimeSeries(w_data, index=data.index)
        return w_data
