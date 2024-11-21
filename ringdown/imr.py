__all__ = ['IMRResult']

import h5py
import os
import numpy as np
import pandas as pd
from . import indexing
from . import qnms
from . import waveforms
from . import target
from . import data
from . indexing import ModeIndexList
from . import utils
from .utils import get_tqdm, get_hdf5_value, get_bilby_dict, \
    get_dict_from_pattern
import lal
import multiprocessing as mp
from lalsimulation import nrfits
import logging

MASS_ALIASES = ['final_mass', 'mf', 'mfinal', 'm_final', 'final_mass_source',
                'remnant_mass']
SPIN_ALIASES = ['final_spin', 'remnant_spin', 'chif', 'chi_f', 'chi_final',
                'af', 'a_final']
TIME_KEY = '_time'


def get_remnant(mass_1, mass_2, spin_1x, spin_1y, spin_1z,
                spin_2x, spin_2y, spin_2z, f_ref, model):
    r = nrfits.eval_nrfit(mass_1, mass_2,
                          [spin_1x, spin_1y, spin_1z],
                          [spin_2x, spin_2y, spin_2z],
                          model, f_ref=float(f_ref),
                          fit_types_list=["FinalMass", "FinalSpin"])
    return r['FinalMass'][0], np.linalg.norm(r['FinalSpin'])


class IMRResult(pd.DataFrame):

    _f_key = 'f_{mode}'
    _g_key = 'g_{mode}'

    _meta = ['attrs', '_psds']

    def __init__(self, *args, attrs=None, psds=None, **kwargs):
        super().__init__(*args, **kwargs)
        if len(args) == 0:
            args = [None]
        self.attrs = attrs or getattr(args[0], 'attrs', {}) or {}
        self.__dict__['_psds'] = psds if psds is not None else {}
        self.__dict__['_waveforms'] = None

    @property
    def config(self):
        """Configuration settings used in the analysis."""
        return self.attrs.get('config', {})

    @property
    def reference_frequency(self) -> float | None:
        """Reference frequency used in analysis in Hz."""
        config_fref = self.attrs.get('config', {}).get('reference-frequency')
        fref = self.attrs.get('reference_frequency',
                              self.attrs.get('f_ref', config_fref))
        if fref is not None:
            return float(fref)
        return None

    def set_reference_frequency(self, f_ref: float) -> None:
        """Set the reference frequency used in analysis in Hz."""
        self.attrs['reference_frequency'] = f_ref
        if 'f_ref' in self.attrs:
            del self.attrs['f_ref']

    @property
    def psds(self) -> dict[data.PowerSpectrum]:
        """Power Spectral Densities used in the analysis."""
        return self.__dict__['_psds'] or {}

    def set_psds(self, psds: dict | str, ifos: list | None = None) -> None:
        """Set the PSDs used in the analysis.

        Arguments
        --------
        psds : dict | str
            Dictionary of PowerSpectralDensity objects or paths to PSD files.
        ifos : list | None
            List of detector names to associate with the PSDs; if None, uses
            the keys of the PSD dictionary or defaults to the detectors in the
            IMR result.
        """
        if ifos is None:
            ifos = self.ifos
        psds = get_dict_from_pattern(psds, ifos)
        if 'psds' not in self.attrs:
            self.attrs['psds'] = {}
        for i, p in psds.items():
            if isinstance(p, str):
                if os.path.isfile(p):
                    p = np.loadtxt(p)
                    self.attrs['psds'][i] = p
                else:
                    raise FileNotFoundError(f"PSD file not found: {p}")
            p = data.PowerSpectrum(p).fill_low_frequencies().gate()
            self.__dict__['_psds'][i] = p

    @property
    def approximant(self) -> str | None:
        """Waveform approximant used in analysis."""
        config_approx = self.attrs.get(
            'config', {}).get('waveform-approximant')
        return self.attrs.get('approximant', config_approx)

    def set_approximant(self, approximant: str) -> None:
        """Set the waveform approximant used in analysis."""
        self.attrs['approximant'] = approximant

    @property
    def _constructor(self):
        return IMRResult

    @property
    def final_mass(self) -> np.ndarray:
        """Remnant mass samples."""
        for k in MASS_ALIASES:
            if k in self.columns:
                return self[k]

    @property
    def final_spin(self) -> np.ndarray:
        """Remnant spin samples."""
        for k in SPIN_ALIASES:
            if k in self.columns:
                return self[k]

    @property
    def minimum_frequency(self):
        """Minimum frequency used in the analysis."""
        x = self.attrs.get('config', {}).get('minimum-frequency', {})
        if isinstance(x, float):
            return {k: x for k in self.ifos}
        return {k: float(v) for k, v in get_bilby_dict(x).items()}

    @property
    def maximum_frequency(self):
        """Maximum frequency used in the analysis."""
        x = self.attrs.get('config', {}).get('maximum-frequency', {})
        if isinstance(x, float):
            return {k: x for k in self.ifos}
        return {k: float(v) for k, v in get_bilby_dict(x).items()}

    @property
    def trigger_time(self):
        """Trigger time used in the analysis."""
        return self.attrs.get('config', {}).get('trigger-time')

    @property
    def sampling_frequency(self):
        """Sampling frequency used in the analysis."""
        return self.attrs.get('config', {}).get('sampling-frequency')

    @property
    def duration(self):
        """Duration of the analysis."""
        return self.attrs.get('config', {}).get('duration')

    @property
    def remnant_mass_scale(self) -> pd.Series:
        """Get best available remmnant mass scale from samples."""
        if self.final_mass is not None:
            return self.final_mass
        logging.info("no remnant mass found; using total mass")
        if 'total_mass' in self:
            return self['total_mass']
        elif 'mass_1' in self and 'mass_2' in self:
            return self['mass_1'] + self['mass_2']
        else:
            raise KeyError("no mass scale found")

    @property
    def remnant_mass_scale_reference(self) -> float:
        """Get the reference remnant mass scale."""
        return np.median(self.remnant_mass_scale)

    def get_kerr_frequencies(self, modes, **kws):
        """Get the Kerr QNM frequencies corresponding to the remnant mass and
        spin for a list of modes.

        Arguments
        ---------
        modes : list of str | list of indexing.ModeIndex
            any argument accepted by :class:`indexing.ModeIndexList`.
        """
        modes = indexing.ModeIndexList(modes)
        m = self.final_mass
        c = self.final_spin
        f_keys = []
        g_keys = []
        for index in modes:
            # check if we have already computed the QNM frequency
            label = index.get_label()
            f_key = self._f_key.format(mode=label)
            g_key = self._g_key.format(mode=label)
            if not (f_key in self.columns and g_key in self.columns):
                qnm = qnms.KerrMode(index)
                f, g = qnm.fgamma(chi=c, m_msun=m, **kws)
                self[f_key] = f
                self[g_key] = g
            f_keys.append(f_key)
            g_keys.append(g_key)
        return self[f_keys + g_keys]

    def get_mode_parameter_dataframe(self, modes, **kws):
        # get frequencies and damping rates
        fg = self.get_kerr_frequencies(modes, **kws)
        modes = indexing.ModeIndexList(modes)
        df = pd.DataFrame()
        for index in modes:
            label = index.get_label()
            df_loc = pd.DataFrame({'f': fg[f'f_{label}'],
                                   'g': fg[f'g_{label}']})
            df_loc['mode'] = label
            df = pd.concat([df, df_loc], ignore_index=True)
        return df

    def get_remnant_parameters(self, f_ref: float | None = None,
                               model: str = 'NRSur7dq4Remnant',
                               nproc: int | None = None,
                               force: bool = False):
        """Compute remnant parameters using the LALSuite nrfits module.

        Arguments
        ---------
        f_ref : float
            Reference frequency for the remnant parameters in Hz; if -1, uses
            earliest point in the waveform; if None, uses the value stored in
            the DataFrame attributes, defaulting to -1 if not found.
        model : str
            Name of the model to use for the remnant parameters; default is
            'NRSur7dq4Remnant'.
        nproc : int | None
            Number of processes to use for parallel computation; if None, uses
            serial computation.
        force : bool
            If True, forces recomputation of the remnant parameters if already
            present in the DataFrame.

        Returns
        -------
        df : IMRResult
            view of DataFrame columns with the remnant parameters: 'final_mass'
            and 'final_spin'.
        """

        keys = ['final_mass', 'final_spin']
        if all([k in self.columns for k in keys]) and not force:
            return self[keys]

        if f_ref is None:
            f_ref = self.reference_frequency

        if nproc is None:
            r = np.vectorize(get_remnant)(self['mass_1']*lal.MSUN_SI,
                                          self['mass_2']*lal.MSUN_SI,
                                          self['spin_1x'], self['spin_1y'],
                                          self['spin_1z'], self['spin_2x'],
                                          self['spin_2y'], self['spin_2z'],
                                          f_ref, model)
        else:
            with mp.Pool(nproc) as p:
                r = p.starmap(get_remnant,
                              zip(self['mass_1']*lal.MSUN_SI,
                                  self['mass_2']*lal.MSUN_SI,
                                  self['spin_1x'], self['spin_1y'],
                                  self['spin_1z'], self['spin_2x'],
                                  self['spin_2y'], self['spin_2z'],
                                  [f_ref]*len(self),
                                  [model]*len(self)))

        r = np.array(r).reshape(len(self), 2)
        self['final_mass'] = r[:, 0] / lal.MSUN_SI
        self['final_spin'] = r[:, 1]
        return self[keys]

    def _get_default_time(self, ref_key='geocent_time'):
        # attempt get trigger time from config file
        # default to median of geocenter time if not found
        tc = self.trigger_time or np.median(self[ref_key])

        fsamp = self.sampling_frequency or self._REFERENCE_SRATE
        dt = 1 / fsamp
        n = (self.duration or 1) * fsamp
        time = np.arange(n)*dt + tc - dt*(n//2)
        if not self.sampling_frequency or not self.duration:
            logging.warning("no time array provided; defaulting to "
                            f"{n*dt} s around {tc} at {1/dt} Hz")
        return time

    @property
    def ifos(self):
        """List of detectors in the DataFrame."""
        # try config first
        if 'detectors' in self.attrs.get('config', {}):
            return self.attrs['config']['detectors']
        time_keys = [k for k in self.columns if TIME_KEY in k]
        return [k.replace(TIME_KEY, '') for k in time_keys]

    def get_peak_times(self, nsamp: int | None = None,
                       ifos: list | None = None,
                       time: np.ndarray | None = None,
                       manual: bool = False,
                       prng: np.random.RandomState | int | None = None,
                       progress: bool = True, **kws) -> pd.DataFrame:
        """Get the peak times of the waveform for a given set of detectors.

        Arguments
        ---------
        nsamp : int | None
            Number of samples to use for the peak time calculation; if None,
            uses all samples in the DataFrame.
        ifos : list of str | None
            List of detector names to use for the peak time calculation; if
            None, uses all detectors in the DataFrame.
        time : np.ndarray | None
            Time array to use for the peak time calculation; if None, uses
            a default time array.
        manual : bool
            If True, estimates the peak time manually from the reconstructed
            waveforms.
        prng : np.random.RandomState | int | None
            Random number generator to use for sampling; if None, uses the
            default random number generator.
        kws : dict
            Additional keyword arguments to pass to the peak
            time calculation.

        Returns
        -------
        peak_times : pd.DataFrame
            DataFrame with the peak times for each detector.
        """
        # subselect samples if requested
        if nsamp is None:
            df = self
        else:
            df = self.sample(nsamp, random_state=prng)

        if ifos is None:
            ifos = self.ifos
        elif isinstance(ifos, str):
            ifos = [ifos]

        if manual:
            # estimate peak time manually from reconstructed waveforms
            if 'geocent_time' in df.columns:
                # use geocenter time as reference
                reference = waveforms.Signal._FROM_GEO_KEY
                ref_key = 'geocent_time'
            else:
                # use first detector as reference
                reference = ifos[0]
                ref_key = f'{reference}{TIME_KEY}'

            if time is None:
                time = self._get_default_time(ref_key)

            if 'reference_frequency' not in kws and 'f_ref' not in kws:
                kws['f_ref'] = self.reference_frequency

            if 'model' not in kws and 'approximant' not in kws:
                kws['model'] = self.approximant

            peak_times_rows = []
            tqdm = get_tqdm(progress)
            for _, sample in tqdm(df.iterrows(), total=len(df), ncols=None,
                                  desc='peak time'):
                h = waveforms.Coalescence.from_parameters(
                    time, **sample, **kws)
                tp = h.get_invariant_peak_time()
                tp_dict = {}
                for ifo in ifos:
                    key = f'{ifo}{TIME_KEY}'
                    if key != ref_key:
                        dt = waveforms.get_delay(ifo, h.t0,
                                                 h.get_parameter('ra'),
                                                 h.get_parameter('dec'),
                                                 reference=reference)
                        tp_dict[ifo] = tp + dt
                    else:
                        tp_dict[ifo] = tp
                peak_times_rows.append(tp_dict)
            return pd.DataFrame(peak_times_rows, index=df.index)
        else:
            # retrieve coalescence time as recorded in samples
            peak_times = {}
            for ifo in ifos:
                key = f'{ifo}{TIME_KEY}'
                if key in df.columns:
                    peak_times[ifo] = df[key]
                else:
                    raise KeyError(f'peak time not found for {ifo}')
            return pd.DataFrame(peak_times)

    def get_best_peak_times(self, average: str = 'median',
                            **kws) -> tuple[pd.Series, str]:
        """Get the peak times corresponding to the average at the best-measured
        detector.

        Arguments
        ---------
        average : str
            Method to use for averaging the peak times; must be 'mean' or
            'median'.
        kws : dict
            Additional keyword arguments to pass to the peak
            time calculation.

        Returns
        -------
        peak_times : pd.Series
            Series with the peak times at the best-measured detector,
            "name" attribute is the index of the sample in the original
            DataFrame.
        best_ifo : str
            Name of the best-measured detector.
        """
        peak_times = self.get_peak_times(**kws)
        # identify best measured peak time
        best_ifo = peak_times.std().idxmin()
        if average == 'mean':
            tp = peak_times[best_ifo].mean()
        elif average == 'median':
            tp = peak_times[best_ifo].median()
        else:
            raise ValueError(f'invalid average method: {average}')
        iloc = (peak_times[best_ifo] - tp).abs().idxmin()
        return peak_times.loc[iloc], best_ifo

    def get_best_peak_target(self, duration='auto', **kws) -> target.SkyTarget:
        """Get the target corresponding to the best-measured peak time.

        Arguments
        ---------
        duration : float
            Duration of the analysis in seconds (optional); if 'auto', uses
            the estimated ringdown duration (default 'auto').
        kws : dict
            Additional keyword arguments to pass to the peak
            time calculation.

        Returns
        -------
        target : target.SkyTarget
            Target constructed from the best-measured peak time.
        """
        peak_times, ref_ifo = self.get_best_peak_times(**kws)
        if duration == 'auto':
            duration = self.estimate_ringdown_duration(**kws)
        sample = self.loc[peak_times.name]
        skyloc = {k: sample[k] for k in ['ra', 'dec', 'psi']}
        t0 = peak_times[ref_ifo]
        return target.Target.construct(t0, reference_ifo=ref_ifo,
                                       duration=duration, **skyloc)

    def get_waveforms(self, nsamp: int | None = None,
                      ifos: list | None = None,
                      time: np.ndarray | None = None,
                      condition: dict | None = None,
                      cache: bool = False,
                      prng: np.random.RandomState | int | None = None,
                      progress: bool = True,
                      **kws) -> data.StrainStack:
        """Get the peak times of the waveform for a given set of detectors.

        Arguments
        ---------
        nsamp : int | None
            Number of samples to use for the peak time calculation; if None,
            uses all samples in the DataFrame.
        ifos : list of str | None
            List of detector names to use for the peak time calculation; if
            None, uses all detectors in the DataFrame.
        time : np.ndarray | None
            Time array to use for the peak time calculation; if None, uses
            a default time array.
        condition : dict | None
            optional conditioning settings; can include a `t0` argument
            which is itself a dictionary for different ifos.
        prng : np.random.RandomState | int | None
            Random number generator to use for sampling; if None, uses the
            default random number generator.
        kws : dict
            Additional keyword arguments to pass to the peak
            time calculation.
        """
        if cache and self._waveforms is not None:
            logging.info("using cached waveforms")
            return self._waveforms

        # subselect samples if requested
        if nsamp is None:
            df = self
        else:
            df = self.sample(nsamp, random_state=prng)

        # get ifos
        if ifos is None:
            ifos = self.ifos
        elif isinstance(ifos, str):
            ifos = [ifos]
        ifos = [ifo for ifo in ifos if 'geo' not in ifo]

        if time is None:
            time = self._get_default_time()

        if 'reference_frequency' not in kws and 'f_ref' not in kws:
            kws['f_ref'] = self.reference_frequency

        if 'model' not in kws and 'approximant' not in kws:
            kws['model'] = self.approximant

        if condition:
            t0 = condition.pop('t0', {})

        wf_dict = {ifo: [] for ifo in ifos}
        tqdm = get_tqdm(progress)
        tqdm_kws = dict(total=len(df), ncols=None, desc='waveforms')
        for _, sample in tqdm(df.iterrows(), **tqdm_kws):
            h = waveforms.get_detector_signals(times=time, ifos=ifos,
                                               **sample, **kws)
            for ifo in ifos:
                if condition:
                    # look for target time 't0' which can be a dict with
                    # entries for each ifo or just a float for all ifos
                    if isinstance(t0, dict):
                        t0 = t0.get(ifo)
                    hi = h[ifo].condition(t0=t0, **condition)
                else:
                    hi = h[ifo]
                wf_dict[ifo].append(hi)
        # waveforms array will be shaped (nifo, nsamp, ntime)
        wfs = np.array([wf_dict[ifo] for ifo in ifos])
        # swap axes to get (nifo, ntime, nsamp)
        h = data.StrainStack(np.swapaxes(wfs, 1, 2))
        if cache:
            logging.info("caching waveforms")
            self._waveforms = h
        elif self._waveforms is not None:
            logging.info("wiping waveform cache")
            self._waveforms = None
        return h

    _FAVORED_APPROXIMANT = 'NRSur7dq4'

    @classmethod
    def from_pesummary(cls, path: str, group: str | None = None,
                       pesummary_read: bool = False,
                       posterior_key: str = 'posterior_samples',
                       attrs=None):
        """Create an IMRResult from a pesummary result.

        Arguments
        ---------
        path : str
            Path to the pesummary result file.
        group : str | None
            Group label to read in from the pesummary result; if None, takes
            the first group in the file.
        pesummary_read : bool
            If True, uses the pesummary reader to read the file; this requires
            the ``pesummary`` package to be installed.
        posterior_keys : str
            Key of group containing posterior samples.

        Returns
        -------
        result : IMRResult
            IMRResult object containing the posterior samples.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"file not found: {path}")

        if pesummary_read:
            try:
                from pesummary.io import read
            except ImportError:
                raise ImportError("missing optional dependency: pesummary")
            pe = read(path)
            if group is None:
                group = pe.labels[0]
                for g in pe.labels:
                    if cls._FAVORED_APPROXIMANT in g:
                        group = g
                        break
                logging.warning(f"no group provided; using {group}")
            config = pe.config.get(group, {}).get('config', {})
            p = {i: data.PowerSpectrum(p).fill_low_frequencies().gate()
                 for i, p in pe.psd.get(group, {}).items()}
            attrs = (attrs or {}).update({'config': config})
            return cls(pe.samples_dict[group], attrs=attrs, psds=p)

        if os.path.splitext(path)[1] in ['.hdf5', '.h5']:
            with h5py.File(path, 'r') as f:
                if group is None:
                    group = list(f.keys())[0]
                    for g in f.keys():
                        if cls._FAVORED_APPROXIMANT in g:
                            group = g
                            break
                    logging.warning(f"no group provided; using {group}")
                if group not in f:
                    raise ValueError(f"group {group} not found")
                c = {k.replace('_', '-'): get_hdf5_value(v[()]) for k, v in
                     f[group].get('config_file', {}).get('config', {}).items()}
                if 'meta_data' in f[group]:
                    if 'other' in f[group]['meta_data']:
                        if 'config_file' in f[group]['meta_data']['other']:
                            x = f[group]['meta_data']['other']['config_file']
                            c.update({k.replace('_', '-'):
                                      get_hdf5_value(v[()])
                                      for k, v in x.items()})
                if 'psds' in f[group]:
                    p = {i: data.PowerSpectrum(p).fill_low_frequencies().gate()
                         for i, p in f[group]['psds'].items()}
                else:
                    p = {}
                if posterior_key in f[group]:
                    attrs = (attrs or {}).update({'config': c})
                    return cls(f[group][posterior_key][()], attrs=attrs,
                               psds=p)
                else:
                    raise ValueError("no {posterior_key} found")
        else:
            raise ValueError(f"unsupported file format: {path}")

    _REFERENCE_SRATE = 16384

    def data_options(self, **options):
        """Return a dictionary of options to obtain data used in the analysis.
        """
        if not self.config:
            return {}
        config = self.config

        if 'ifos' not in options:
            options['ifos'] = self.ifos

        if 'path' not in options and 'channel' not in options \
            and self._data_dict:
            # look for data locally based on config
            path = {}
            for ifo in options['ifos']:
                p = self._data_dict.get(ifo)
                if os.path.isfile(p):
                    logging.info(f"found local data for {ifo}: {p}")
                    path[ifo] = p
                else:
                    logging.info(f"missing local data for {ifo}: {p}")
                    break
            else:
                options['path'] = path
                options['kind'] = 'frame'

        if 'channel' not in options:
            if 'path' in options and options.get('kind') == 'frame':
                options['channel'] = self._channel_dict
            elif 'path' not in options:
                options['channel'] = 'gwosc'

        if 'path' not in options and 'channel' in options:
            # add gwosc specific options
            key_map = {'t0': 'trigger-time', 'seglen': 'duration'}
            for k, v in key_map.items():
                if v in config:
                    options[k] = config[v]
                else:
                    logging.warning(f"missing {v} in config")
            options['sample_rate'] = self._REFERENCE_SRATE
        logging.info(f"using data options: {options}")
        return options

    @property
    def _data_dict(self):
        """Return a dictionary of paths to data used in the analysis."""
        return get_bilby_dict(self.config.get('data-dict', {}))

    @property
    def _channel_dict(self):
        """Return the channel used for the analysis."""
        d = get_bilby_dict(self.config.get('channel-dict', {}))
        return {i.strip(): f'{i.strip()}:{v}' for i, v in d.items()}

    @property
    def condition_options(self):
        """Return a dictionary of options to condition the data used in the
        analysis.
        """
        if 'config' not in self.attrs:
            return {}
        config = self.attrs['config']
        if self.psds:
            sample_rate = self.psds[list(self.psds.keys())[0]].f_samp
            ds = self._REFERENCE_SRATE / sample_rate
        elif 'sampling-frequency' in config:
            sample_rate = config['sampling-frequency']
            ds = self._REFERENCE_SRATE / sample_rate
        else:
            logging.warning("missing sampling frequency in config")
            ds = None
        return {'ds': ds}

    def get_patched_psds(self, f_min: float | None = 0,
                         f_max: float | None = None,
                         max_dynamic_range: float =
                         data.PowerSpectrum._DEF_MAX_DYN_RANGE,
                         **kws) -> dict[data.PowerSpectrum]:
        """Patch the PSDs to the minimum and maximum frequencies used in the
        analysis.

        Arguments
        ---------
        f_min : float
            Minimum frequency to patch to; if None, uses the minimum frequency
            in the analysis.
        f_max : float
            Maximum frequency to patch to; if None, uses the maximum frequency
            in the analysis.

        """
        psds = {}
        for ifo, psd in self.psds.items():
            f_min = f_min or self.minimum_frequency.get(ifo, f_min)
            f_max = f_max or self.maximum_frequency.get(ifo, f_max)
            psds[ifo] = psd.patch(f_min, f_max, **kws).gate(max_dynamic_range)
        return psds

    def get_acfs(self, patch_psd: bool = True, **kws) \
            -> dict[data.AutoCovariance]:
        """Get the AutoCorrelation Functions corresponding to the Power
        Spectral Densities used in the analysis.

        Arguments
        ---------
        patch_psd : bool
            If True, patches the PSDs to the minimum and maximum frequencies
            used in the analysis.
        **kws : dict
            Additional keyword arguments to pass to the PSD computation.
        """
        if patch_psd:
            psds = self.get_patched_psds(**kws)
        else:
            psds = self.psds
        return {ifo: psd.to_acf() for ifo, psd in psds.items()}

    def _ringdown_start_indices(self, time=None, **kws):
        if time is None:
            time = self._get_default_time()
        # check if time has 'get' method
        if isinstance(time, dict):
            time_dict = time
        else:
            time_dict = {i: time for i in self.ifos}
        target = self.get_best_peak_target(duration=0, **kws)
        start_times = target.get_detector_times_dict(self.ifos)
        start_indices = {ifo: np.argmin(np.abs(time_dict[ifo] - t0))
                         for ifo, t0 in start_times.items()}
        return start_indices

    def estimate_ringdown_duration(self, acfs: dict | None = None,
                                   start_indices: dict | None = None,
                                   initial_guess: float | None = None,
                                   nsamp: int = 100,
                                   q: float = 0.1,
                                   return_wfs: bool = False,
                                   acf_kws: dict | None = None,
                                   **kws) -> int:
        """Estimate the duration of the ringdown analysis required to obtain
        stable SNRs.

        Arguments
        ---------
        acfs : dict | None
            ACFs to use for the analysis; if None, computes the ACFs from the
            PSDs.
        start_indices : dict | None
            Start indices for the waveforms; if None, uses the peak times.
        initial_guess : float | None
            Initial guess for the duration of the analysis in seconds; if None,
            estimates based on the mass scale.
        nsamp : int
            Number of posterior draws to use to estimate SNR distribution.
        q : float
            Credible level at which to require stable network SNRs: median SNR
            at the midpoint must be within the ``q`` symmetric CL of the final
            median value.
        return_wfs : bool
            If True, returns the waveforms used to estimate the duration.
        acf_kws : dict | None
            Additional keyword arguments to pass to the ACF computation.
        **kws : dict
            Additional keyword arguments to pass to the waveform computation.

        Returns
        -------
        duration : int
            Estimated duration of the analysis in seconds.
        """
        if acfs is None:
            acfs = self.get_acfs(**(acf_kws or {}))

        if not acfs:
            raise ValueError("ACFs not found")

        if initial_guess is None:
            # estimate based on mass scale
            duration = 50 * self.remnant_mass_scale_reference * qnms.T_MSUN
        else:
            duration = initial_guess

        # get waveforms to compute SNRS
        nsamp = min(nsamp, len(self))
        waveforms = self.get_waveforms(nsamp=nsamp, **kws)

        t = self._get_default_time()
        time = kws.get('time', t)
        # check if time has 'get' method
        if isinstance(time, dict):
            time_dict = time
        else:
            time_dict = {i: time for i in self.ifos}

        # get start times
        if not start_indices:
            start_indices = self._ringdown_start_indices(time_dict, **kws)

        dt = t[1] - t[0]
        n = int(duration // dt)
        duration = n * dt

        # check all acfs have the same dt as the waveforms
        for ifo, acf in acfs.items():
            dt_wf = time_dict[ifo][1] - time_dict[ifo][0]
            if acf.delta_t != dt_wf:
                raise ValueError(f"ACF for {ifo} has different "
                                 "time step than waveforms")
            if dt_wf != dt:
                raise ValueError(f"waveform time step {dt_wf} does not "
                                 f"match requested time step {dt}")

        cholesky = {}
        stable_snr = False
        qs = [(1 - q)/2, 0.5, 1-(1-q)/2]
        while duration < self.duration / 2:
            # update cholesky matrices, waveforms and compute SNRs
            for ifo, acf in acfs.items():
                cholesky[ifo] = acf.iloc[:n].cholesky
            wfs = waveforms.slice(start_indices, n)
            snrs = wfs.compute_snr(cholesky, cumulative=True, network=True)

            # check if SNR at midpoint is within bounds
            snr_l, snr_m, snr_h = np.quantile(snrs[-1, :], qs)
            snr_halfway = np.median(snrs[len(snrs)//2, :])
            stable_snr = np.abs(snr_halfway - snr_m) < snr_h - snr_l
            if stable_snr:
                break
            n *= 2
            duration = n * dt

        if not stable_snr:
            logging.warning("SNR not stable; returning maximum duration")

        if return_wfs:
            return duration, wfs
        return duration

    def estimate_ringdown_prior(self, a_scale_factor: float = 50,
                                frequency_scale_factor: float = 10,
                                reference_cl: float = 0.99,
                                modes: ModeIndexList | str | None = None,
                                start_indices: dict | None = None,
                                nsamp: float = 100, **kws):
        """Estimate the prior settings for a ringdown analysis.

        Arguments
        ---------
        a_scale_factor : float
            scale by which to multiply maximum strain in order to obtain
            maximum amplitude scale for prior.
        frequency_scale_factor : float
            scale by which to multiply the posterior range for the remnant
            mass (if Kerr modes are passed) or the 220 mode frequency (if
            generic modes are passed) to obtain the prior range; this works
            by expanding the IMR posterior range (computed at the
            ``reference_cl`` level) by this factor on each side.
        reference_cl : float
            Credible level at which to compute the posterior range for the
            remnant mass or 220 mode to be used as reference for mass or
            frequency prior.
        modes : ModeIndexList | str | None
            List of modes to use to decide how to set frequency or mass prior;
            if Kerr indices are passed, sets a mass prior; if 'generic' index
            is passed, sets a frequency prior; if None, returns amplitude
            settings only.
        start_indices : dict | None
            Start indices for the waveforms; if None, uses the peak times.
        nsamp : int
            Number of samples to draw from posterior.

        Returns
        -------
        opts : dict
            Dictionary of prior settings.
        """
        waveforms = self.get_waveforms(nsamp=nsamp, **kws)
        if start_indices is None:
            start_indices = self._ringdown_start_indices(nsamp=nsamp, **kws)
        # get value of waveforms at the start of the analysis segment
        h = waveforms.slice(start_indices, n=1)
        opts = {
            'a_scale_max': a_scale_factor * float(np.max(np.abs(h)))
        }
        # get mode-dependent settings
        if not modes:
            # we have no mode information so just return amplitude settings
            return opts
        elif isinstance(modes, ModeIndexList):
            generic_modes = modes.is_generic
        else:
            generic_modes = modes == 'generic' or isinstance(modes, int)

        q = 0.5*(1 - reference_cl)
        qs = [q, 0.5, 1-q]
        if generic_modes:
            logging.info("estimating frequency prior based on 220 mode")
            mode = (1, -2, 2, 2, 0)
            f = self.sample(nsamp).get_kerr_frequencies([mode])['f_220']
            l, m, h = np.quantile(f, qs)
            logging.info(f"quantiles {qs}: {l}, {m}, {h}")
            fmax = 0.5 * self.sampling_frequency
            fmin = 1 / self.duration
            for k in ['f', 'g']:
                opts.update({
                    f'{k}_max': min(2*m, m + frequency_scale_factor*(h - m)),
                    f'{k}_min': max(m/2, m - frequency_scale_factor*(m - l))
                })
                # round to nearest integer
                for kk in [f'{k}_min', f'{k}_max']:
                    opts[kk] = np.round(opts[kk])
                # check values for safety
                if opts[f'{k}_max'] >= fmax:
                    logging.warning("upper frequency bound set to Nyquist")
                    opts[f'{k}_max'] = fmax
                if opts[f'{k}_min'] <= fmin:
                    logging.warning("lower frequency bound set to 1/T")
                    opts[f'{k}_min'] = fmin
        else:
            logging.info("estimating mass prior")
            l, m, h = np.quantile(self.remnant_mass_scale, qs)
            logging.info(f"quantiles {qs}: {l}, {m}, {h}")
            opts.update({
                'm_max': np.ceil(m + frequency_scale_factor*(h - m)),
                'm_min': np.floor(max(m/2, m - frequency_scale_factor*(m - l)))
            })
        return opts

    @classmethod
    def construct(cls, path, psds=None, reference_frequency=None,
                  approximant=None, **kws):
        # get attributes from keyword arguments
        info = {k: v for k, v in locals().items() if k != 'cls'}
        info.update(info.pop('attrs', {}))
        info.update(info.pop('kws', {}))
        info['path'] = str(path)
        attrs = {'construct' : info}
        
        if isinstance(path, str):
            try:
                r = cls.from_pesummary(path, attrs=attrs, **kws)
            except Exception as e:
                logging.warning(f"failed to read pesummary file: {e}")
                r = pd.read_hdf(path,**kws)
            path = os.path.abspath(path)
            r.attrs['path'] = path
            r.attrs.update(attrs)
        else:
            r = cls(path, attrs=attrs, **kws)
        if psds is not None:
            r.set_psds(psds)
        if approximant is not None:
            r.set_approximant(approximant)
        if reference_frequency is not None:
            r.set_reference_frequency(reference_frequency)
        return r

    @property
    def path(self):
        """Path to the file from which the result was read."""
        return self.attrs.get('path', '')

    @classmethod
    def from_config(cls, config_input, overwrite_data=False,
                    imr_sec='imr', **kws):
        """Create an IMRResult from a configuration file."""
        config = utils.load_config(config_input)

        if not config.has_section(imr_sec):
            logging.warning("no IMR section found in config")
            return cls()

        # get imr options
        imr = {k: utils.try_parse(v) for k, v in config[imr_sec].items()
               if k != 'initialize_fit'}

        # get path to IMR result file
        if 'path' not in imr and not 'imr_result' in imr:
            raise ValueError("no path to IMR result provided")
        imr_path = imr.pop('path', imr.pop('imr_result', None))

        # check if we should overwrite data based on 'data' section
        overwrite_data |= config.get('data', 'overwrite_data', fallback=False)
        overwrite_data &= config.has_section('data')
        
        if overwrite_data:
            logging.info("loading data from disk (ignoring IMR data)")
            data_kws = {k: utils.try_parse(v)
                        for k, v in config['data'].items()}
            if 'ifos' in config['data']:
                data_kws['ifos'] = utils.get_ifo_list(config, 'data')
        else:
            data_kws = {}
        
        logging.info("loading IMR result")
        return cls.construct(imr_path, **data_kws, **imr, **kws)
