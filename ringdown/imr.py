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
from .utils import get_tqdm, get_hdf5_value
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
    def psds(self) -> dict:
        """Power Spectral Densities used in the analysis."""
        return self.__dict__['_psds'] or {}

    def set_psds(self, psds: dict) -> None:
        """Set the PSDs used in the analysis."""
        for i, p in psds.items():
            self.__dict__['_psds'][i] = data.PowerSpectrum(p)

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
        tc = np.median(self[ref_key])
        dt = 1 / 16384
        n = 1 / dt + 1
        time = np.arange(n)*dt + tc - dt*(n//2)
        logging.warning("no time array provided; defaulting to "
                        f"{time[-1]-time[0]} s around {tc} at "
                        f"{1/dt} Hz")
        return time

    @property
    def _ifos(self):
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
            ifos = self._ifos
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
            for _, sample in tqdm(df.iterrows(), total=len(df), ncols=None):
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
        iloc = (peak_times[best_ifo] - tp).idxmin()
        return peak_times.loc[iloc], best_ifo

    def get_best_peak_target(self, **kws) -> target.SkyTarget:
        """Get the target corresponding to the best-measured peak time.

        Arguments
        ---------
        kws : dict
            Additional keyword arguments to pass to the peak
            time calculation.

        Returns
        -------
        target : target.SkyTarget
            Target constructed from the best-measured peak time.
        """
        peak_times, ref_ifo = self.get_best_peak_times(**kws)
        sample = self.loc[peak_times.name]
        skyloc = {k: sample[k] for k in ['ra', 'dec', 'psi']}
        t0 = peak_times[ref_ifo]
        return target.Target.construct(t0, reference_ifo=ref_ifo, **skyloc)

    def get_waveforms(self, nsamp: int | None = None,
                      ifos: list | None = None,
                      time: np.ndarray | None = None,
                      condition: dict | None = None,
                      prng: np.random.RandomState | int | None = None,
                      progress: bool = True,
                      **kws) -> pd.DataFrame:
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
        # subselect samples if requested
        if nsamp is None:
            df = self
        else:
            df = self.sample(nsamp, random_state=prng)

        # get ifos
        if ifos is None:
            ifos = self._ifos
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
        return data.StrainStack(np.swapaxes(wfs, 1, 2))

    @classmethod
    def from_pesummary(cls, path, group: str | None = None,
                       pesummary_read: bool = False,
                       posterior_key='posterior_samples'):
        """Create an IMRResult from a pesummary result.
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
                logging.warning(f"no group provided; using {group}")
            config = pe.config.get(group, {}).get('config', {})
            p = {i: data.PowerSpectrum(p).fill_low_frequencies().gate()
                 for i, p in pe.psd.get(group, {}).items()}
            return cls(pe.samples_dict[group], attrs={'config': config},
                       psds=p)

        if os.path.splitext(path)[1] in ['.hdf5', '.h5']:
            with h5py.File(path, 'r') as f:
                if group is None:
                    group = list(f.keys())[0]
                    logging.warning(f"no group provided; using {group}")
                if group not in f:
                    raise ValueError(f"group {group} not found")
                c = {k: get_hdf5_value(v[()]) for k, v in
                     f[group].get('config_file', {}).get('config', {}).items()}
                if 'psds' in f[group]:
                    p = {i: data.PowerSpectrum(p).fill_low_frequencies().gate()
                         for i, p in f[group]['psds'].items()}
                else:
                    p = {}
                if posterior_key in f[group]:
                    return cls(f[group][posterior_key][()],
                               attrs={'config': c}, psds=p)
                else:
                    raise ValueError("no {posterior_key} found")

    _REFERENCE_SRATE = 16384

    @property
    def data_options(self):
        """Return a dictionary of options to obtain data used in the analysis.
        """
        if 'config' not in self.attrs:
            return {}
        config = self.attrs['config']
        key_map = {'t0': 'trigger-time', 'ifos': 'detectors',
                   'seglen': 'duration'}
        options = {}
        for k, v in key_map.items():
            if v in config:
                options[k] = config[v]
            else:
                logging.warning(f"missing {v} in config")
        options['sample_rate'] = self._REFERENCE_SRATE
        return options

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

    def get_patched_psds(self, f_min=0, f_max=None):
        c = self.attrs.get('config', {})
        psds = {}
        for ifo, psd in self.psds.items():
            f_min = f_min or c.get('minimum-frequency', {}).get(ifo, f_min)
            f_max = f_max or c.get('maximum-frequency', {}).get(ifo, f_max)
            psds[ifo] = psd.patch(f_min, f_max).gate()
        return psds

    def get_acfs(self, patch_psd=True):
        if patch_psd:
            psds = self.get_patched_psds()
        else:
            psds = self.psds
        return {ifo: psd.to_acf() for ifo, psd in psds.items()}
