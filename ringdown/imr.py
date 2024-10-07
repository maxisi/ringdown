__all__ = ['IMRResult']

import numpy as np
import pandas as pd
from . import indexing
from . import qnms
from . import waveforms
from . import target
from .utils import get_tqdm
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

    @property
    def _constructor(self):
        return IMRResult

    @property
    def final_mass(self) -> np.ndarray:
        for k in MASS_ALIASES:
            if k in self.columns:
                return self[k]

    @property
    def final_spin(self) -> np.ndarray:
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

    def get_remnant_parameters(self, f_ref: float = -1,
                               model: str = 'NRSur7dq4Remnant',
                               nproc: int | None = None,
                               force: bool = False):
        """Compute remnant parameters using the LALSuite nrfits module.

        Arguments
        ---------
        f_ref : float
            Reference frequency for the remnant parameters in Hz; if -1, uses
            earliest point in the waveform.
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

        wf_dict = {ifo: [] for ifo in ifos}
        tqdm = get_tqdm(progress)
        for _, sample in tqdm(df.iterrows(), total=len(df), ncols=None):
            h = waveforms.get_detector_signals(times=time, ifos=ifos,
                                               **sample, **kws)
            for ifo in ifos:
                wf_dict[ifo].append(h[ifo])
        # waveforms array will be shaped (nifo, nsamp, ntime)
        wfs = np.array([wf_dict[ifo] for ifo in ifos])
        # swap axes to get (nifo, ntime, nsamp)
        return np.swapaxes(wfs, 1, 2)
