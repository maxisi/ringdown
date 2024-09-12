__all__ = ['IMRResult']

import numpy as np
import pandas as pd
from .. import indexing
from .. import qnms
import lal
import multiprocessing as mp
from lalsimulation import nrfits

MASS_ALIASES = ['final_mass', 'mf', 'mfinal', 'm_final', 'final_mass_source',
                'remnant_mass']
SPIN_ALIASES = ['final_spin', 'remnant_spin', 'chif', 'chi_f', 'chi_final',
                'af', 'a_final']


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
