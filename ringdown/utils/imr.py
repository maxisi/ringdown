import numpy as np
import pandas as pd
from .. import indexing
from .. import qnms

MASS_ALIASES = ['final_mass', 'mf', 'mfinal', 'm_final', 'final_mass_source',
                'remnant_mass']
SPIN_ALIASES = ['final_spin', 'remnant_spin', 'chif', 'chi_f', 'chi_final',
                'af', 'a_final']

class IMRResult(pd.DataFrame):
    
    _f_key = 'f_{mode}'
    _g_key = 'g_{mode}'

    @property
    def _constructor(self):
        return pd.DataFrame
    
    @property
    def final_mass(self):
        for k in MASS_ALIASES:
            if k in self.columns:
                return self[k]
            
    @property
    def final_spin(self):
        for k in SPIN_ALIASES:
            if k in self.columns:
                return self[k]
            
    def get_kerr_frequencies(self, *modes, **kws):
        modes = indexing.ModeIndexList(*modes)
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
    
    def compute_remnant_parameters(self, model='NRSur7dq4Remnant'):    
        from lalsimulation import nrfits
        fit_types_list=["FinalMass", "FinalSpin"]
        def get_remnant(mass_1, mass_2, spin_1x, spin_1y, spin_1z, 
                        spin_2x, spin_2y, spin_2z, f_ref):
            remnant = nrfits.eval_nrfit(mass_1, mass_2,
                                        [spin_1x, spin_1y, spin_1z], 
                                        [spin_2x, spin_2y, spin_2z],
                                        model, f_ref=f_ref,
                                        fit_types_list=fit_types_list)
            return remnant['FinalMass'], np.linalg.norm(remnant['FinalSpin'])

        remnant = np.vectorize(get_remnant)(self['mass_1'], self['mass_2'],
                                            self['spin_1x'], self['spin_1y'],
                                            self['spin_1z'], self['spin_2x'],
                                            self['spin_2y'], self['spin_2z'],
                                            self['f_ref'])
        remnant = np.array(remnant)
        self['final_mass'] = remnant[:,0]
        self['final_spin'] = remnant[:,1]
        return self[['final_mass', 'final_spin']]