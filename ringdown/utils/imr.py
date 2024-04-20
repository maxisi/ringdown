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
            
    def get_kerr_frequencies(self, modes, **kws):
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
    
    def compute_remnant_parameters(self, f_ref=-1, model='NRSur7dq4Remnant',
                                   nproc=None, suppress_warnings=True, 
                                   force=False):
        keys = ['final_mass', 'final_spin']
        if all([k in self.columns for k in keys]) and not force:
            return self[keys]
        
        filter = 'ignore' if suppress_warnings else 'default'

        if nproc is None:
            r = np.vectorize(get_remnant)(self['mass_1']*lal.MSUN_SI, 
                                          self['mass_2']*lal.MSUN_SI,
                                          self['spin_1x'], self['spin_1y'],
                                          self['spin_1z'], self['spin_2x'],
                                          self['spin_2y'], self['spin_2z'],
                                          f_ref, model)
        else:
            with mp.Pool(nproc) as p:
                r = p.starmap(get_remnant, zip(self['mass_1']*lal.MSUN_SI, 
                                               self['mass_2']*lal.MSUN_SI,
                                               self['spin_1x'], self['spin_1y'],
                                               self['spin_1z'], self['spin_2x'],
                                               self['spin_2y'], self['spin_2z'],
                                               [f_ref]*len(self),
                                               [model]*len(self)))
                
        r = np.array(r)
        self['final_mass'] = r[:,0] / lal.MSUN_SI
        self['final_spin'] = r[:,1]
        return self[keys]