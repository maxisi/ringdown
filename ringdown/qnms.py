__all__ = ['construct_mode_list', 'construct_mode_coordinates',
           'KerrMode', 'get_ftau']

import numpy as np
import qnm
import lal
from collections import namedtuple
from . import utils
import logging

T_MSUN = lal.GMSUN_SI / lal.C_SI**3

def get_mode_label(mode, **kws):
    return GenericModeIndex(mode).get_label(**kws)

def get_mode_coordinate(mode, **kws):
    return GenericModeIndex(mode).get_coordinate()

def construct_mode_coordinates(modes : int | list) -> list:
    """Construct mode indices for InferenceData object.

    Arguments
    ---------
    modes : int or list
        Number of modes or list of mode indices.

    Returns
    -------
    idxs : list
        List of mode indices.
    """
    try:
        idxs = list(range(int(modes)))
    except TypeError:
        try:
            mode_list = [GenericModeIndex(m) for m in modes]
            idxs = [m.get_coordinate() for m in mode_list]
        except Exception:
            raise ValueError(f"Could not parse modes: {modes}")
    return idxs
        
class GenericModeIndex(object):
    def __init__(self, *args):
        try:
            self.index = ModeIndex.construct(*args)
        except Exception:
            self.index =int(args[0])
 
    def __str__(self):
        return str(self.index)
    
    def __repr__(self):
        return f'GenericModeIndex(index={self.index})'

    def get_label(self, **kws):
        if isinstance(self.index, ModeIndex):
            return self.index.to_label(**kws)
        else:
            return str(self.index)
        
    def get_coordinate(self):
        if isinstance(self.index, ModeIndex):
            return self.index.to_bytestring()
        else:
            return int(self.index)
        
ModeIndexBase = namedtuple('ModeIndex', ['p', 's', 'l', 'm', 'n'])

class ModeIndex(ModeIndexBase):
    @classmethod
    def from_string(cls, string):
        if ',' in string:
            p, s, l, m, n = map(int, string.split(','))
            return cls(p, s, l, m, n)
        else:
            # Try to parse old-style lmn strings:
            idxs = utils.string_to_tuple(string)
            if len(idxs) == 3:
                logging.warning("Assuming prograde and spin weight -2 "
                                f"for mode index: {string}")
                l, m, n = idxs
                p, s = 1, -2
            elif len(idxs) == 4:
                logging.warning("Assuming spin weight -2 for mode index: "
                                f"{string}")
                p, l, m, n = idxs
                s = -2
            elif len(idxs) == 5:
                p, s, l, m, n = idxs
            else:
                raise ValueError(f"Could not parse mode index: {string}")
            return cls(p, s, l, m, n)
    
    @classmethod
    def from_bytestring(cls, s):
        return cls.from_string(s.decode('utf-8'))
    
    @classmethod
    def construct(cls, s):
        if isinstance(s, ModeIndex):
            return s
        elif isinstance(s, bytes):
            return cls.from_bytestring(s)
        elif isinstance(s, str):
            return cls.from_string(s)
        else:
            return cls(*s)

    def to_bytestring(self):
        s = f'{self.p},{self.s},{self.l},{self.m},{self.n}'
        return bytes(s, 'utf-8')
    
    def to_label(self, include_prograde=True, include_spinweight=False):
        s = f'{self.l}{self.m}{self.n}'
        if include_spinweight:
            s = f'{self.s}{s}'
        if include_prograde:
            s = f'{self.p}{s}'
        return s

def construct_mode_list(modes : str | None) -> list:
    if modes is None:
        modes = []
    elif isinstance(modes, str):
        # assume modes is a string like "(p0,s0,l0,m0,n0),(p1,s1,l1,m1,n1)"
        from ast import literal_eval
        modes = literal_eval(modes)
    mode_list = []
    for (p, s, l, m, n) in modes:
        mode_list.append(ModeIndex(p, s, l, m, n))
    return mode_list

class KerrMode(object):

    _cache = {}

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]
        self.index = ModeIndex(*args, **kwargs)

    @property
    def coefficients(self):
        i = self.index
        if i not in self._cache:
            self._cache[i] = self.compute_coefficients(i)
        return self._cache[i]

    @staticmethod
    def compute_coefficients(mode, n_chi=1000, **kws):
        p, s, l, m, n = mode
        chis = np.linspace(0, 1, n_chi)[:-1]
        logchis = np.log1p(-chis)
        M = np.column_stack((chis, np.ones_like(chis), logchis, logchis**2,
                             logchis**3, logchis**4))

        q = qnm.modes_cache(s, l, p*abs(m), n)
        sgn = 1 if m == 0 else np.sign(m)
        f = sgn*np.array([q(c)[0].real for c in chis])/(2*np.pi)
        g = np.array([abs(q(c)[0].imag) for c in chis])

        coeff_f = np.linalg.lstsq(M, f, rcond=None, **kws)[0]
        coeff_g = np.linalg.lstsq(M, g, rcond=None, **kws)[0]
        return coeff_f, coeff_g

    def __call__(self, *args, **kwargs):
        f, tau = self.ftau(*args, **kwargs)
        return 2*np.pi*f - 1j/tau

    def ftau(self, chi, m_msun=None, approx=False):
        if approx:
            logchi = np.log1p(-chi)
            c = (chi, np.ones_like(chi), logchi, logchi**2, logchi**3, logchi**4)
            f, g = [np.dot(coeff, c) for coeff in self.coefficients]
        else:
            p, s, l, m, n = self.index
            q = qnm.modes_cache(s, l, p*abs(m), n)
            def omega(c):
                return q(c)[0]
            f = np.sign(m)*np.vectorize(omega)(chi).real/(2*np.pi)
            g = abs(np.vectorize(omega)(chi).imag)
        if m_msun:
           f /= (m_msun * T_MSUN)
           g /= (m_msun * T_MSUN)
            
        return f, 1/g

def get_ftau(M, chi, n, l=2, m=2):
    q22 = qnm.modes_cache(-2, l, m, n)
    omega, _, _ = q22(a=chi)
    f = np.real(omega)/(2*np.pi) / (T_MSUN*M)
    gamma = abs(np.imag(omega)) / (T_MSUN*M)
    return f, 1./gamma

