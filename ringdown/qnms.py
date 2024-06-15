__all__ = ['KerrMode', 'get_ftau']

import numpy as np
import qnm
import lal
from . import indexing
from .utils import docstring_parameter

T_MSUN = lal.GMSUN_SI / lal.C_SI**3


def get_ftau(M, chi, n, l=2, m=2):  # noqa: E741
    """Get the frequency and damping time of a Kerr quasinormal mode.

    This is a wrapper around the package `qnm`.

    Arguments
    ---------
    M : float
        Black hole mass in solar masses.
    chi : float
        Dimensionless spin parameter.
    n : int
        Overtone number.
    l : int
        Spherical harmonic index (def. 2)
    m : int
        Azimuthal harmonic index (def. 2)

    Returns
    -------
    f : float
        Frequency of the mode in Hz.
    tau : float
        Damping time of the mode in seconds.
    """
    q22 = qnm.modes_cache(-2, l, m, n)
    omega, _, _ = q22(a=chi)
    f = np.real(omega)/(2*np.pi) / (T_MSUN*M)
    gamma = abs(np.imag(omega)) / (T_MSUN*M)
    return f, 1./gamma


class KerrMode(object):
    """A Kerr quasinormal mode.
    """

    _cache = {}

    @docstring_parameter(indexing.HarmonicIndex.construct.__doc__)
    def __init__(self, *args, **kwargs):
        """All arguments are passed to `indexing.ModeIndex.construct`,
        in order to identify the mode index (p, s, l, m, n) from
        a string, tuple or some other input.

        Docs for `indexing.ModeIndex.construct`:

        {0}
        """
        if len(args) == 1:
            args = args[0]
        self.index = indexing.HarmonicIndex.construct(*args, **kwargs)

    @property
    def coefficients(self):
        i = tuple(self.index)
        if i not in self._cache:
            self._cache[i] = self.compute_coefficients(i)
        return self._cache[i]

    @staticmethod
    def compute_coefficients(mode, n_chi=4096, **kws):
        p, s, l, m, n = mode
        # chis = np.linspace(0, 1, n_chi, endpoint=False)
        # logchis = np.log1p(-chis)
        logchis = np.linspace(0, -10, n_chi)
        chis = 1 - np.exp(logchis)
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

    def fgamma(self, chi, m_msun=None, approx=False):
        if approx:
            logchi = np.log1p(-chi)
            c = (chi, np.ones_like(chi), logchi,
                 logchi**2, logchi**3, logchi**4)
            f, g = [np.dot(coeff, c) for coeff in self.coefficients]
        else:
            p, s, l, m, n = self.index
            q = qnm.modes_cache(s, l, p*abs(m), n)

            def omega(c):
                return q(c)[0]
            f = np.sign(m)*np.vectorize(omega)(chi).real/(2*np.pi)
            g = abs(np.vectorize(omega)(chi).imag)
        if m_msun is not None:
            f /= (m_msun * T_MSUN)
            g /= (m_msun * T_MSUN)
        return f, g

    def ftau(self, chi, m_msun=None, approx=False):
        f, g = self.fgamma(chi, m_msun, approx)
        return f, 1/g


class ParameterLabel(object):

    _PARAMETER_KEY_MAP = {
        'm': '$M / M_\\odot$',
        'chi': '$\\chi$',
        'f': '$f_{{{mode}}} / \\mathrm{{Hz}}$',
        'g': '$\\gamma_{{{mode}}} / \\mathrm{{Hz}}$',
        'df': '$\\delta f_{{{mode}}}$',
        'dg': '$\\delta \\gamma_{{{mode}}}$',
        'a': '$A_{{{mode}}}$',
        'phi': '$\\phi_{{{mode}}}$',
        'theta': '$\\theta_{{{mode}}}$',
        'ellip': '$\\epsilon_{{{mode}}}$',
        'h_det': '$h(t) [\\mathrm{{{ifo}}}]$',
        'h_det_mode': '$h_{{{mode}}}(t) [\\mathrm{{{ifo}}}]$',
    }

    def __init__(self, parameter):
        self.parameter = parameter.lower()
        if self.parameter not in self._PARAMETER_KEY_MAP:
            raise ValueError(f"Parameter {parameter} not recognized.")

    def __str__(self):
        return self.parameter

    def __repr__(self):
        return (f"ParameterLabel('{self.parameter}')")

    @property
    def is_mode_specific(self):
        label = self._PARAMETER_KEY_MAP[self.parameter]
        return '{{{mode}}}' in label

    @property
    def is_strain(self):
        return self.parameter.startswith('h_det')

    def get_latex(self, mode=None, ifo=None, **kws):
        label = self._PARAMETER_KEY_MAP[self.parameter]
        subst = {}
        if mode is not None:
            mode_index = indexing.get_mode_label(mode, **kws)
            subst['mode'] = mode_index
        elif self.is_mode_specific:
            label = label.replace('_{{{mode}}}', '')
        if ifo is not None:
            subst['ifo'] = ifo
        else:
            label = label.replace(' [\\mathrm{{{ifo}}}]', '')
        return label.format(**subst)

    def get_key(self, mode=None, ifo=None, **kws):
        key = self.parameter
        if mode is not None:
            mode_index = indexing.get_mode_label(mode, **kws)
            if key == 'h_det_mode':
                key = key.replace('mode', mode_index)
            elif self.is_mode_specific:
                key = f'{key}_{mode_index}'
        if ifo is not None:
            key = key.replace('det', ifo)
        return key

    def get_label(self, latex=False, **kws):
        if latex:
            return self.get_latex(**kws)
        else:
            return self.get_key(**kws)


def get_parameter_label_map(pars=None, modes=None, ifos=None, **kws):
    label_dict = {}
    pars = pars or ParameterLabel._PARAMETER_KEY_MAP.keys()
    if modes is None:
        modes = [None]
    if ifos is None:
        ifos = [None]
    for k in pars:
        p = ParameterLabel(k)
        for i in ifos:
            for m in modes:
                label_dict[p.get_key(mode=m, ifo=i, **kws)] = \
                    p.get_latex(mode=m, ifo=i, **kws)
    return label_dict
