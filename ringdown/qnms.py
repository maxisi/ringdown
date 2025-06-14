__all__ = ["KerrMode", "get_ftau"]

import numpy as np
import qnm
from . import indexing
from .utils import docstring_parameter
from .config import T_MSUN


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
    f = np.real(omega) / (2 * np.pi) / (T_MSUN * M)
    gamma = abs(np.imag(omega)) / (T_MSUN * M)
    return f, 1.0 / gamma


class KerrMode(object):
    """A Kerr quasinormal mode."""

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
    def compute_coefficients(mode, **kws):
        p, s, l, m, n = mode
        sgn = p if m == 0 else p * np.sign(m)
        q = qnm.modes_cache(s, l, p * abs(m), n)

        # Only use spins pre-computed by qnm package
        chis = np.array(q.a)
        log_1m_chis = np.log1p(-chis)
        log_1m_chis_2 = log_1m_chis * log_1m_chis
        log_1m_chis_3 = log_1m_chis_2 * log_1m_chis
        log_1m_chis_4 = log_1m_chis_2 * log_1m_chis_2
        log_sqrt_1m_chis2 = 0.5 * np.log1p(-(chis**2))
        log_sqrt_1m_chis2_2 = log_sqrt_1m_chis2 * log_sqrt_1m_chis2
        log_sqrt_1m_chis2_3 = log_sqrt_1m_chis2_2 * log_sqrt_1m_chis2
        log_sqrt_1m_chis2_4 = log_sqrt_1m_chis2_2 * log_sqrt_1m_chis2_2
        log_sqrt_1m_chis2_5 = log_sqrt_1m_chis2_3 * log_sqrt_1m_chis2_2
        log_sqrt_1m_chis2_6 = log_sqrt_1m_chis2_3 * log_sqrt_1m_chis2_3

        M = np.column_stack(
            (
                np.ones_like(log_1m_chis),
                log_1m_chis,
                log_1m_chis_2,
                log_1m_chis_3,
                log_1m_chis_4,
                log_sqrt_1m_chis2,
                log_sqrt_1m_chis2_2,
                log_sqrt_1m_chis2_3,
                log_sqrt_1m_chis2_4,
                log_sqrt_1m_chis2_5,
                log_sqrt_1m_chis2_6,
            )
        )

        f = sgn * np.array([q(chi)[0].real for chi in chis]) / (2 * np.pi)
        g = np.array([abs(q(chi)[0].imag) for chi in chis])

        coeff_f = np.linalg.lstsq(M, f, rcond=None, **kws)[0]
        coeff_g = np.linalg.lstsq(M, g, rcond=None, **kws)[0]

        return coeff_f, coeff_g

    def __call__(self, *args, **kwargs):
        f, tau = self.ftau(*args, **kwargs)
        return 2 * np.pi * f - 1j / tau

    def fgamma(self, chi, m_msun=None, approx=False):
        if approx:
            log_1m_chi = np.log1p(-chi)
            log_1m_chi_2 = log_1m_chi * log_1m_chi
            log_1m_chi_3 = log_1m_chi_2 * log_1m_chi
            log_1m_chi_4 = log_1m_chi_2 * log_1m_chi_2
            log_sqrt_1m_chi2 = 0.5 * np.log1p(-(chi**2))
            log_sqrt_1m_chi2_2 = log_sqrt_1m_chi2 * log_sqrt_1m_chi2
            log_sqrt_1m_chi2_3 = log_sqrt_1m_chi2_2 * log_sqrt_1m_chi2
            log_sqrt_1m_chi2_4 = log_sqrt_1m_chi2_2 * log_sqrt_1m_chi2_2
            log_sqrt_1m_chi2_5 = log_sqrt_1m_chi2_3 * log_sqrt_1m_chi2_2
            log_sqrt_1m_chi2_6 = log_sqrt_1m_chi2_3 * log_sqrt_1m_chi2_3

            v = np.stack(
                [
                    1.0,
                    log_1m_chi,
                    log_1m_chi_2,
                    log_1m_chi_3,
                    log_1m_chi_4,
                    log_sqrt_1m_chi2,
                    log_sqrt_1m_chi2_2,
                    log_sqrt_1m_chi2_3,
                    log_sqrt_1m_chi2_4,
                    log_sqrt_1m_chi2_5,
                    log_sqrt_1m_chi2_6,
                ]
            )

            f, g = [np.dot(coeff, v) for coeff in self.coefficients]
        else:
            p, s, l, m, n = self.index
            q = qnm.modes_cache(s, l, p * abs(m), n)

            def omega(c):
                return q(c)[0]

            f = np.sign(m) * np.vectorize(omega)(chi).real / (2 * np.pi)
            g = abs(np.vectorize(omega)(chi).imag)
        if m_msun is not None:
            f /= m_msun * T_MSUN
            g /= m_msun * T_MSUN
        return f, g

    def ftau(self, chi, m_msun=None, approx=False):
        f, g = self.fgamma(chi, m_msun, approx)
        return f, 1 / g
