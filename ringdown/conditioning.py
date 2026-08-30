"""Conditioning diagnostics for multimode ringdown fits.

When two quasinormal modes in a fit model lie close in the complex
frequency plane, extracting them from data becomes badly conditioned, and
how badly depends on what is being estimated. Writing gap for the distance
between the two mode frequencies, the error amplification scales as

- amplitudes with frequencies fixed by (M, chi):  1 / gap
- frequencies themselves (Fisher):                1 / gap^2
- amplitudes with free frequencies:               1 / gap^3

The first exponent was measured directly on SXS waveforms across the
(2, 2, 5) and (2, 2, 6) avoided crossing, and the task hierarchy was
measured in a controlled synthetic study; see
https://github.com/maiconburn/recoverability-criticality
(DOI 10.5281/zenodo.22156019). The frequency case matches the Fisher
forecast scaling of arXiv:2605.16199, and the free-frequency amplitude
case matches the classical super-resolution scaling of Prony-type
estimation.

These diagnostics are meant to be run before or alongside a fit, to flag
mode pairs that the data cannot cleanly separate at the target spin.
"""

__all__ = ["TASK_EXPONENTS", "mode_gaps", "conditioning_report"]

import numpy as np

from .qnms import KerrMode

TASK_EXPONENTS = {
    "amplitudes_fixed_frequencies": 1,
    "free_frequencies": 2,
    "amplitudes_free_frequencies": 3,
}


def _omegas(modes, chi, m_msun=None, approx=False):
    kerr_modes = [KerrMode(m) for m in modes]
    return [complex(k(chi, m_msun, approx)) for k in kerr_modes]


def mode_gaps(modes, chi, m_msun=None, approx=False):
    """Pairwise distances between mode frequencies in the complex plane.

    Arguments
    ---------
    modes : list
        Mode identifiers accepted by :class:`ringdown.qnms.KerrMode`,
        e.g. ``[(1, -2, 2, 2, 0), (1, -2, 2, 2, 1)]``.
    chi : float
        Dimensionless spin of the remnant.
    m_msun : float, optional
        Remnant mass in solar masses; if given, gaps are in angular
        frequency units of 1/s, otherwise they are dimensionless.
    approx : bool, optional
        Use the fitting-coefficient approximation of `KerrMode`.

    Returns
    -------
    gaps : dict
        Mapping ``(mode_i, mode_j) -> |omega_i - omega_j|``.
    """
    omegas = _omegas(modes, chi, m_msun, approx)
    gaps = {}
    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            gaps[(tuple(modes[i]), tuple(modes[j]))] = \
                abs(omegas[i] - omegas[j])
    return gaps


def conditioning_report(modes, chi, m_msun=None, approx=False):
    """Error-amplification report for every mode pair in a fit model.

    For each pair this returns the gap and the amplification factors
    ``gap ** -p`` for the three estimation tasks in
    :data:`TASK_EXPONENTS`. The factors are relative: what matters is how
    they compare between pairs, and how they grow as the spin approaches
    an avoided crossing. The pair with the smallest gap is flagged.

    Returns
    -------
    report : list of dict
        One entry per pair, sorted from smallest to largest gap, with
        keys ``pair``, ``gap``, ``amplification`` (a dict keyed by task)
        and ``smallest_gap`` (bool).
    """
    gaps = mode_gaps(modes, chi, m_msun, approx)
    report = []
    for pair, gap in sorted(gaps.items(), key=lambda kv: kv[1]):
        report.append({
            "pair": pair,
            "gap": gap,
            "amplification": {task: gap ** (-p)
                              for task, p in TASK_EXPONENTS.items()},
            "smallest_gap": False,
        })
    if report:
        report[0]["smallest_gap"] = True
    return report
