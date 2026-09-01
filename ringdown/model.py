"""Module defining the probabilistic model (likelihood and prior)
for ringdown data.
"""

__all__ = ["make_model", "get_arviz", "rd_design_matrix"]

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import jax

import numpyro
import numpyro.distributions as dist
from . import qnms
from .indexing import ModeIndexList
from .result import Result
from .utils.swsh import construct_sYlm, calc_YpYc

import arviz as az
import xarray as xr
from arviz_base import dict_to_dataset
import logging

logger = logging.getLogger(__name__)


def smooth_truncation_logprob(a, a_max, delta, clamp_value=20.0):
    """Compute a numerically stable log-potential for amplitude truncation.

    This function implements a smooth transition in log-probability space that
    approaches 0 for amplitudes well below a_max and -∞ for amplitudes well
    above a_max. The transition occurs over a width delta, centered at a_max.

    The function uses a softplus-based implementation of log((1+tanh z)/2) for
    numerical stability, where z = (a_max - a)/delta. The clamp_value parameter
    ensures numerical stability by bounding the intermediate calculations.

    Arguments
    ---------
    a : array_like
        The amplitude values to evaluate.
    a_max : float
        The maximum amplitude at which the transition occurs.
    delta : float
        The width of the smooth transition region.
    clamp_value : float, optional
        The maximum absolute value for intermediate calculations to ensure
        numerical stability. Default is 20.0, which provides a good balance
        between numerical stability and function behavior.

    Returns
    -------
    log_prob : array_like
        The log-probability values, approaching 0 for a << a_max and -∞ for
        a >> a_max, with a smooth transition in between.
    """
    # 1) compute z
    z = (a_max - a) / delta

    # 2) replace NaNs / ±Inf with finite
    z = jnp.nan_to_num(z, nan=-clamp_value, posinf=clamp_value,
                       neginf=-clamp_value)

    # 3) clamp to avoid extreme values
    z = jnp.clip(z, -clamp_value, clamp_value)

    # 4) stable log((1+tanh z)/2) = -softplus(-2 z)
    return -jax.nn.softplus(-2.0 * z)


def rd_design_matrix(
    ts,
    f,
    gamma,
    Fp,
    Fc,
    Ascales,
    aligned=False,
    YpYc=None,
    single_polarization=False,
):
    """Construct the design matrix for a generic ringdown model.

    For each detector, this is a matrix whose rows are the cosine and sine
    basis functions for the damped sinusoids that make up the ringdown model;
    the columns are times.

    There are four quadratures per mode
    :math:`(F_+ \\cos\\omega t, F_+ \\sin\\omega t,
    F_\\times \\cos\\omega t, F_\\times \\sin\\omega t)`,
    so that the design matrix has shape ``(nifo, nt, nquads*nmode)``, i.e.,::

        [
            [
                # [0:nmode] are the plus cosine quadratures
                Fp * exp(-gamma_0*t_0) * cos(omega_0*t_0),
                Fp * exp(-gamma_1*t_0) * cos(omega_1*t_0),
                ...,
                # [nmode:2*nmode] are the plus sine quadratures
                Fp * exp(-gamma_0*t_0) * sin(omega_0*t_0),
                Fp * exp(-gamma_1*t_0) * sin(omega_1*t_0),
                ...,
                # [2*nmode:3*nmode] are the cross cosine quadratures
                Fc * exp(-gamma_0*t_0) * cos(omega_0*t_0),
                Fc * exp(-gamma_1*t_0) * cos(omega_1*t_0),
                ...,
                # [3*nmode:4*nmode] are the cross sine quadratures
                Fc * exp(-gamma_0*t_0) * sin(omega_0*t_0),
                Fc * exp(-gamma_1*t_0) * sin(omega_1*t_0),
                ...
            ],
            [
                Fp * exp(-gamma_0*t_1) * cos(omega_0*t_1),
                Fp * exp(-gamma_1*t_1) * cos(omega_1*t_1),
                ...,
                Fp * exp(-gamma_0*t_1) * sin(omega_0*t_1),
                Fp * exp(-gamma_1*t_1) * sin(omega_1*t_1),
                ...,
                Fc * exp(-gamma_0*t_1) * cos(omega_0*t_1),
                Fc * exp(-gamma_1*t_1) * cos(omega_1*t_1),
                ...,
                Fc * exp(-gamma_0*t_1) * sin(omega_0*t_1),
                Fc * exp(-gamma_1*t_1) * sin(omega_1*t_1),
                ...
            ],
            ...
        ]

    For the aligned model we have that, for each :math:`(\\ell, m)` mode and
    suppressing the exponential decay,

    .. math::
        h_+ = A_{\\ell m} Y_{\\ell m}^+ \\cos(\\omega_{\\ell m} t
        - \\phi_{\\ell m}) \\\\

        h_\\times = A_{\\ell m} Y_{\\ell m}^\\times \\sin(\\omega_{\\ell m} t
        - \\phi_{\\ell m})

    This means that the quadratures are:

    .. math::
        x_+ = A_{\\ell m} Y_{\\ell m}^+ \\cos\\phi_{\\ell m} \\\\ y_+ =
        A_{\\ell m} Y_{\\ell m}^+ \\sin\\phi_{\\ell m} \\\\ x_\\times =
        - A_{\\ell m} Y_{\\ell m}^\\times \\sin\\phi_{\\ell m} \\\\ y_\\times
        = A_{\\ell m} Y_{\\ell m}^\\times \\cos\\phi_{\\ell m}

    We want to combine these four quadratures into two. To do this, note that
    the overall signal at a given detector looks like:

    .. math::
        h = A_{\\ell m} \\left( F_+ Y_{\\ell m}^+ \\cos\\phi_{\\ell m} -
        F_\\times Y_{\\ell m}^\\times \\sin\\phi_{\\ell m} \\right)
        \\cos(\\omega_{\\ell m} t) +
            A_{\\ell m} \\left( F_+ Y_{\\ell m}^+ \\sin\\phi_{\\ell m} +
            F_\\times Y_{\\ell m}^\\times \\cos\\phi_{\\ell m} \\right)
            \\sin(\\omega_{\\ell m} t)

    where :math:`F_+` and :math:`F_\\times` are the plus and cross polarization
    antenna patterns.

    We want to sum the cosine and sine columns to get the right linear
    combination per the equation above. The right linear combination becomes
    apparent if we write the above as in inner product:

    .. math::
        h = (x, y) \\cdot M

    where :math:`x = A_{\\ell m} \\cos \\phi_{\\ell m}` and :math:`y = A_{\\ell
    m} \\sin \\phi_{\\ell m}` while :math:`M` is the matrix

    .. math::
        M = \\begin{pmatrix}
            F_+ Y_{\\ell m}^+ \\cos\\omega t + F_\\times Y_{\\ell m}^\\times
            \\sin\\omega t \\\\
            F_+ Y_{\\ell m}^+ \\sin\\omega t - F_\\times Y_{\\ell m}^\\times
            \\cos\\omega t \\\\
        \\end{pmatrix}

    This function effects that summation to return a design matrix
    corresponding to two quadratures.

    Arguments
    ---------
    ts : array_like
        The times at which to evaluate the design matrix; shape ``(nifo, nt)``.
    f : array_like
        The frequencies of the damped sinusoids; shape ``(nmode,)``.
    gamma : array_like
        The damping rates of the damped sinusoids; shape ``(nmode,)``.
    Fp : array_like
        The plus polarization coefficients; shape ``(nifo,)``.
    Fc : array_like
        The cross polarization coefficients; shape ``(nifo,)``.
    Ascales : array_like
        The amplitude scales of the damped sinusoids; shape ``(nmode,)``.

    Returns
    -------
    design_matrix : array_like
        The design matrix; shape (nifo, nt, nquads*nmode).
    """
    if aligned:
        logger.debug("aligned model is not reviewed for LVK use")
    if single_polarization:
        logger.debug("single_polarization model is not reviewed for LVK use")

    # times should be originally shaped (nifo, nt)
    # take it to (nifo, nt, 1) where the last dimension is the mode
    ts = jnp.atleast_2d(ts)[:, :, jnp.newaxis]

    # get number of detectors, times, and modes
    nifo = ts.shape[0]
    nmode = jnp.shape(f)[0]

    f = jnp.reshape(f, (1, 1, nmode))
    gamma = jnp.reshape(gamma, (1, 1, nmode))
    Fp = jnp.reshape(Fp, (nifo, 1, 1))
    Fc = jnp.reshape(Fc, (nifo, 1, 1))
    Ascales = jnp.reshape(Ascales, (1, 1, nmode))

    # ct and st will have shape (nifo, nt, nmode)
    decay = jnp.exp(-gamma * ts)
    ct = Ascales * decay * jnp.cos(2 * np.pi * f * ts)
    st = Ascales * decay * jnp.sin(2 * np.pi * f * ts)

    if single_polarization:
        dm = jnp.concatenate((Fp * ct, Fp * st), axis=2)
    else:
        dm = jnp.concatenate((Fp * ct, Fp * st, Fc * ct, Fc * st), axis=2)

    if aligned and not single_polarization:
        # NOTE: we could add a polarization dof here via the azimuthal angle
        # argument of YpYc, restoring theta---but this is degenerate with psi
        # (we might still want to add that option, if we don't want to fix psi)
        Yp_mat = jnp.reshape(YpYc[0], (1, 1, nmode))
        Yc_mat = jnp.reshape(YpYc[1], (1, 1, nmode))
        dm = jnp.concatenate(
            [
                # Yp * Fp * cos + Yc * Fc * sin
                Yp_mat * dm[:, :, :nmode] + Yc_mat * dm[:, :, 3 * nmode:],
                # Yp * Fp * sin - Yc * Fc * cos
                Yp_mat * dm[:, :, nmode: 2 * nmode]
                - Yc_mat * dm[:, :, 2 * nmode: 3 * nmode],
            ],
            axis=2,
        )
    elif aligned:
        raise ValueError("aligned model requires single_polarization=False")
    return dm


def chi_factors(chi, coeffs):
    log_1m_chi = jnp.log1p(-chi)
    log_1m_chi_2 = log_1m_chi * log_1m_chi
    log_1m_chi_3 = log_1m_chi_2 * log_1m_chi
    log_1m_chi_4 = log_1m_chi_2 * log_1m_chi_2
    log_sqrt_1m_chi2 = 0.5 * jnp.log1p(-(chi**2))
    log_sqrt_1m_chi2_2 = log_sqrt_1m_chi2 * log_sqrt_1m_chi2
    log_sqrt_1m_chi2_3 = log_sqrt_1m_chi2_2 * log_sqrt_1m_chi2
    log_sqrt_1m_chi2_4 = log_sqrt_1m_chi2_2 * log_sqrt_1m_chi2_2
    log_sqrt_1m_chi2_5 = log_sqrt_1m_chi2_3 * log_sqrt_1m_chi2_2
    log_sqrt_1m_chi2_6 = log_sqrt_1m_chi2_3 * log_sqrt_1m_chi2_3

    v = jnp.stack(
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

    return jnp.dot(coeffs, v)


def Aellip_from_quadratures(Apx, Apy, Acx, Acy):
    # should be slightly cheaper than calling the two functions separately
    term1 = jnp.sqrt(jnp.square(Acy + Apx) + jnp.square(Acx - Apy))
    term2 = jnp.sqrt(jnp.square(Acy - Apx) + jnp.square(Acx + Apy))
    A = 0.5 * (term1 + term2)
    e = 0.5 * (term1 - term2) / A
    return A, e


def phiR_from_quadratures(Apx, Apy, Acx, Acy):
    return jnp.arctan2(-Acx + Apy, Acy + Apx)


def phiL_from_quadratures(Apx, Apy, Acx, Acy):
    return jnp.arctan2(-Acx - Apy, -Acy + Apx)


def get_quad_derived_quantities(
    nmodes,
    design_matrices,
    quads,
    a_scale,
    YpYc,
    store_h_det,
    store_h_det_mode,
    compute_h_det=False,
):
    nifo, ntimes, nquads_nmodes = design_matrices.shape
    nquads = nquads_nmodes // nmodes

    if nquads == 2:
        ax_unit = quads[:nmodes]
        ay_unit = quads[nmodes:]

        a_norm = jnp.sqrt(jnp.square(ax_unit) + jnp.square(ay_unit))
        a = numpyro.deterministic("a", a_scale * a_norm)
        numpyro.deterministic("phi", jnp.arctan2(ay_unit, ax_unit))
        if YpYc is not None:
            numpyro.deterministic("ellip", YpYc[2])
        # theta = 0 for the aligned model
        # ellip = 0 and theta = 0,pi/2 for the single polarization model
    else:
        apx_unit = quads[:nmodes]
        apy_unit = quads[nmodes: 2 * nmodes]
        acx_unit = quads[2 * nmodes: 3 * nmodes]
        acy_unit = quads[3 * nmodes:]

        numpyro.deterministic("apx", apx_unit * a_scale)
        numpyro.deterministic("apy", apy_unit * a_scale)
        numpyro.deterministic("acx", acx_unit * a_scale)
        numpyro.deterministic("acy", acy_unit * a_scale)

        a_norm, e = Aellip_from_quadratures(
            apx_unit, apy_unit, acx_unit, acy_unit
        )
        a = numpyro.deterministic("a", a_scale * a_norm)
        numpyro.deterministic("ellip", e)

        phi_r = numpyro.deterministic(
            "phi_r",
            phiR_from_quadratures(apx_unit, apy_unit, acx_unit, acy_unit),
        )
        phi_l = numpyro.deterministic(
            "phi_l",
            phiL_from_quadratures(apx_unit, apy_unit, acx_unit, acy_unit),
        )

        numpyro.deterministic("theta", -0.5 * (phi_r + phi_l))
        numpyro.deterministic("phi", 0.5 * (phi_r - phi_l))

    if compute_h_det or store_h_det_mode or store_h_det:
        # initialize strain array, note that now we will want times
        # to be the last dimension (unlike in the design matrix)
        h_det_mode = jnp.zeros((nifo, nmodes, ntimes))

        # multiply each quadrature by their respective amplitude
        # (note that this is still of shape (nifo, ntimes, nquads))
        hh = design_matrices * quads[jnp.newaxis, jnp.newaxis, :]

        for i in range(nmodes):
            h_det_mode = h_det_mode.at[:, i, :].set(
                jnp.sum(hh[:, :, i::nmodes], axis=2)
            )
        h_det = jnp.sum(h_det_mode, axis=1)

        if store_h_det_mode:
            numpyro.deterministic("h_det_mode", h_det_mode)
        if store_h_det:
            numpyro.deterministic("h_det", h_det)

        return a, h_det
    else:
        return a, None


def make_model(
    modes: int | list[(int, int, int, int)],
    a_scale_max: float,
    marginalized: bool = True,
    m_min: float | None = None,
    m_max: float | None = None,
    chi_min: float = 0.0,
    chi_max: float = 0.99,
    cosi_min: float | None = None,
    cosi_max: float | None = None,
    cosi: float | None = None,
    df_min: None | float | list[None | float] = None,
    df_max: None | float | list[None | float] = None,
    dg_min: None | float | list[None | float] = None,
    dg_max: None | float | list[None | float] = None,
    f_min: None | float | list[float] = None,
    f_max: None | float | list[float] = None,
    g_min: None | float | list[float] = None,
    g_max: None | float | list[float] = None,
    flat_amplitude_prior: bool = False,
    mode_ordering: None | str = None,
    single_polarization: bool = False,
    prior: bool = False,
    predictive: bool = False,
    store_h_det: bool = False,
    store_h_det_mode: bool = False,
    amplitude_cutoff_fraction: float = 0.1,
):
    """
    Arguments
    ---------
    modes : int or list[tuple]
        If integer, the number of damped sinusoids to use.  If list of tuples,
        each entry should be of the form `(p, s, ell, m, n)`, where `p` is `1`
        for prograde `-1` for retrograde; `s` is the spin weight (`-2` for the
        usual GW modes); `ell` and `m` refer to the usual angular numbers; and
        `n` is the overtone number.

    a_scale_max : float
        The maximum value of the amplitude scale parameter. This is used to
        define the prior on the amplitude scale parameter.

    marginalized : bool
        Whether or not to marginalize over the quadrature amplitudes
        analytically.

    m_min : float
        The minimum mass of the black hole in solar masses.

    m_max : float
        The maximum mass of the black hole in solar masses.

    chi_min : float
        The minimum dimensionless spin of the black hole.

    chi_max : float
        The maximum dimensionless spin of the black hole.

    cosi_min : float or None
        The minimum inclination angle to the angular orbital momentum of the
        black hole.

    cosi_max : float or None
        The maximum inclination angle to the angular orbital momentum of the
        black hole.

    cosi : float or None
        The inclination angle to the angular orbital momentum of the black
        hole. If not `None`, then `cosi_min` and `cosi_max` are ignored and
        the value of `cosi` is fixed.

    df_min : None or float or list[None or float]
        The minimum fractional deviation from the GR frequency.  If a list
        then it should have the same length as `modes`.

    df_max : None or float or list[None or float]
        The maximum fractional deviation from the GR frequency.  If a list,
        then it should have the same length as `modes`.

    dg_min : None or float or list[None or float]
        The minimum fractional deviation from the GR damping rate.  If a list,
        then it should have the same length as `modes`.

    dg_max : None or float or list[None or float]
        The maximum fractional deviation from the GR damping rate.  If a list,
        then it should have the same length as `modes`.

    flat_amplitude_prior : bool
        Whether or not to impose a flat prior on the amplitude scale parameter.
        This is only relevant if `marginalized` is `False`.

    mode_ordering : None or str
        Relevant to the case where `modes` is an integer, and the model
        consists of arbitrary damped sinusoids.  If `None`, then the
        frequencies and damping rates are only constrained by the bounds
        `f_min`, `f_max`, `g_min`, and `g_max`.  If `'f'`, then the
        frequencies are constrained to be in increasing order; if `'g'`,
        then the damping rates are constrained to be in increasing order.

    single_polarization : bool
        if true, assumes a single, linear polarization: either plus or cross,
        with `theta = 0`. This should only be used for testing in simulated
        data for a single detector! (default: False)

    prior : bool
        Whether or not to compute the likelihood.  If `True`, then the
        likelihood is not computed, and the model is used for prior predictive
        sampling.

    predictive : bool
        Whether to generate the quadrature amplitudes when `marginalized=True`.

    store_h_det : bool
        Whether to store the detector-frame waveform in the model.

    store_h_det_mode : bool
        Whether to store the mode-by-mode detector-frame waveform in the model.

    amplitude_cutoff_fraction : float
        Controls the smoothness of the amplitude cutoff when using
        `flat_amplitude_prior`. The cutoff is implemented using a smooth window
        that transitions from 1 to 0 over [a_scale_max * (1 -
        amplitude_cutoff_fraction), a_scale_max]. Larger values make the
        transition region wider, while smaller values make it narrower. Default
        is 0.1, which means the transition occurs over the upper 10% of the
        amplitude range.

    Returns
    -------
    model : function
        A model function that can be used with `numpyro` to sample from the
        posterior distribution of the ringdown parameters.
    """

    n_modes = modes if isinstance(modes, int) else len(modes)

    # check arguments for free damped sinusoid fits
    if mode_ordering is not None:
        if not isinstance(modes, int):
            raise ValueError(
                "mode_ordering is only relevant if modes is an int"
            )
        if mode_ordering not in ["f", "g"]:
            raise ValueError('mode_ordering must be None, "f", or "g"')
        elif mode_ordering == "f":
            if not np.isscalar(f_min) or not np.isscalar(f_max):
                raise ValueError(
                    'mode_ordering is "f" but f_min or f_max are not scalars'
                )
        elif mode_ordering == "g":
            if not np.isscalar(g_min) or not np.isscalar(g_max):
                raise ValueError(
                    'mode_ordering is "g" but g_min or g_max are not scalars'
                )
    elif isinstance(modes, int):
        # for sampling below, we want to make sure all of these are arrays
        if np.isscalar(f_min):
            f_min = [f_min] * n_modes
        if np.isscalar(f_max):
            f_max = [f_max] * n_modes
        if np.isscalar(g_min):
            g_min = [g_min] * n_modes
        if np.isscalar(g_max):
            g_max = [g_max] * n_modes
        # turn lists into jnp arrays
        if not np.isscalar(f_min) or not np.isscalar(f_max):
            f_min = jnp.array(f_min)
            f_max = jnp.array(f_max)
        if not np.isscalar(g_min) or not np.isscalar(g_max):
            g_min = jnp.array(g_min)
            g_max = jnp.array(g_max)
        for a in [f_min, f_max, g_min, g_max]:
            if a.shape[0] != n_modes:
                raise ValueError(
                    "f_min, f_max, g_min, and g_max must have "
                    "the same length as modes"
                )

    # if df_min and df_max are floats, then make it an array
    if df_min is not None and np.isscalar(df_min):
        df_min = [df_min] * n_modes
    if df_max is not None and np.isscalar(df_max):
        df_max = [df_max] * n_modes
    if dg_min is not None and np.isscalar(dg_min):
        dg_min = [dg_min] * n_modes
    if dg_max is not None and np.isscalar(dg_max):
        dg_max = [dg_max] * n_modes

    # if only one of cosi_min or cosi_max is set, set the other to its extremum
    if cosi_min is None and cosi_max is not None:
        cosi_min = -1.0
    if cosi_min is not None and cosi_max is None:
        cosi_max = 1.0
    fixed_cosi = None

    if cosi_min is not None or cosi is not None:
        if isinstance(modes, int):
            raise ValueError("must specify harmonics for aligned model")
        if cosi is not None:
            if cosi < -1 or cosi > 1:
                raise ValueError("cosi must be between -1 and 1")
            if cosi_min is not None or cosi_max is not None:
                logger.warning(
                    "ignoring cosi_min and cosi_max since cosi is fixed"
                )
            fixed_cosi = cosi
        mode_array = np.array(modes)
        swsh = construct_sYlm(-2, mode_array[:, 2], mode_array[:, 3])
    else:
        swsh = None

    def model(
        times,
        strains,
        ls,
        fps,
        fcs,
        predictive: bool = predictive,
        store_h_det: bool = store_h_det,
        store_h_det_mode: bool = store_h_det_mode,
    ):
        """The ringdown model.

        Arguments
        ---------
        times : array_like
            The times at which the ringdown waveform should be evaluated; list
            of 1D arrays for each IFO, or a 2D array with shape (n_det,
            n_times).
        strains : array_like
            The strain data; list of 1D arrays for each IFO, or a 2D array with
            shape (n_det, n_times).
        ls : array_like
            The noise covariance matrices; list of 2D arrays for each IFO, or a
            3D array with shape (n_det, n_times, n_times).
        fps : array_like
            The "plus" polarization coefficients for each IFO; length `n_det`.
        fcs : array_like
            The "cross" polarization coefficients for each IFO; length `n_det`.
        """
        times, strains, ls, fps, fcs = map(
            jnp.array, (times, strains, ls, fps, fcs)
        )

        n_det = times.shape[0]

        # Here is where the particular model choice is made:
        #
        # If modes is an int, then we use a model with f-gamma (i.e. we don't
        # impose any GR constraint on the frequencies / damping rates)
        #
        # If modes is a list, then we use a model with GR-imposed frequencies
        # and damping rates (possibly with deviations).  If you have a
        # beyond-Kerr model, this is where you would put your logic to
        # implement it.
        if isinstance(modes, int):
            if mode_ordering == "f":
                # Sure would be nice if numpyro had `OrderedBoundedVector` or
                # something similar.  But because it doesn't, we have to do it
                # with transformations.  This takes:
                #
                # Unconstrained reals -> ordered reals -> (0,1) ordered reals
                # -> (f_min, f_max) ordered reals
                #
                # Since we want a flat prior on f, but the sampler sees
                # f_latent, we need a Jacobian factor:
                #
                # log_jac = log(d(f)/d(f_latent))
                #
                # which, happily, is provided by the composed transformation
                f_latent = numpyro.sample(
                    "f_latent",
                    dist.ImproperUniform(
                        dist.constraints.real, (), (n_modes,)),
                )
                f_transform = dist.transforms.ComposeTransform(
                    [
                        dist.transforms.OrderedTransform(),
                        dist.transforms.SigmoidTransform(),
                        dist.transforms.AffineTransform(f_min, f_max - f_min),
                    ]
                )
                f = numpyro.deterministic("f", f_transform(f_latent))
                numpyro.factor(
                    "f_transform", f_transform.log_abs_det_jacobian(
                        f_latent, f)
                )

                g = numpyro.sample(
                    "g", dist.Uniform(g_min, g_max), sample_shape=(modes,)
                )
            elif mode_ordering == "g":
                f = numpyro.sample(
                    "f", dist.Uniform(f_min, f_max), sample_shape=(modes,)
                )

                g_latent = numpyro.sample(
                    "g_latent",
                    dist.ImproperUniform(
                        dist.constraints.real, (), (n_modes,)),
                )
                g_transform = dist.transforms.ComposeTransform(
                    [
                        dist.transforms.OrderedTransform(),
                        dist.transforms.SigmoidTransform(),
                        dist.transforms.AffineTransform(g_min, g_max - g_min),
                    ]
                )
                g = numpyro.deterministic("g", g_transform(g_latent))
                numpyro.factor(
                    "g_transform", g_transform.log_abs_det_jacobian(
                        g_latent, g)
                )
            else:
                f = numpyro.sample("f", dist.Uniform(f_min, f_max))
                g = numpyro.sample("g", dist.Uniform(g_min, g_max))
        elif isinstance(modes, list):
            fcoeffs = []
            gcoeffs = []
            for mode in modes:
                c = qnms.KerrMode(mode).coefficients
                fcoeffs.append(c[0])
                gcoeffs.append(c[1])
            fcoeffs = jnp.array(fcoeffs)
            gcoeffs = jnp.array(gcoeffs)

            m = numpyro.sample("m", dist.Uniform(m_min, m_max))
            chi = numpyro.sample("chi", dist.Uniform(chi_min, chi_max))

            f0 = 1 / (m * qnms.T_MSUN)
            f_gr = f0 * chi_factors(chi, fcoeffs)
            g_gr = f0 * chi_factors(chi, gcoeffs)

            if df_min is None or df_max is None:
                f = numpyro.deterministic("f", f_gr)
            else:
                df_unit = numpyro.sample(
                    "df_unit", dist.Uniform(0, 1), sample_shape=(n_modes,)
                )
                # Don't want to shadow df_min and df_max
                df_low = jnp.array([0.0 if x is None else x for x in df_min])
                df_high = jnp.array([0.0 if x is None else x for x in df_max])

                df = numpyro.deterministic(
                    "df", df_unit * (df_high - df_low) + df_low
                )
                f = numpyro.deterministic("f", f_gr * jnp.exp(df))

            if dg_min is None or dg_max is None:
                g = numpyro.deterministic("g", g_gr)
            else:
                dg_unit = numpyro.sample(
                    "dg_unit", dist.Uniform(0, 1), sample_shape=(n_modes,)
                )
                dg_low = jnp.array([0.0 if x is None else x for x in dg_min])
                dg_high = jnp.array([0.0 if x is None else x for x in dg_max])

                dg = numpyro.deterministic(
                    "dg", dg_unit * (dg_high - dg_low) + dg_low
                )
                g = numpyro.deterministic("g", g_gr * jnp.exp(dg))
        # At this point the frequencies `f` and damping rates `g` of the
        # various modes should be established, and we can proceed with the
        # rest of the model.

        if swsh:
            if fixed_cosi is None:
                cosi = numpyro.sample("cosi", dist.Uniform(cosi_min, cosi_max))
            else:
                cosi = fixed_cosi
            YpYc = calc_YpYc(cosi, swsh)
        else:
            YpYc = None

        if marginalized:
            # NOTE: notation in the following block follows
            # https://arxiv.org/abs/2005.14199
            # for ease of reference: we use the same variable names
            # and matrices are capitalized

            a_scale = numpyro.sample(
                "a_scale",
                dist.Uniform(0, a_scale_max),
                sample_shape=(n_modes,)
            )
            # get design matrices which will have shape
            # (n_det, ntime, nquads*nmode)
            dms = rd_design_matrix(
                times,
                f,
                g,
                fps,
                fcs,
                a_scale,
                aligned=swsh,
                YpYc=YpYc,
                single_polarization=single_polarization,
            )

            n_quad_n_modes = dms.shape[2]

            # Marginalize analytically over the linear quadrature
            # amplitudes. Their prior is a zero-mean unit Gaussian,
            # N(0, Lambda) with Lambda = I (the scale of the quadratures has
            # been absorbed into the design matrix, otherwise we would write
            # Lambda = a_scale**2 I).
            #
            # Marginalizing over all detectors at once has a closed form.
            # With the whitened design matrix W_i = L_i^-1 M_i and whitened
            # data z_i = L_i^-1 y_i for detector i (so that C_i^-1 never has
            # to be formed, since C_i = L_i L_i^T),
            #
            #   A_inv = Lambda^-1 + sum_i W_i^T W_i = I + sum_i W_i^T W_i
            #   v     = sum_i W_i^T z_i
            #   Q     = sum_i |z_i|^2
            #   A_inv = R R^T  (Cholesky),   u = R^-1 v
            #
            # and the total marginal log likelihood is
            #
            #   logl = -Q/2 + |u|^2/2
            #          - sum_i sum_t log[L_i]_tt - sum_j log R_jj
            #
            # The two log-determinant terms come from the Gaussian
            # refactorization |Lambda| |C| = |A| |B|, i.e.
            #   log|B| = log|C| + log|Lambda| - log|A|
            # with log|Lambda| = 0 and log|A| = -log|A_inv| = -2 sum_j log
            # R_jj (note that it is the *logs* that flip sign, not the
            # determinants); as usual, the parameter-independent
            # -0.5*n*log(2pi) is dropped.
            #
            # This is exactly the sum of the per-detector conditional log
            # likelihoods that a sequential scheme -- marginalizing one
            # detector at a time and using its posterior as the next
            # detector's prior -- would accumulate; the recursion telescopes
            # onto the expression above. See docs/marginalized_likelihood.md
            # for the derivation and the proof.
            #
            # Q and sum_i sum_t log[L_i]_tt carry no dependence on the
            # sampled parameters and could be dropped, but they are retained
            # so that the reported log likelihood keeps its usual numerical
            # value rather than being offset by a constant.

            # accumulators over detectors; A_inv starts from the prior
            # precision Lambda^-1 = I
            A_inv = jnp.eye(n_quad_n_modes)
            v = jnp.zeros(n_quad_n_modes)
            Q = 0.0
            logdetL = 0.0

            if not prior:
                # unrolled at trace time: n_det is a small Python int
                for i in range(n_det):
                    # the design matrix M takes us from the space of
                    # quadratures to the space of the data for this detector
                    # (M is ntime x nquads*nmode), while L whitens its noise,
                    # taking it from N(0, C) to N(0, I) (L is ntime x ntime)
                    M = dms[i, :, :]
                    L = ls[i, :, :]
                    y = strains[i, :]

                    # whiten the design matrix and the data with one forward
                    # substitution each
                    W = jsp.linalg.solve_triangular(L, M, lower=True)
                    z = jsp.linalg.solve_triangular(L, y, lower=True)

                    A_inv = A_inv + jnp.dot(W.T, W)
                    v = v + jnp.dot(W.T, z)
                    Q = Q + jnp.dot(z, z)
                    logdetL = logdetL + jnp.sum(jnp.log(jnp.diag(L)))

            # A_inv_chol is R above. It is computed unconditionally so that
            # the predictive draw below has a single code path: when
            # prior=True the accumulators keep their initial values, so
            # R = chol(I) = I and u = 0, and the draw reduces to the unit
            # Gaussian prior.
            A_inv_chol = jsp.linalg.cholesky(A_inv, lower=True)
            u = jsp.linalg.solve_triangular(A_inv_chol, v, lower=True)

            if not prior:
                numpyro.factor(
                    "logl_total",
                    -0.5 * Q
                    + 0.5 * jnp.dot(u, u)
                    - logdetL
                    - jnp.sum(jnp.log(jnp.diag(A_inv_chol))),
                )

            if predictive:
                # Generate the actual quadrature amplitudes by taking a draw
                # from the marginal posterior N(A v, A), with A = A_inv^-1
                # the covariance and A v its mean.

                # A_inv = A_inv_chol A_inv_chol.T = R R^T, so that
                # A = R^{-T} R^{-1}; to achieve the desired covariance we can
                # left-multiply iid N(0,1) variables xi by R^{-T}, since
                # < (R^{-T} xi)(R^{-T} xi)^T > = R^{-T} I R^{-1} = A.
                # The mean is A v = R^{-T} R^{-1} v = R^{-T} u with
                # u = R^{-1} v as computed above, so mean and fluctuation
                # can be obtained together with a single back substitution:
                #   quads = R^{-T} (u + xi)
                # When prior=True this is R = I, u = 0 and hence quads = xi,
                # i.e. a draw from the unit Gaussian prior, as intended.
                if swsh or single_polarization:
                    ax_unit = numpyro.sample(
                        "ax_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                    )
                    ay_unit = numpyro.sample(
                        "ay_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                    )
                    unit_quads = jnp.concatenate((ax_unit, ay_unit))
                else:
                    apx_unit = numpyro.sample(
                        "apx_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                    )
                    apy_unit = numpyro.sample(
                        "apy_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                    )
                    acx_unit = numpyro.sample(
                        "acx_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                    )
                    acy_unit = numpyro.sample(
                        "acy_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                    )
                    unit_quads = jnp.concatenate(
                        (apx_unit, apy_unit, acx_unit, acy_unit)
                    )

                quads = jsp.linalg.solve(A_inv_chol.T, u + unit_quads)
                get_quad_derived_quantities(
                    n_modes,
                    dms,
                    quads,
                    a_scale,
                    YpYc,
                    store_h_det,
                    store_h_det_mode,
                )
        else:
            a_scales = a_scale_max * jnp.ones(n_modes)
            dms = rd_design_matrix(
                times,
                f,
                g,
                fps,
                fcs,
                a_scales,
                aligned=swsh,
                YpYc=YpYc,
                single_polarization=single_polarization,
            )
            if swsh or single_polarization:
                ax_unit = numpyro.sample(
                    "ax_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                )
                ay_unit = numpyro.sample(
                    "ay_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                )
                quads = jnp.concatenate((ax_unit, ay_unit))
            else:
                apx_unit = numpyro.sample(
                    "apx_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                )
                apy_unit = numpyro.sample(
                    "apy_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                )
                acx_unit = numpyro.sample(
                    "acx_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                )
                acy_unit = numpyro.sample(
                    "acy_unit", dist.Normal(0, 1), sample_shape=(n_modes,)
                )
                quads = jnp.concatenate(
                    (apx_unit, apy_unit, acx_unit, acy_unit)
                )

            a, h_det = get_quad_derived_quantities(
                n_modes,
                dms,
                quads,
                a_scale_max,
                YpYc,
                store_h_det,
                store_h_det_mode,
                compute_h_det=(not prior),
            )

            if flat_amplitude_prior:
                # We need a Jacobian that is A^-3 for the generic model
                # (4 quadratures) and A^-1 for the aligned model
                # (2 quadratures)
                n_quad_n_modes = dms.shape[2]
                n_quad = n_quad_n_modes / n_modes

                numpyro.factor(
                    "flat_a_prior",
                    (1 - n_quad) * jnp.sum(jnp.log(a))
                    + 0.5 * jnp.sum(jnp.square(quads))
                )

                if prior:
                    # Soft log-potential:
                    # ~0 for a << a_scale_max, → -∞ for a ≫ a_scale_max
                    cutoff_lp = smooth_truncation_logprob(
                        a,
                        a_scale_max,
                        a_scale_max * amplitude_cutoff_fraction,
                    )
                    numpyro.factor(
                        "cutoff_lp",
                        jnp.sum(cutoff_lp)
                    )

            if not prior:
                for i, strain in enumerate(strains):
                    numpyro.sample(
                        f"logl_{i}",
                        dist.MultivariateNormal(
                            h_det[i, :], scale_tril=ls[i, :, :]
                        ),
                        obs=strain,
                    )

    return model


MODEL_VARIABLES_BY_MODE = [
    "a_scale",
    "a",
    "acx",
    "acy",
    "apx",
    "apy",
    "acx_unit",
    "acy_unit",
    "apx_unit",
    "apy_unit",
    "ax_unit",
    "ay_unit",
    "ellip",
    "f",
    "g",
    "omega",
    "phi",
    "phi_l",
    "phi_r",
    "quality",
    "tau",
    "theta",
    "df",
    "dg",
    "df_unit",
    "dg_unit",
    "g_latent",
    "f_latent"
]
MODEL_DIMENSIONS = {k: ["mode"] for k in MODEL_VARIABLES_BY_MODE}
MODEL_DIMENSIONS["h_det"] = ["ifo", "time_index"]
MODEL_DIMENSIONS["h_det_mode"] = ["ifo", "mode", "time_index"]


def get_arviz(
    sampler,
    modes: list | None = None,
    ifos: list | None = None,
    injections: list | None = None,
    epoch: list | None = None,
    scale: list | None = None,
    attrs: dict | None = None,
    store_data: bool = True,
):
    """Convert a numpyro sampler to an arviz dataset.

    Arguments
    ---------
    sampler : numpyro.MCMC
        The sampler to convert after running.
    modes : None or array_like
        Coordinates of modes to include in the dataset; if `None`, then all
        modes are included and indexed by integers.
    ifos : None or array_like
        The ifos to include in the dataset.  If `None`, then all ifos are
        included.
    injections : None or array_like
        The injections to include in the dataset.  If `None`, then no
        injections are included.
    epoch : None or array_like
        The epoch of each ifo. If `None`, then all epochs are set to zero.
    scale : None or float
        The scale of the strain. If `None`, then the scale is set to one.
    attrs : None or dict
        Attributes to include in the arviz dataset.
    record_data : bool
        Whether to record the observed data and auxiliary quantities in the
        arviz dataset.

    Returns
    -------
    dataset : Result
        The result dataset, a :class:`xarray.DataTree` subclass.
    """
    samples = sampler.get_samples()
    params_in_model = samples.keys()
    # get dimensions
    dims = {k: v for k, v in MODEL_DIMENSIONS.items() if k in params_in_model}
    for x in params_in_model:
        if x not in MODEL_DIMENSIONS and len(samples[x].shape) > 1:
            logger.warning(f"{x} not in model dimensions; please report issue")

    # get coordinates
    # assume that all fits will have an 'f' parameter
    n_mode = samples["f"].shape[1]
    modes = ModeIndexList(modes or n_mode).get_coordinates()
    if len(modes) != n_mode:
        raise ValueError(f"expected {n_mode} modes, got {len(modes)}")
    # get ifo from shape of Fc, assuming it's last argument provided to model
    n_ifo = len(sampler._args[-1])
    if ifos is None:
        ifos = np.arange(n_ifo, dtype=int)
    elif len(ifos) != n_ifo:
        raise ValueError(f"expected {n_ifo} ifos, got {len(ifos)}")
    # get epochs
    if epoch is None:
        epoch = np.zeros(n_ifo)
    elif len(epoch) != n_ifo:
        raise ValueError(f"expected {n_ifo} epochs, got {len(epoch)}")
    # get times from model arguments
    n_analyze = len(sampler._args[0][0])
    time_index = np.arange(n_analyze, dtype=int)
    coords = {
        "ifo": ifos,
        "mode": modes,
        "time_index": time_index,
        "time_index_1": time_index,
    }
    # always include strain scale in constant_data
    in_data = {"scale": scale or 1.0}

    if store_data:
        # get constant_data
        in_dims = {
            "time": ["ifo", "time_index"],
            "strain": ["ifo", "time_index"],
            "cholesky_factor": ["ifo", "time_index", "time_index_1"],
            "fp": ["ifo"],
            "fc": ["ifo"],
            "epoch": ["ifo"],
        }
        in_data.update({k: np.array(v)
                       for k, v in zip(in_dims.keys(), sampler._args)})
        in_data["epoch"] = np.array(epoch)
        in_data["scale"] = scale or 1.0
        # get injections, if provided
        if injections is not None:
            in_data["injection"] = np.array(injections)
            in_dims["injection"] = ["ifo", "time_index"]
        dims.update(in_dims)
        obs_data = {"strain": in_data.pop("strain")}
    else:
        # keep only 'scale' in constant_data when store_data is False
        pass

    # NOTE: as of arviz 1.x, from_numpyro no longer computes the pointwise
    # log likelihood by default, so request it explicitly
    result = az.from_numpyro(
        sampler,
        dims=dims,
        coords=coords,
        constant_data=in_data,
        log_likelihood=True,
    )
    # from_numpyro converts the posterior samples to numpy but leaves JAX
    # arrays in other groups (sample_stats, log_likelihood, observed sites);
    # convert everything to numpy to avoid device-pinned data and spurious
    # x64-truncation warnings in downstream xarray operations
    result = result.map_over_datasets(xr.Dataset.as_numpy)
    result.attrs.update(attrs or {})

    if store_data:
        # add observed data
        if hasattr(result, "observed_data"):
            logger.info("added strain to observed data")
            # make sure we have the right coordinates
            result.observed_data.coords.update(coords)
            result.observed_data["strain"] = xr.DataArray(
                obs_data["strain"], dims=in_dims["strain"]
            )
        else:
            logger.info("creating observed data in arviz dataset")
            # We assume that observed_data isn't created yet.
            result["observed_data"] = dict_to_dataset(
                obs_data,
                coords=coords,
                dims={"strain": in_dims["strain"]},
            )
    return Result(result)


def get_neff_from_numpyro(sampler):
    """Get the effective sample size from a numpyro sampler.

    Arguments
    ---------
    sampler : numpyro.MCMC
        The sampler to compute the effective sample size from.

    Returns
    -------
    neff : dict
        The effective sample size for each parameter.
    """
    import io
    import contextlib
    import pandas as pd

    # Create a string buffer to capture the output
    buffer = io.StringIO()
    # Redirect standard output to the buffer
    with contextlib.redirect_stdout(buffer):
        sampler.print_summary()
    # Get the content from the buffer
    output = buffer.getvalue()

    neff = pd.read_csv(io.StringIO(output), sep=r"\s+")["n_eff"].drop("Number")

    return neff
