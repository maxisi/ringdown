"""Equivalence tests for the marginalized-likelihood block of
:mod:`ringdown.model`.

The helpers :func:`_sequential_reference` and :func:`_make_reference_model`
are a *frozen* transcription of the sequential, one-detector-at-a-time
marginalization that the model used before the closed-form rewrite: each
detector's marginal posterior on the quadratures serves as the prior for the
next detector.  The closed form now in ``ringdown.model`` is an exact
algebraic identity for the same quantity (see
``docs/marginalized_likelihood.md``), so the two must agree in the potential
energy, in every gradient component, and -- given the same standard-normal
draw -- pointwise in the predictive quadratures.

The reference lives here rather than in the package on purpose: it is only
ever needed to pin the production code, and keeping a second likelihood path
in ``ringdown/`` would have to be maintained and compiled for no user-facing
benefit.

Tolerances are 1e-11 *relative*, not machine epsilon.  The test covariances
are deliberately realistically conditioned -- an aLIGO-like PSD with its
seismic wall left in gives ``cond(C)`` of a few times 1e9 over a
ringdown-length segment -- and at that conditioning two algebraically
identical forms can differ by well over 1e-15 from shared roundoff amplified
by ``cond(C)``.  The measured deviation across the configurations here is
<= 4e-13 in the gradient and <= 3e-14 in the potential energy, so 1e-11
leaves margin without being vacuous.  Two deliberately worst-case
combinations (nearly degenerate modes on top of an ill-conditioned
covariance; a single detector, whose plus and cross design-matrix columns
are exactly proportional) reach ~1e-8 and are asserted separately, against
an ``eps * cond(C)`` bound, rather than being allowed to loosen this one.
"""

import numpy as np
import pytest
import scipy.linalg

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import potential_energy

from ringdown.model import (
    get_quad_derived_quantities,
    make_model,
    rd_design_matrix,
)

# sampling rate and prior bounds used throughout
DT = 1.0 / 2048.0
F_MIN, F_MAX = 100.0, 400.0
G_MIN, G_MAX = 10.0, 200.0
A_SCALE_MAX = 1.0

# relative tolerance for "these are the same number" comparisons; see the
# module docstring for why this is not 1e-15
RTOL = 1e-11

# (n_det, n_analyze, n_modes)
CONFIGS = [(1, 205, 1), (2, 205, 2), (3, 205, 3), (2, 410, 2)]


@pytest.fixture(scope="module", autouse=True)
def _enable_x64():
    """Run this module in double precision, as production fits do."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


# -----------------------------------------------------------------------
# synthetic data with a realistically conditioned noise covariance
# -----------------------------------------------------------------------

def _aligo_psd(f):
    """Analytic aLIGO-like power spectral density."""
    x = f / 215.0
    return 1e-49 * (
        x ** -4.14
        - 5 * x ** -2
        + 111 * (1 - x ** 2 + 0.5 * x ** 4) / (1 + 0.5 * x ** 2)
    )


def _aligo_acf(n, dt=DT, f_low=0.2, n_fft=4096):
    """Unit-normalized autocovariance function for an aLIGO-like PSD.

    The steep low-frequency wall is only floored below ``f_low`` rather than
    removed, so that the Toeplitz covariance built from the result has
    ``cond(C)`` of order 1e10, as real strain covariances do.
    """
    df = 1.0 / (n_fft * dt)
    f = np.arange(n_fft // 2 + 1) * df
    psd = np.where(
        f >= f_low, _aligo_psd(np.maximum(f, f_low)), _aligo_psd(f_low)
    )
    rho = np.fft.irfft(psd)[:n_fft] / dt
    return rho[:n] / rho[0]


def _aligo_covariance(n_t, det=0):
    return scipy.linalg.toeplitz(_aligo_acf(n_t, f_low=0.2 + 0.05 * det))


def _make_data(n_det, n_t, n_modes, seed=42, white=False):
    """Times, strains, Cholesky factors and antenna patterns.

    The strain is (by default coloured) noise plus an injected ringdown, so
    that the likelihood is informative rather than a pure-noise plateau.
    With ``white=True`` the noise covariance is the identity, which isolates
    the algebra from the conditioning of ``C``.
    """
    rng = np.random.default_rng(seed)
    times = np.tile(np.arange(n_t) * DT, (n_det, 1))

    if white:
        ls = np.array([np.eye(n_t) for _ in range(n_det)])
    else:
        ls = np.array(
            [
                np.linalg.cholesky(_aligo_covariance(n_t, i))
                for i in range(n_det)
            ]
        )

    fps = np.array([1.0, 0.7, -0.4][:n_det] + [0.5] * max(0, n_det - 3))
    fcs = np.array([0.3, -0.9, 0.6][:n_det] + [-0.2] * max(0, n_det - 3))

    # injected signal at O(1) amplitude; scale invariance of the model means
    # unit-scale data with a_scale_max = 1 stands in for the production
    # 1e-21 / 1e-21 pairing
    f_true = np.linspace(150.0, 300.0, n_modes)
    g_true = np.linspace(30.0, 80.0, n_modes)
    a_true = np.full(n_modes, 0.5)
    dms = np.asarray(
        rd_design_matrix(
            jnp.array(times),
            jnp.array(f_true),
            jnp.array(g_true),
            jnp.array(fps),
            jnp.array(fcs),
            jnp.array(a_true),
        )
    )
    quads_true = rng.normal(size=dms.shape[2])
    signal = dms @ quads_true

    noise = np.einsum("ijk,ik->ij", ls, rng.normal(size=(n_det, n_t)))
    strains = signal + noise
    return times, strains, ls, fps, fcs


# -----------------------------------------------------------------------
# frozen reference: the sequential per-detector marginalization
# -----------------------------------------------------------------------

def _sequential_reference(dms, ls, strains, n_det, prior):
    """Sequential per-detector marginalization (pre-rewrite behaviour).

    Emits one ``logl_{i}`` factor per detector and returns the running
    posterior mean and precision Cholesky, which the predictive draw
    consumes.
    """
    n_quad_n_modes = dms.shape[2]

    mu = jnp.zeros(n_quad_n_modes)
    Lambda_inv = jnp.eye(n_quad_n_modes)
    Lambda_inv_chol = jnp.eye(n_quad_n_modes)

    if not prior:
        for i in range(n_det):
            M = dms[i, :, :]
            L = ls[i, :, :]
            y = strains[i, :]

            A_inv = Lambda_inv + jnp.dot(
                M.T, jsp.linalg.cho_solve((L, True), M)
            )
            A_inv_chol = jsp.linalg.cholesky(A_inv, lower=True)

            a = jsp.linalg.cho_solve(
                (A_inv_chol, True),
                jnp.dot(Lambda_inv, mu)
                + jnp.dot(M.T, jsp.linalg.cho_solve((L, True), y)),
            )

            b = jnp.dot(M, mu)
            r = y - b
            Cinv_r = jsp.linalg.cho_solve((L, True), r)

            M_A_Mt_Cinv_r = jnp.dot(
                M,
                jsp.linalg.cho_solve(
                    (A_inv_chol, True), jnp.dot(M.T, Cinv_r)
                ),
            )
            Cinv_M_A_Mt_Cinv_r = jsp.linalg.cho_solve(
                (L, True), M_A_Mt_Cinv_r
            )

            log_sqrt_det_B = (
                jnp.sum(jnp.log(jnp.diag(L)))
                - jnp.sum(jnp.log(jnp.diag(Lambda_inv_chol)))
                + jnp.sum(jnp.log(jnp.diag(A_inv_chol)))
            )

            logl = (
                -0.5 * jnp.dot(r, Cinv_r - Cinv_M_A_Mt_Cinv_r)
                - log_sqrt_det_B
            )

            numpyro.factor(f"logl_{i}", logl)

            mu = a
            Lambda_inv = A_inv
            Lambda_inv_chol = A_inv_chol

    return mu, Lambda_inv_chol


def _make_reference_model(
    n_modes,
    a_scale_max=A_SCALE_MAX,
    prior=False,
    predictive=False,
    store_h_det=False,
    store_h_det_mode=False,
):
    """Generic-damped-sinusoid model using the sequential reference.

    Mirrors the sample sites, their order and their priors in
    :func:`ringdown.model.make_model` for ``modes=n_modes`` (no mode
    ordering, no spin-weighted harmonics, both polarizations), so that
    potential energies are directly comparable at the same unconstrained
    point.
    """
    f_min = jnp.array([F_MIN] * n_modes)
    f_max = jnp.array([F_MAX] * n_modes)
    g_min = jnp.array([G_MIN] * n_modes)
    g_max = jnp.array([G_MAX] * n_modes)

    def model(
        times,
        strains,
        ls,
        fps,
        fcs,
        predictive=predictive,
        store_h_det=store_h_det,
        store_h_det_mode=store_h_det_mode,
    ):
        times, strains, ls, fps, fcs = map(
            jnp.array, (times, strains, ls, fps, fcs)
        )
        n_det = times.shape[0]

        f = numpyro.sample("f", dist.Uniform(f_min, f_max))
        g = numpyro.sample("g", dist.Uniform(g_min, g_max))

        a_scale = numpyro.sample(
            "a_scale", dist.Uniform(0, a_scale_max), sample_shape=(n_modes,)
        )
        dms = rd_design_matrix(times, f, g, fps, fcs, a_scale)

        mu, Lambda_inv_chol = _sequential_reference(
            dms, ls, strains, n_det, prior
        )

        if predictive:
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
            quads = mu + jsp.linalg.solve(Lambda_inv_chol.T, unit_quads)
            get_quad_derived_quantities(
                n_modes,
                dms,
                quads,
                a_scale,
                None,
                store_h_det,
                store_h_det_mode,
            )

    return model


def _make_production_model(n_modes, **kws):
    return make_model(
        n_modes,
        A_SCALE_MAX,
        marginalized=True,
        f_min=F_MIN,
        f_max=F_MAX,
        g_min=G_MIN,
        g_max=G_MAX,
        **kws,
    )


# -----------------------------------------------------------------------
# (a) potential energy and gradients
# -----------------------------------------------------------------------

def _compare_at_points(n_det, n_t, n_modes, points, white=False):
    """Max relative deviation in potential energy and gradient."""
    args = _make_data(n_det, n_t, n_modes, white=white)
    new = _make_production_model(n_modes)
    ref = _make_reference_model(n_modes)

    def pe_new(p):
        return potential_energy(new, args, {}, p)

    def pe_ref(p):
        return potential_energy(ref, args, {}, p)

    grad_new = jax.grad(pe_new)
    grad_ref = jax.grad(pe_ref)

    worst_pe = 0.0
    worst_grad = 0.0
    for p in points:
        v_new = float(pe_new(p))
        v_ref = float(pe_ref(p))
        assert np.isfinite(v_ref)
        worst_pe = max(worst_pe, abs(v_new - v_ref) / abs(v_ref))

        g_new = grad_new(p)
        g_ref = grad_ref(p)
        assert set(g_new) == set(g_ref)
        flat_new = np.concatenate([np.ravel(g_new[k]) for k in sorted(g_new)])
        flat_ref = np.concatenate([np.ravel(g_ref[k]) for k in sorted(g_ref)])
        assert np.all(np.isfinite(flat_ref))
        scale = np.linalg.norm(flat_ref)
        worst_grad = max(
            worst_grad, np.max(np.abs(flat_new - flat_ref)) / scale
        )

    return worst_pe, worst_grad


def _random_points(n_modes, n_points, seed):
    rng = np.random.default_rng(seed)
    return [
        {
            "f": jnp.array(rng.normal(size=n_modes)),
            "g": jnp.array(rng.normal(size=n_modes)),
            "a_scale": jnp.array(rng.normal(size=n_modes)),
        }
        for _ in range(n_points)
    ]


@pytest.mark.parametrize("n_det,n_t,n_modes", CONFIGS)
def test_potential_energy_and_gradient_match_reference(n_det, n_t, n_modes):
    """The closed form reproduces the sequential recursion exactly."""
    points = _random_points(n_modes, 8, seed=1234 + n_det)
    worst_pe, worst_grad = _compare_at_points(n_det, n_t, n_modes, points)
    assert worst_pe < RTOL, f"potential energy differs by {worst_pe:.3g}"
    assert worst_grad < RTOL, f"gradient differs by {worst_grad:.3g}"


def _degenerate_points():
    """Unconstrained points at which the two modes nearly coincide."""
    return [
        {
            "f": jnp.array([0.3, 0.3 + offset]),
            "g": jnp.array([-0.2, -0.2 + offset]),
            "a_scale": jnp.array([0.4, 0.4 + offset]),
        }
        for offset in (1e-3, 1e-5, 1e-7, 1e-9)
    ]


def test_near_degenerate_modes_match_reference():
    """Nearly parallel design-matrix columns, benign noise covariance.

    White noise isolates the effect of the degeneracy itself: with
    ``cond(C) = 1`` the two forms must still agree essentially to machine
    precision even though the quadrature posterior is nearly singular.
    """
    worst_pe, worst_grad = _compare_at_points(
        2, 205, 2, _degenerate_points(), white=True
    )
    assert worst_pe < RTOL, f"potential energy differs by {worst_pe:.3g}"
    assert worst_grad < RTOL, f"gradient differs by {worst_grad:.3g}"


def test_near_degenerate_modes_ill_conditioned():
    """Degenerate modes *and* a realistic noise covariance.

    This is the worst case for both formulations at once: the quadrature
    Gram matrix is nearly singular and the whitening is ill conditioned, so
    the two algebraically identical expressions may disagree at the level
    ``eps * cond(C)`` from shared roundoff, whichever way round they are
    evaluated.  Both stay comfortably inside that bound -- the measured
    deviation is ~2e-8, against a bound of ~1e-5 -- but it is well above
    the 1e-11 that holds everywhere else, so it is asserted separately
    rather than being allowed to loosen the main tolerance.
    """
    cond = np.linalg.cond(_aligo_covariance(205))
    bound = 10 * np.finfo(np.float64).eps * cond
    worst_pe, worst_grad = _compare_at_points(
        2, 205, 2, _degenerate_points()
    )
    assert worst_pe < bound, f"potential energy differs by {worst_pe:.3g}"
    assert worst_grad < bound, f"gradient differs by {worst_grad:.3g}"


# -----------------------------------------------------------------------
# (b) predictive draw
# -----------------------------------------------------------------------

def _fixed_values(n_modes, seed=8):
    rng = np.random.default_rng(seed)
    return {
        "f": jnp.array(np.linspace(160.0, 310.0, n_modes)),
        "g": jnp.array(np.linspace(35.0, 75.0, n_modes)),
        "a_scale": jnp.array(np.linspace(0.3, 0.6, n_modes)),
        "apx_unit": jnp.array(rng.normal(size=n_modes)),
        "apy_unit": jnp.array(rng.normal(size=n_modes)),
        "acx_unit": jnp.array(rng.normal(size=n_modes)),
        "acy_unit": jnp.array(rng.normal(size=n_modes)),
    }


def _traced(model, args, values):
    seeded = handlers.seed(handlers.substitute(model, values), rng_seed=0)
    return handlers.trace(seeded).get_trace(*args)


def _deterministic_values(trace):
    return {
        k: np.asarray(v["value"])
        for k, v in trace.items()
        if v["type"] == "deterministic"
    }


@pytest.mark.parametrize(
    "n_det,n_t,n_modes,white",
    [
        # a single detector sees both polarizations through proportional
        # design-matrix columns, so its quadrature posterior is exactly
        # degenerate; that case is checked against a benign covariance,
        # since with an ill-conditioned one the shared roundoff in the
        # posterior *mean* along the flat direction reaches ~1e-8
        (1, 205, 1, True),
        (2, 205, 2, True),
        (2, 205, 2, False),
        (3, 205, 2, False),
    ],
)
def test_predictive_draw_matches_reference(n_det, n_t, n_modes, white):
    """Same xi in, same quadratures and derived quantities out."""
    args = _make_data(n_det, n_t, n_modes, white=white)
    values = _fixed_values(n_modes)
    kws = dict(predictive=True, store_h_det=True, store_h_det_mode=True)

    new = _traced(_make_production_model(n_modes, **kws), args, values)
    ref = _traced(_make_reference_model(n_modes, **kws), args, values)

    d_new = _deterministic_values(new)
    d_ref = _deterministic_values(ref)
    assert set(d_new) == set(d_ref)
    expected = {
        "a", "ellip", "phi", "phi_r", "phi_l", "theta",
        "apx", "apy", "acx", "acy", "h_det", "h_det_mode",
    }
    assert expected <= set(d_new)

    for key in sorted(d_new):
        scale = max(np.max(np.abs(d_ref[key])), 1e-300)
        dev = np.max(np.abs(d_new[key] - d_ref[key])) / scale
        assert dev < RTOL, f"{key} differs by {dev:.3g}"


def test_single_logl_site_replaces_per_detector_sites():
    """One merged factor site, still matching the ``logl_`` prefix."""
    n_det, n_t, n_modes = 3, 205, 2
    args = _make_data(n_det, n_t, n_modes)
    values = _fixed_values(n_modes)

    new = _traced(_make_production_model(n_modes), args, values)
    assert [k for k in new if k.startswith("logl")] == ["logl_total"]
    assert new["logl_total"]["is_observed"]

    ref = _traced(_make_reference_model(n_modes), args, values)
    ref_sites = [k for k in ref if k.startswith("logl_")]
    assert len(ref_sites) == n_det
    ref_total = sum(float(ref[k]["fn"].log_factor) for k in ref_sites)
    total = float(new["logl_total"]["fn"].log_factor)
    assert abs(total - ref_total) / abs(ref_total) < RTOL


# -----------------------------------------------------------------------
# (c) prior mode
# -----------------------------------------------------------------------

@pytest.mark.parametrize("model_factory", ["production", "reference"])
def test_prior_mode_returns_unit_quadratures(model_factory):
    """With ``prior=True`` the draw is the standard normal itself."""
    n_det, n_t, n_modes = 2, 205, 2
    args = _make_data(n_det, n_t, n_modes)
    values = _fixed_values(n_modes)
    kws = dict(prior=True, predictive=True)
    if model_factory == "production":
        model = _make_production_model(n_modes, **kws)
    else:
        model = _make_reference_model(n_modes, **kws)

    trace = _traced(model, args, values)
    assert not [k for k in trace if k.startswith("logl")]

    a_scale = np.asarray(values["a_scale"])
    for quad in ("apx", "apy", "acx", "acy"):
        np.testing.assert_allclose(
            np.asarray(trace[quad]["value"]),
            np.asarray(values[f"{quad}_unit"]) * a_scale,
            rtol=1e-12,
        )


def test_prior_mode_matches_reference():
    n_det, n_t, n_modes = 2, 205, 2
    args = _make_data(n_det, n_t, n_modes)
    values = _fixed_values(n_modes)
    kws = dict(prior=True, predictive=True, store_h_det=True)

    new = _deterministic_values(
        _traced(_make_production_model(n_modes, **kws), args, values)
    )
    ref = _deterministic_values(
        _traced(_make_reference_model(n_modes, **kws), args, values)
    )
    assert set(new) == set(ref)
    for key in sorted(new):
        scale = max(np.max(np.abs(ref[key])), 1e-300)
        dev = np.max(np.abs(new[key] - ref[key])) / scale
        assert dev < RTOL, f"{key} differs by {dev:.3g}"


# -----------------------------------------------------------------------
# (d) sampling smoke test
# -----------------------------------------------------------------------

def _run_nuts(model, args, seed=0, warmup=300, samples=300):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), *args)
    return mcmc.get_samples()


def test_nuts_posteriors_agree():
    """Seeded NUTS gives the same posterior to within Monte Carlo error."""
    n_det, n_t, n_modes = 2, 205, 2
    args = _make_data(n_det, n_t, n_modes)

    new = _run_nuts(_make_production_model(n_modes), args)
    ref = _run_nuts(_make_reference_model(n_modes), args)

    for key in ("f", "g", "a_scale"):
        x_new = np.asarray(new[key])
        x_ref = np.asarray(ref[key])
        for j in range(n_modes):
            se_new = np.std(x_new[:, j]) / np.sqrt(
                float(effective_sample_size(x_new[None, :, j]))
            )
            se_ref = np.std(x_ref[:, j]) / np.sqrt(
                float(effective_sample_size(x_ref[None, :, j]))
            )
            se = np.sqrt(se_new ** 2 + se_ref ** 2)
            delta = abs(x_new[:, j].mean() - x_ref[:, j].mean())
            assert delta < 3 * se, (
                f"{key}[{j}] means differ by {delta:.3g} > 3 x {se:.3g}"
            )
