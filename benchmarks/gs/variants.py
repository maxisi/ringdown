"""Model factories for the Gohberg-Semencul (GS) benchmark kit.

Every variant is a numpyro model with the SAME sample sites, names, order and
priors as `ringdown.model.make_model(modes=modes, marginalized=True, **kw)` on
main: `m`, `chi` (Uniform), deterministics `f`, `g`, then `a_scale` (Uniform,
shape (n_modes,)), then a single `numpyro.factor("logl_total", ...)`.  The
variants differ only in how the per-detector contraction M_i^T C_i^{-1} M_i is
evaluated (docs/gohberg_semencul_likelihood.md section 11):

  main           ringdown.model.make_model literally (trsm against L, z and
                 log|C| recomputed every call)
  main_hoisted   same algebra, but z_i = L_i^{-1} y_i, Q and log|C| are
                 theta-independent constants passed in
  gemm_linv      dense L_i^{-1} passed in; W_i = L_i^{-1} M_i by GEMM
  gemm_cinv      dense C_i^{-1} passed in; C_i^{-1} M_i by GEMM (numerical control)
  gs_pr          GS with spectra of a and atilde passed in, PR #141's kernel
                 (flips for transposes, vmap over columns)
  gs_pr_ascoded  GS as PR #141 codes it: a passed in, spectra computed in the
                 model body from a (theta-independent work that XLA:CPU hoists
                 out of while loops and XLA:GPU recomputes per gradient)
  gs_full        batched GS producing C^{-1} M with conjugate-spectrum
                 correlations, 4 batched FFT passes
  gs_half        Gram form: A^{-1} += (P^T P - R^T R) / sigma^2 with
                 P = L(a)^T M, R = L(atilde)^T M (one rfft + one batched irfft)
  floor          all sites, no likelihood (kernel-launch floor)

Model signature for every variant: model(times, strains, consts, fps, fcs),
where `consts` is the dict returned by `build_consts`.  Constants are passed
as ARGUMENTS so that under jit they are entry parameters (as numpyro's MCMC
passes model args) rather than embedded HLO literals.  Note that with jax
0.11 XLA does NOT constant-fold work on such literals (an rfft of a
closed-over vector stays in the compiled gradient on CPU and GPU); what does
remove theta-independent work is loop-invariant code motion out of while
loops, which XLA:CPU applies (the timing fori_loop, NUTS tree building) and
XLA:GPU does not.  Passing the constants as arguments keeps the plain-gradient
census unambiguous; the looped census reports the hoisting (harness.py).

The one-shot algebra follows benchmarks/h100/bench.py make_oneshot_unroll_sep
(lines 412-438) and the prior/design-matrix head is bench.py's `_head`
(lines 295-309) with the `f`/`g` deterministics of ringdown.model added.

Unconstrained sites expected by numpyro's `potential_energy` for these
models (obtained from `initialize_model` on the main model, see
`check_unconstrained_sites`): {'m': (), 'chi': (), 'a_scale': (n_modes,)}.
All three are Uniform sites, so numpyro's biject_to gives
x = lo + (hi - lo) * sigmoid(z) with log|dx/dz| = log(hi - lo) + log sigmoid(z)
+ log sigmoid(-z), which `head_design_matrices` reproduces.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import initialize_model

import ringdown.model as rdm
from ringdown import qnms
from ringdown.model import chi_factors, rd_design_matrix

import gs_kernels as gk

VARIANTS = ["main", "main_hoisted", "gemm_linv", "gemm_cinv", "gs_pr",
            "gs_pr_ascoded", "gs_full", "gs_half", "floor"]
GS_VARIANTS = ["gs_pr", "gs_pr_ascoded", "gs_full", "gs_half"]

# prior-bound defaults shared by every model in the kit (bench.py KW with the
# tests/test_model.py a_scale_max); prep_inputs stores a_scale_max per family
KW_DEFAULT = dict(a_scale_max=1.0, m_min=50.0, m_max=150.0, chi_min=0.0,
                  chi_max=0.99)


def modes_of(n):
    return [(1, -2, 2, 2, i) for i in range(n)]


def _coeffs(modes):
    fc, gc = [], []
    for mo in modes:
        c = qnms.KerrMode(mo).coefficients
        fc.append(c[0])
        gc.append(c[1])
    return jnp.array(fc), jnp.array(gc)


# ---------------------------------------------------------------------------
# shared head: priors -> frequencies -> design matrix (bench.py:295-309, plus
# the f/g deterministic sites that ringdown.model.make_model emits)
# ---------------------------------------------------------------------------
def _head(modes, a_scale_max, m_min, m_max, chi_min, chi_max):
    """Shared non-linear front end: priors -> frequencies -> design matrix."""
    n_modes = len(modes)
    fco, gco = _coeffs(modes)

    def head(times, fps, fcs):
        m = numpyro.sample("m", dist.Uniform(m_min, m_max))
        chi = numpyro.sample("chi", dist.Uniform(chi_min, chi_max))
        f0 = 1 / (m * qnms.T_MSUN)
        f = numpyro.deterministic("f", f0 * chi_factors(chi, fco))
        g = numpyro.deterministic("g", f0 * chi_factors(chi, gco))
        a_scale = numpyro.sample("a_scale", dist.Uniform(0, a_scale_max),
                                 sample_shape=(n_modes,))
        return rd_design_matrix(times, f, g, fps, fcs, a_scale)
    return head


def _finish(A_inv, v, Q, logdetC):
    """One-shot tail shared by every hoisted variant (bench.py:433-437 with
    the exact log|C| constant in place of sum log diag L)."""
    Lc = jsp.linalg.cholesky(A_inv, lower=True)
    u = jsp.linalg.solve_triangular(Lc, v, lower=True)
    numpyro.factor("logl_total", -0.5 * Q + 0.5 * jnp.dot(u, u)
                   - 0.5 * logdetC - jnp.sum(jnp.log(jnp.diag(Lc))))


def _nfft(times, nfft_mode):
    """Padding length from the static time-axis length (trace-time int)."""
    return gk.nfft_for(int(times.shape[1]), nfft_mode)


def _kw(kw):
    out = dict(KW_DEFAULT)
    out.update(kw)
    return out


# ---------------------------------------------------------------------------
# model factories
# ---------------------------------------------------------------------------
def make_model(variant, modes, nfft_mode="pow2", **kw):
    """Return model(times, strains, consts, fps, fcs) for the named variant.

    `nfft_mode` ('pow2' or 'fast') selects the FFT padding for the GS
    variants; it must match the mode used in `build_consts` (the model
    asserts the spectra length agrees at trace time).
    """
    kw = _kw(kw)
    if variant not in VARIANTS:
        raise ValueError("unknown variant %r" % variant)

    if variant == "main":
        base = rdm.make_model(modes=modes, marginalized=True, **kw)

        def model(times, strains, consts, fps, fcs):
            return base(times, strains, consts["ls"], fps, fcs)
        return model

    head = _head(modes, **kw)

    def st(A, B):
        return jsp.linalg.solve_triangular(A, B, lower=True)

    if variant == "floor":
        def model(times, strains, consts, fps, fcs):
            times, fps, fcs = map(jnp.asarray, (times, fps, fcs))
            dms = head(times, fps, fcs)
            # 0 * x keeps the design matrix alive (XLA does not fold float
            # multiplication by zero) while contributing exactly nothing, so
            # U_floor is the pure prior potential with identical sites
            numpyro.factor("logl_total", 0.0 * jnp.sum(dms[0, 0, :]))
        return model

    if variant == "main_hoisted":
        def model(times, strains, consts, fps, fcs):
            times, fps, fcs = map(jnp.asarray, (times, fps, fcs))
            dms = head(times, fps, fcs)
            k, n_det = dms.shape[2], dms.shape[0]
            ls, z = consts["ls"], consts["z"]
            A_inv, v = jnp.eye(k, dtype=dms.dtype), jnp.zeros(k, dtype=dms.dtype)
            for i in range(n_det):
                W = st(ls[i], dms[i])
                A_inv = A_inv + W.T @ W
                v = v + W.T @ z[i]
            _finish(A_inv, v, jnp.sum(consts["Q"]), jnp.sum(consts["logdetC"]))
        return model

    if variant == "gemm_linv":
        def model(times, strains, consts, fps, fcs):
            times, fps, fcs = map(jnp.asarray, (times, fps, fcs))
            dms = head(times, fps, fcs)
            k, n_det = dms.shape[2], dms.shape[0]
            Linv, w = consts["Linv"], consts["w"]
            A_inv, v = jnp.eye(k, dtype=dms.dtype), jnp.zeros(k, dtype=dms.dtype)
            for i in range(n_det):
                W = Linv[i] @ dms[i]
                A_inv = A_inv + W.T @ W
                v = v + dms[i].T @ w[i]
            _finish(A_inv, v, jnp.sum(consts["Q"]), jnp.sum(consts["logdetC"]))
        return model

    if variant == "gemm_cinv":
        def model(times, strains, consts, fps, fcs):
            times, fps, fcs = map(jnp.asarray, (times, fps, fcs))
            dms = head(times, fps, fcs)
            k, n_det = dms.shape[2], dms.shape[0]
            Cinv, w = consts["Cinv"], consts["w"]
            A_inv, v = jnp.eye(k, dtype=dms.dtype), jnp.zeros(k, dtype=dms.dtype)
            for i in range(n_det):
                M = dms[i]
                S = M.T @ (Cinv[i] @ M)
                S = 0.5 * (S + S.T)
                A_inv = A_inv + S
                v = v + M.T @ w[i]
            _finish(A_inv, v, jnp.sum(consts["Q"]), jnp.sum(consts["logdetC"]))
        return model

    if variant in ("gs_pr", "gs_full"):
        applier = gk.gs_pr_cinv if variant == "gs_pr" else gk.gs_full_cinv

        def model(times, strains, consts, fps, fcs):
            times, fps, fcs = map(jnp.asarray, (times, fps, fcs))
            nfft = _nfft(times, nfft_mode)
            ah, bh, s2, w = consts["ah"], consts["bh"], consts["sigma2"], consts["w"]
            assert ah.shape[-1] == nfft // 2 + 1, \
                "consts built with a different nfft mode than the model"
            dms = head(times, fps, fcs)
            k, n_det = dms.shape[2], dms.shape[0]
            A_inv, v = jnp.eye(k, dtype=dms.dtype), jnp.zeros(k, dtype=dms.dtype)
            for i in range(n_det):
                M = dms[i]
                CiM = applier(M, ah[i], bh[i], nfft, s2[i])
                S = M.T @ CiM
                S = 0.5 * (S + S.T)
                A_inv = A_inv + S
                v = v + M.T @ w[i]
            _finish(A_inv, v, jnp.sum(consts["Q"]), jnp.sum(consts["logdetC"]))
        return model

    if variant == "gs_pr_ascoded":
        def model(times, strains, consts, fps, fcs):
            times, fps, fcs = map(jnp.asarray, (times, fps, fcs))
            nfft = _nfft(times, nfft_mode)
            a, s2, w = consts["a"], consts["sigma2"], consts["w"]
            n_det = a.shape[0]
            # exactly as PR #141's model body: spectra of a and of
            # atilde = (0, a_{N-1}, ..., a_1) computed from the AR coefficients
            # inside the traced function (jnp.pad(ac[1:][::-1], (1, 0)))
            fft_as, fft_bs = [], []
            for i in range(n_det):
                ac = a[i]
                rev = jnp.pad(ac[1:][::-1], (1, 0))
                fft_as.append(jnp.fft.rfft(ac, n=nfft))
                fft_bs.append(jnp.fft.rfft(rev, n=nfft))
            dms = head(times, fps, fcs)
            k = dms.shape[2]
            A_inv, v = jnp.eye(k, dtype=dms.dtype), jnp.zeros(k, dtype=dms.dtype)
            for i in range(n_det):
                M = dms[i]
                CiM = gk.gs_pr_cinv(M, fft_as[i], fft_bs[i], nfft, s2[i])
                S = M.T @ CiM
                S = 0.5 * (S + S.T)
                A_inv = A_inv + S
                v = v + M.T @ w[i]
            _finish(A_inv, v, jnp.sum(consts["Q"]), jnp.sum(consts["logdetC"]))
        return model

    if variant == "gs_half":
        def model(times, strains, consts, fps, fcs):
            times, fps, fcs = map(jnp.asarray, (times, fps, fcs))
            nfft = _nfft(times, nfft_mode)
            ah, bh, s2, w = consts["ah"], consts["bh"], consts["sigma2"], consts["w"]
            assert ah.shape[-1] == nfft // 2 + 1, \
                "consts built with a different nfft mode than the model"
            dms = head(times, fps, fcs)
            k, n_det = dms.shape[2], dms.shape[0]
            A_inv, v = jnp.eye(k, dtype=dms.dtype), jnp.zeros(k, dtype=dms.dtype)
            for i in range(n_det):
                M = dms[i]
                P, R = gk.gs_half_grams(M, ah[i], bh[i], nfft, s2[i])
                A_inv = A_inv + (P.T @ P - R.T @ R) / s2[i]
                v = v + M.T @ w[i]
            _finish(A_inv, v, jnp.sum(consts["Q"]), jnp.sum(consts["logdetC"]))
        return model

    raise ValueError(variant)   # pragma: no cover


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
CONST_KEYS = {
    "main": ["ls"],
    "main_hoisted": ["ls", "z", "Q", "logdetC"],
    "gemm_linv": ["Linv", "w", "Q", "logdetC"],
    "gemm_cinv": ["Cinv", "w", "Q", "logdetC"],
    "gs_pr": ["ah", "bh", "sigma2", "w", "Q", "logdetC"],
    "gs_full": ["ah", "bh", "sigma2", "w", "Q", "logdetC"],
    "gs_half": ["ah", "bh", "sigma2", "w", "Q", "logdetC"],
    "gs_pr_ascoded": ["a", "sigma2", "w", "Q", "logdetC"],
    "floor": [],
}


COEFFS_POLICIES = ("pr", "refined")
SPECTRA_POLICIES = ("f64", "leg")


def logdetC_source(inputs):
    """Where the inputs' `logdetC_exact` comes from: 'longdouble' for files
    written (or refreshed) by the current prep_inputs.py, else 'f64lev' for
    older files whose logdetC_exact is the float64 Levinson sum (biased by
    ~eps cond N: 2.7e-8 nats at cond 5e8, 1.6e-4 at 1.5e12)."""
    return "longdouble" if "logdetC_f64lev" in inputs else "f64lev"


def build_consts(variant, inputs, dtype, nfft_mode="pow2", coeffs="pr",
                 spectra_from="f64"):
    """Theta-independent per-detector constants for `variant`, as jnp arrays.

    `inputs` is a mapping with the prep_inputs.py keys (npz or dict):
    L, Linv, Cinv (n_det,N,N); a, atilde, w, z (n_det,N); sigma2, Q,
    logdetC_exact (n_det,).  Everything is cast to `dtype` (spectra to the
    matching complex dtype).  The exact log-determinant is used for every
    variant (the PR's N log sigma^2 shortcut is a diagnostic, not a model);
    see `logdetC_source` for which evaluation of it the inputs carry.

    `coeffs` selects the Yule-Walker filter the GS variants gs_pr, gs_full,
    gs_half are built from: 'pr' (default) = a, atilde, sigma2 from
    scipy.linalg.solve_toeplitz as PR #141 computes them; 'refined' = a_ld,
    atilde_ld, sigma2_ld from the longdouble Levinson recursion rounded to
    float64 (needs inputs with those keys).  Both feed the same compiled
    model, so the difference isolates the precompute's forward error (F1 of
    the note's section 9.3) from the FFT route itself.  gs_pr_ascoded always
    takes the PR filter (it is the PR's model body).

    `spectra_from` selects how the spectra ah, bh are formed for gs_pr,
    gs_full, gs_half in a float32 leg: 'f64' (default) = rfft in float64 of
    the float64 filter, then cast to complex64; 'leg' = cast the filter to
    the leg dtype first and rfft in that dtype (what gs_pr_ascoded's in-model
    jnp.fft.rfft does, and what a float32 production path would do).  The two
    differ at the float32-eps level (~1e-7 relative).  Irrelevant in float64.

    The padding length is not returned (it must be a trace-time Python int,
    not a traced array); the model derives it from times.shape[1] with the
    same `gs_kernels.nfft_for(N, nfft_mode)`.  Use `nfft_of(inputs, mode)`
    to get the number the consts were built with.
    """
    real = np.dtype(dtype)
    cplx = np.complex128 if real == np.float64 else np.complex64
    if coeffs not in COEFFS_POLICIES:
        raise ValueError("coeffs must be one of %r" % (COEFFS_POLICIES,))
    if spectra_from not in SPECTRA_POLICIES:
        raise ValueError("spectra_from must be one of %r" % (SPECTRA_POLICIES,))
    keys = CONST_KEYS[variant]
    out = {}
    if coeffs == "refined" and variant in ("gs_pr", "gs_full", "gs_half"):
        for k in ("a_ld", "atilde_ld", "sigma2_ld"):
            if k not in inputs:
                raise KeyError("coeffs='refined' needs inputs key %r (run "
                               "prep_inputs.py --refresh-precompute)" % k)
        k_a, k_at, k_s2 = "a_ld", "atilde_ld", "sigma2_ld"
    else:
        k_a, k_at, k_s2 = "a", "atilde", "sigma2"

    def arr(name, dt=real):
        return jnp.asarray(np.asarray(inputs[name]), dtype=dt)

    if "ls" in keys:
        out["ls"] = arr("L")
    if "z" in keys:
        out["z"] = arr("z")
    if "Linv" in keys:
        out["Linv"] = arr("Linv")
    if "Cinv" in keys:
        out["Cinv"] = arr("Cinv")
    if "w" in keys:
        out["w"] = arr("w")
    if "Q" in keys:
        out["Q"] = arr("Q")
    if "logdetC" in keys:
        out["logdetC"] = arr("logdetC_exact")
    if "sigma2" in keys:
        out["sigma2"] = arr(k_s2)
    if "a" in keys:
        out["a"] = arr("a")          # gs_pr_ascoded: the PR's filter, always
    if "ah" in keys:
        a = np.asarray(inputs[k_a], dtype=np.float64)
        at = np.asarray(inputs[k_at], dtype=np.float64)
        nfft = nfft_of(inputs, nfft_mode)
        fft_dtype = real if spectra_from == "leg" else np.float64
        ah, bh = [], []
        for i in range(a.shape[0]):
            ahi, bhi = gk.spectra(a[i], at[i], nfft, dtype=fft_dtype)
            ah.append(ahi)
            bh.append(bhi)
        out["ah"] = jnp.asarray(np.stack(ah), dtype=cplx)
        out["bh"] = jnp.asarray(np.stack(bh), dtype=cplx)
    return out


def nfft_of(inputs, nfft_mode="pow2"):
    N = int(np.asarray(inputs["times"]).shape[1])
    return gk.nfft_for(N, nfft_mode)


def data_args(inputs, dtype):
    """(times, strains, fps, fcs) from an inputs mapping, cast to dtype."""
    return tuple(jnp.asarray(np.asarray(inputs[k]), dtype=dtype)
                 for k in ("times", "strains", "fps", "fcs"))


# ---------------------------------------------------------------------------
# unconstrained-coordinate head for the extended-precision reference
# ---------------------------------------------------------------------------
UNCONSTRAINED_SITES = ("m", "chi", "a_scale")


def unconstrained_site_shapes(n_modes):
    """Site names and shapes of the unconstrained point `potential_energy`
    expects for these models: m and chi are scalars, a_scale has shape
    (n_modes,).  Site order here is the sampling order in the model."""
    return {"m": (), "chi": (), "a_scale": (n_modes,)}


def check_unconstrained_sites(model, args, n_modes, seed=1):
    """Assert, via numpyro.infer.util.initialize_model, that `model` has
    exactly the unconstrained sites of `unconstrained_site_shapes`."""
    init = initialize_model(jax.random.PRNGKey(seed), model, model_args=args)
    got = {k: tuple(v.shape) for k, v in init.param_info.z.items()}
    exp = unconstrained_site_shapes(n_modes)
    if got != exp:
        raise AssertionError("unconstrained sites %r != expected %r" % (got, exp))
    return got


def head_design_matrices(modes, **kw):
    """Pure-JAX twin of the model head in unconstrained coordinates.

    Returns f(z, times, fps, fcs) -> (dms, U_prior) where z is the dict
    {'m': (), 'chi': (), 'a_scale': (n_modes,)} of unconstrained values,
    dms has shape (n_det, N, k) and U_prior = -(sum of prior log densities
    + log|det Jacobian|) is the prior part of numpyro's potential energy,
    i.e. exactly potential_energy of the `floor` variant.  Differentiable
    in z, so jax.vjp pulls a longdouble dlogL/dM back to a gradient in
    unconstrained coordinates.
    """
    kw = _kw(kw)
    n_modes = len(modes)
    fco, gco = _coeffs(modes)
    dists = {"m": dist.Uniform(kw["m_min"], kw["m_max"]),
             "chi": dist.Uniform(kw["chi_min"], kw["chi_max"]),
             "a_scale": dist.Uniform(0.0, kw["a_scale_max"])}

    def f(z, times, fps, fcs):
        x, logj = {}, 0.0
        for name in UNCONSTRAINED_SITES:
            d = dists[name]
            t = biject_to(d.support)
            xv = t(z[name])
            x[name] = xv
            # numpyro's potential: -sum(log_prob(x) + log|dx/dz|)
            logj = logj + jnp.sum(d.log_prob(xv)) \
                + jnp.sum(t.log_abs_det_jacobian(z[name], xv))
        f0 = 1 / (x["m"] * qnms.T_MSUN)
        fr = f0 * chi_factors(x["chi"], fco)
        gr = f0 * chi_factors(x["chi"], gco)
        dms = rd_design_matrix(times, fr, gr, fps, fcs, x["a_scale"])
        assert dms.shape[2] == 4 * n_modes
        return dms, -logj
    return f
