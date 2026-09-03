"""Self-tests for gs_kernels.py and ref_longdouble.py.

Plain script, no pytest. Run from the repo root:

    JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python benchmarks/gs/tests_kernels.py

It sets JAX_PLATFORMS=cpu (if unset) and jax_enable_x64 before importing
jax. Every check prints the number it measured next to the tolerance it was
held to; a failed check raises at the end after all checks have run, so the
full table is always visible.

ACF families: white, exp-cos (benchmarks/h100/bench.py make_args) and an
aLIGO-like ACF with a floored low-frequency wall (copy of
tests/test_model.py _aligo_acf) whose Toeplitz matrix has cond ~1e10 at
N = 205. Sizes N = 64, 205 for everything, N = 1024 for the cheap checks.

Tolerances that involve cond(C) are scaled as c * cond(C) * eps64 with a
floor, because both routes under comparison (GS and Cholesky) are only
accurate to that level; the actual numbers are what matters and are
printed.
"""

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import scipy.linalg

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gs_kernels as gk  # noqa: E402
import ref_longdouble as rl  # noqa: E402

EPS = np.finfo(np.float64).eps
DT = 1.0 / 2048.0

FAILURES = []


def check(name, value, tol, cmp="<="):
    """Record one check. cmp is '<=' (value must be <= tol) or '>' (must exceed)."""
    ok = (value <= tol) if cmp == "<=" else (value > tol)
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {name:<62s} {value:11.3e}  ({cmp} {tol:.2e})")
    if not ok:
        FAILURES.append((name, value, tol, cmp))
    return ok


def relmax(x, ref):
    """max |x - ref| / max |ref|; plain max |x - ref| when ref is identically zero
    (white noise has atilde = 0 exactly)."""
    scale = float(np.max(np.abs(ref)))
    err = float(np.max(np.abs(x - ref)))
    return err / scale if scale > 0 else err


# -----------------------------------------------------------------------------
# ACF families
# -----------------------------------------------------------------------------


def _aligo_psd(f):
    """Analytic aLIGO-like PSD (copy of tests/test_model.py)."""
    x = f / 215.0
    return 1e-49 * (x**-4.14 - 5 * x**-2 + 111 * (1 - x**2 + 0.5 * x**4) / (1 + 0.5 * x**2))


def aligo_acf(n, dt=DT, f_low=0.2, n_fft=4096):
    """Unit-normalized aLIGO-like ACF (copy of tests/test_model.py _aligo_acf)."""
    if n > 2048:
        n_fft = 16384
    df = 1.0 / (n_fft * dt)
    f = np.arange(n_fft // 2 + 1) * df
    psd = np.where(f >= f_low, _aligo_psd(np.maximum(f, f_low)), _aligo_psd(f_low))
    rho = np.fft.irfft(psd)[:n_fft] / dt
    return rho[:n] / rho[0]


def expcos_acf(n, dt=DT):
    """benchmarks/h100/bench.py make_args ACF, with its diagonal jitter folded in."""
    lags = np.arange(n) * dt
    acf = np.exp(-lags / 0.01) * np.cos(2 * np.pi * 120 * lags) + 1e-3 * (lags == 0)
    acf[0] += n * 1e-9
    return acf


def white_acf(n):
    acf = np.zeros(n)
    acf[0] = 1.0
    return acf


FAMILIES = {"white": white_acf, "expcos": expcos_acf, "aligo": aligo_acf}


# -----------------------------------------------------------------------------
# per-(family, N) checks
# -----------------------------------------------------------------------------


def run_case(fam, N, k=8, cheap_only=False, seed=0):
    rng = np.random.default_rng(seed)
    acf = FAMILIES[fam](N)
    t0 = time.time()
    lev = gk.levinson(acf)
    den = gk.dense_factors(acf)
    C, L, Cinv, cond = den["C"], den["L"], den["Cinv"], den["cond"]
    a, atilde, sigma2 = lev["a"], lev["atilde"], lev["sigma2"]
    print(f"\n=== {fam}  N={N}  cond(C)={cond:.3e}  sigma2/gamma0={sigma2 / acf[0]:.3e}  "
          f"max|kappa|={np.max(np.abs(lev['refl'])):.6f}  levinson_resid={lev['levinson_resid']:.2e}  "
          f"(precompute {time.time() - t0:.2f}s)")
    ctol = max(1e-12, 10 * cond * EPS)  # conditioning-limited tolerance

    # -- Levinson vs first column of inv(C), eq. (3.6) ------------------------
    x = a / sigma2
    check("a/sigma2 vs Cinv[:,0]  (relmax)", relmax(x, Cinv[:, 0]), ctol)
    check("levinson_resid ||Ca - s2 e0||/(||C|| ||a||)", lev["levinson_resid"], 100 * EPS)

    # -- innovation variances and log-determinant -----------------------------
    check("sigma2_all[-1] vs sigma2  (rel)", abs(lev["sigma2_all"][-1] - sigma2) / sigma2, ctol)
    sign, ld = np.linalg.slogdet(C)
    assert sign > 0
    check("logdetC_f64lev vs slogdet  (rel)", abs(lev["logdetC_f64lev"] - ld) / max(abs(ld), 1.0), 1e-8)
    gap = lev["logdetC_pr"] - lev["logdetC_f64lev"]
    check("logdetC_pr - logdetC_f64lev <= 0   (value shown)", gap, 1e-9)
    if fam == "white":
        check("white: logdetC_pr == logdetC_f64lev  (abs)", abs(gap), 1e-9)
    check("logdet_levinson_ld vs slogdet  (rel)",
          abs(float(rl.logdet_levinson_ld(acf)) - ld) / max(abs(ld), 1.0), 1e-8)
    check("logdet_levinson_ld vs 2 sum log diag L  (rel)",
          abs(float(rl.logdet_levinson_ld(acf)) - 2 * np.sum(np.log(np.diag(L)))) / max(abs(ld), 1.0), 1e-8)
    check("logdetC_f64lev vs logdet_levinson_ld  (rel)",
          abs(lev["logdetC_f64lev"] - float(rl.logdet_levinson_ld(acf))) / max(abs(ld), 1.0), 1e-8)

    # -- extended-precision reference -----------------------------------------
    M = rng.normal(size=(N, k))
    y = L @ rng.normal(size=N) + M @ rng.normal(size=k)
    ref_CiM = rl.refine_solve(L, C, M, rounds=2)
    ref_w = rl.refine_solve(L, C, y, rounds=2)
    check("refine_solve residual (longdouble, rel)", rl.ld_residual(C, M, ref_CiM), 1e-14)
    check("refine_solve residual for y (longdouble, rel)", rl.ld_residual(C, y[:, None], ref_w[:, None]), 1e-14)
    chol_CiM = scipy.linalg.cho_solve((L, True), M)
    ref64 = ref_CiM.astype(np.float64)
    print(f"        (info) cho_solve vs refined reference: {relmax(chol_CiM, ref64):.3e}")

    # -- GS formula, dense (isolates Levinson error from FFT error) ------------
    if not cheap_only:
        Cinv_gs = gk.np_dense_gs_cinv(a, atilde, sigma2)
        check("dense GS (4.4) vs Cinv  (relmax)", relmax(Cinv_gs, Cinv), ctol)
        check("dense GS vs Cinv, symmetric part only (sanity)", relmax(0.5 * (Cinv_gs + Cinv_gs.T), Cinv), ctol)

    # -- FFT appliers vs cho_solve, both padding modes, numpy and jax -----------
    for mode in ("pow2", "fast"):
        nfft = gk.nfft_for(N, mode)
        assert nfft >= 2 * N - 1
        ah, bh = gk.spectra(a, atilde, nfft)
        jM, jah, jbh = jnp.asarray(M), jnp.asarray(ah), jnp.asarray(bh)
        assert jM.dtype == jnp.float64 and jah.dtype == jnp.complex128
        outs = {
            "np_gs_pr_cinv": gk.np_gs_pr_cinv(M, ah, bh, nfft, sigma2),
            "np_gs_full_cinv": gk.np_gs_full_cinv(M, ah, bh, nfft, sigma2),
            "gs_pr_cinv": np.asarray(jax.jit(gk.gs_pr_cinv, static_argnums=3)(jM, jah, jbh, nfft, sigma2)),
            "gs_full_cinv": np.asarray(jax.jit(gk.gs_full_cinv, static_argnums=3)(jM, jah, jbh, nfft, sigma2)),
        }
        for name, out in outs.items():
            assert out.shape == (N, k), (name, out.shape)
            check(f"{name:<16s} nfft={nfft:5d} ({mode}) vs cho_solve (relmax)", relmax(out, chol_CiM), ctol)
            check(f"{name:<16s} nfft={nfft:5d} ({mode}) vs refined ref (relmax)", relmax(out, ref64), ctol)
        # the half-Gram route: (P^T P - R^T R)/sigma2 == M^T C^{-1} M
        G_ref = M.T @ chol_CiM
        G_ref = 0.5 * (G_ref + G_ref.T)
        for name, fn in (("np_gs_half_grams", gk.np_gs_half_grams),
                         ("gs_half_grams", jax.jit(gk.gs_half_grams, static_argnums=3))):
            args = (M, ah, bh, nfft, sigma2) if name.startswith("np") else (jM, jah, jbh, nfft, sigma2)
            P, R = fn(*args)
            P, R = np.asarray(P), np.asarray(R)
            assert P.shape == (N, k) and R.shape == (N, k)
            G = (P.T @ P - R.T @ R) / sigma2
            check(f"{name:<16s} nfft={nfft:5d} ({mode}) Gram vs M^T cho_solve(M)", relmax(G, G_ref), ctol)
            if name.startswith("np"):
                # cross-check the factors against dense L(a)^T M, L(atilde)^T M
                La, Lb = gk.lower_toeplitz(a), gk.lower_toeplitz(atilde)
                check(f"{name:<16s} nfft={nfft:5d} ({mode}) P vs dense L(a)^T M", relmax(P, La.T @ M), 1e-12)
                check(f"{name:<16s} nfft={nfft:5d} ({mode}) R vs dense L(a~)^T M", relmax(R, Lb.T @ M), 1e-12)
                print(f"        (info) Gram cancellation ratio ||P^T P|| / ||P^T P - R^T R||: "
                      f"{np.linalg.norm(P.T @ P) / np.linalg.norm(P.T @ P - R.T @ R):.3f}")
        # numpy and jax agree to roundoff (same padding, same algebra)
        check(f"jax vs numpy gs_full_cinv agree  ({mode})", relmax(outs["gs_full_cinv"], outs["np_gs_full_cinv"]), 1e-10)
        check(f"jax vs numpy gs_pr_cinv agree    ({mode})", relmax(outs["gs_pr_cinv"], outs["np_gs_pr_cinv"]), 1e-10)
        if mode == "pow2":
            t1, t2 = gk.np_gs_full_terms(M, ah, bh, nfft)
            print(f"        (info) vector cancellation ratio ||L(a)L(a)^T M|| / ||sigma2 C^-1 M||: "
                  f"{np.linalg.norm(t1) / np.linalg.norm(t1 - t2):.3f}")

    # -- padding bound (5.2): 2N-2 wraps around, 2N-1 does not -----------------
    for nfft, must_pass in ((2 * N - 2, False), (2 * N - 1, True)):
        ah, bh = gk.spectra(a, atilde, nfft)
        errs = {
            "np_gs_full_cinv": relmax(gk.np_gs_full_cinv(M, ah, bh, nfft, sigma2), chol_CiM),
            "np_gs_pr_cinv": relmax(gk.np_gs_pr_cinv(M, ah, bh, nfft, sigma2), chol_CiM),
            "gs_full_cinv": relmax(np.asarray(gk.gs_full_cinv(jnp.asarray(M), jnp.asarray(ah), jnp.asarray(bh), nfft, sigma2)), chol_CiM),
            "gs_pr_cinv": relmax(np.asarray(gk.gs_pr_cinv(jnp.asarray(M), jnp.asarray(ah), jnp.asarray(bh), nfft, sigma2)), chol_CiM),
        }
        for name, e in errs.items():
            if must_pass:
                check(f"{name:<16s} nfft=2N-1={nfft} passes", e, ctol)
            elif fam == "white":
                # a = e0, atilde = 0: nothing to wrap, so no error is expected
                print(f"        (info) {name:<16s} nfft=2N-2={nfft} white noise (no wrap possible): {e:.3e}")
            else:
                check(f"{name:<16s} nfft=2N-2={nfft} FAILS (> 1e-3)", e, 1e-3, cmp=">")


def run_fd():
    print("\n=== oneshot_core_ld gradient vs longdouble central finite differences (N=16, k=4, n_det=2)")
    worst = rl.fd_selftest(N=16, k=4, n_det=2, h=1e-6, rounds=3, verbose=True)
    check("fd gradient worst relative error", worst, 1e-9)
    # also a lower-level sanity check: the closed form against the note's
    # description with a different hand-built alpha route
    M_list, y_list, L_list, C_list, logdet_list = rl._tiny_problem(16, 4, 2)
    out = rl.oneshot_core_ld(M_list, y_list, L_list, C_list, logdet_list)
    A_inv64 = np.eye(4)
    v64 = np.zeros(4)
    Q64 = 0.0
    for M, y, L, ld in zip(M_list, y_list, L_list, logdet_list):
        W = scipy.linalg.solve_triangular(L, M, lower=True)
        z = scipy.linalg.solve_triangular(L, y, lower=True)
        A_inv64 += W.T @ W
        v64 += W.T @ z
        Q64 += z @ z
    R64 = np.linalg.cholesky(A_inv64)
    u64 = scipy.linalg.solve_triangular(R64, v64, lower=True)
    ll64 = -0.5 * Q64 + 0.5 * u64 @ u64 - 0.5 * float(np.sum(np.asarray(logdet_list, dtype=np.float64))) - np.sum(np.log(np.diag(R64)))
    check("oneshot_core_ld loglike vs f64 main-style algebra (abs)", abs(float(out["loglike"]) - ll64), 1e-10)


def run_f32():
    """dtype propagation: float32 M with complex64 spectra and a python-float
    sigma2 must come back float32 (no silent promotion), with f32-level error
    on a well-conditioned ACF."""
    print("\n=== float32 dtype propagation of the jax appliers (expcos, N=205)")
    N, k = 205, 8
    rng = np.random.default_rng(1)
    acf = expcos_acf(N)
    lev = gk.levinson(acf)
    den = gk.dense_factors(acf)
    M = rng.normal(size=(N, k))
    chol_CiM = scipy.linalg.cho_solve((den["L"], True), M)
    nfft = gk.nfft_for(N, "pow2")
    ah, bh = gk.spectra(lev["a"], lev["atilde"], nfft)
    jM = jnp.asarray(M, dtype=jnp.float32)
    jah, jbh = jnp.asarray(ah, dtype=jnp.complex64), jnp.asarray(bh, dtype=jnp.complex64)
    s2 = float(lev["sigma2"])
    for name, fn in (("gs_pr_cinv", gk.gs_pr_cinv), ("gs_full_cinv", gk.gs_full_cinv)):
        out = jax.jit(fn, static_argnums=3)(jM, jah, jbh, nfft, s2)
        assert out.dtype == jnp.float32, (name, out.dtype)
        check(f"{name:<16s} f32 vs f64 cho_solve (relmax)", relmax(np.asarray(out, dtype=np.float64), chol_CiM), 1e-3)
    P, R = jax.jit(gk.gs_half_grams, static_argnums=3)(jM, jah, jbh, nfft, s2)
    assert P.dtype == jnp.float32 and R.dtype == jnp.float32
    G = (np.asarray(P, dtype=np.float64).T @ np.asarray(P, dtype=np.float64)
         - np.asarray(R, dtype=np.float64).T @ np.asarray(R, dtype=np.float64)) / s2
    G_ref = M.T @ chol_CiM
    check("gs_half_grams    f32 Gram vs f64 (relmax)", relmax(G, 0.5 * (G_ref + G_ref.T)), 1e-3)


def main():
    t0 = time.time()
    print(f"jax {jax.__version__} backend={jax.default_backend()} x64={jax.config.jax_enable_x64} "
          f"numpy {np.__version__} scipy={scipy.__version__} longdouble eps={rl.EPS_LD:.2e}")
    for N in (64, 205):
        for fam in FAMILIES:
            run_case(fam, N, k=8)
    for fam in FAMILIES:
        run_case(fam, 1024, k=8, cheap_only=True)
    run_fd()
    run_f32()
    print(f"\n{len(FAILURES)} failed checks; total time {time.time() - t0:.1f}s")
    if FAILURES:
        for name, value, tol, cmp in FAILURES:
            print(f"  FAIL {name}: {value:.3e} not {cmp} {tol:.2e}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
