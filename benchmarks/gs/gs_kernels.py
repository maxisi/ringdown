"""Gohberg-Semencul (GS) kernels for the benchmark kit.

Two layers, deliberately separated:

1. Numpy/scipy precompute (float64 in, float64 out): Levinson-Durbin,
   innovation variances, exact log-determinant, dense reference factors,
   filter spectra, FFT padding lengths. Run once per (ACF, N).

2. Appliers that turn the precomputed constants into C^{-1} M (or the two
   half-Gram factors) for a matrix M of shape (N, k). Each applier exists
   twice with the same signature: a JAX version (jnp, used inside the
   numpyro models of variants.py) and a numpy twin prefixed np_ (used by
   the self-tests, the extended-precision reference and the diagnostics).

Notation follows docs/gohberg_semencul_likelihood.md:

    C a = sigma2 e0,     a[0] = 1                         (3.5)
    atilde = (0, a[N-1], ..., a[1])                       (4.2)
    C^{-1} = [L(a) L(a)^T - L(atilde) L(atilde)^T] / sigma2   (4.4)
    log|C| = sum_m log sigma_m^2                          (3.10)

where L(x) is the lower-triangular Toeplitz matrix with first column x, so
L(x) psi is the causal convolution x * psi and L(x)^T psi is the
cross-correlation. Both are done with zero-padded FFTs of length nfft >= 2N-1
(5.2); the transposes are done either by flipping (5.3, the PR's route) or
by conjugating the filter spectrum (correlation theorem, the V4/V6 route).

No ringdown import here. JAX is imported lazily so the numpy layer works
in a jax-free interpreter.
"""

import numpy as np
import scipy.fft
import scipy.linalg

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # numpy-only use is allowed
    jax = None
    jnp = None


# =============================================================================
# numpy precompute
# =============================================================================


def levinson_durbin(acf, dtype=np.float64):
    """Levinson-Durbin recursion, eq. (3.9) of the note.

    Returns (a, sigma2_all, refl):
      a          (N,) order-(N-1) prediction-error filter, a[0] = 1
      sigma2_all (N,) innovation variances sigma_m^2 for m = 0..N-1
      refl       (N-1,) reflection coefficients kappa_m, m = 1..N-1

    Plain O(N^2) loop with vectorized inner products; `dtype` may be
    np.longdouble (ref_longdouble.py has its own copy so that the reference
    does not depend on this module).
    """
    acf = np.asarray(acf, dtype=dtype)
    N = acf.shape[0]
    a = np.zeros(N, dtype=dtype)
    a[0] = 1
    sigma2_all = np.empty(N, dtype=dtype)
    refl = np.empty(max(N - 1, 0), dtype=dtype)
    s = acf[0]
    sigma2_all[0] = s
    for m in range(1, N):
        # kappa_m = -(gamma_m + sum_{j=1}^{m-1} a_j gamma_{m-j}) / sigma_{m-1}^2
        kappa = -(acf[m] + np.dot(a[1:m], acf[m - 1:0:-1])) / s
        # a^(m) = (a^(m-1), 0) + kappa * (0, J a^(m-1))
        a[: m + 1] = a[: m + 1] + kappa * a[m::-1]
        s = s * (1 - kappa * kappa)
        sigma2_all[m] = s
        refl[m - 1] = kappa
    return a, sigma2_all, refl


def levinson(acf):
    """Yule-Walker solve and innovation bookkeeping for one ACF (float64).

    Returns a dict with
      a             (N,)  a[0] = 1, from scipy.linalg.solve_toeplitz exactly
                          as PR #141's Fit.run_input does
      atilde        (N,)  (0, a[N-1], ..., a[1])
      sigma2        float acf[0] + a[1:] @ acf[1:]           (3.4), PR's sigma_sq
      sigma2_all    (N,)  sigma_m^2 from the Levinson-Durbin recursion
      refl          (N-1,) reflection coefficients
      logdetC_f64lev float sum(log(sigma2_all)) in float64    (3.10)
      logdetC_pr    float N * log(sigma2)                    (8.2)
      levinson_resid float ||C a - sigma2 e0|| / (||C||_F ||a||)

    `logdetC_f64lev` is the float64 evaluation of the exact formula (3.10).
    It is NOT accurate enough to serve as the model constant at high
    conditioning: each sigma_m^2 carries an absolute error ~ eps ||acf|| that
    is ~ eps cond relative to the smallest innovation variances, and the N
    relative errors add up (gw150914 N=1024, cond 5e8: 2.7e-8 nats; N=4096,
    cond 1.5e12: 1.6e-4 nats, versus 2e-11 / 1e-7 for the longdouble
    recursion and 9e-10 / 2e-6 for 2 sum log diag L).  prep_inputs.py stores
    the longdouble value (ref_longdouble.levinson_ld) as `logdetC_exact` and
    this one as a diagnostic.
    """
    acf = np.asarray(acf, dtype=np.float64)
    N = acf.shape[0]
    # PR #141: R = acf[:-1], r = acf[1:], a' = solve_toeplitz((R, R), -r)
    a1 = scipy.linalg.solve_toeplitz((acf[:-1], acf[:-1]), -acf[1:])
    a = np.concatenate([[1.0], a1])
    atilde = np.concatenate([[0.0], a[1:][::-1]])
    sigma2 = float(acf[0] + np.dot(a[1:], acf[1:]))

    _, sigma2_all, refl = levinson_durbin(acf, dtype=np.float64)
    logdetC_f64lev = float(np.sum(np.log(sigma2_all)))
    logdetC_pr = float(N * np.log(sigma2))

    C = scipy.linalg.toeplitz(acf)
    e0 = np.zeros(N)
    e0[0] = 1.0
    resid = C @ a - sigma2 * e0
    levinson_resid = float(
        np.linalg.norm(resid) / (np.linalg.norm(C) * np.linalg.norm(a))
    )
    return dict(
        a=a,
        atilde=atilde,
        sigma2=sigma2,
        sigma2_all=sigma2_all,
        refl=refl,
        logdetC_f64lev=logdetC_f64lev,
        logdetC_pr=logdetC_pr,
        levinson_resid=levinson_resid,
    )


def dense_factors(acf):
    """Dense reference factors of C = toeplitz(acf) in float64.

    Returns dict(L, Linv, Cinv, cond, C):
      L    lower Cholesky factor
      Linv L^{-1} by triangular solve against the identity
      Cinv C^{-1} by cho_solve against the identity
      cond 2-norm condition number from eigvalsh (C is SPD)
      C    the dense matrix itself (extra key, convenient for callers)
    """
    acf = np.asarray(acf, dtype=np.float64)
    N = acf.shape[0]
    C = scipy.linalg.toeplitz(acf)
    L = scipy.linalg.cholesky(C, lower=True)
    eye = np.eye(N)
    Linv = scipy.linalg.solve_triangular(L, eye, lower=True)
    Cinv = scipy.linalg.cho_solve((L, True), eye)
    ev = scipy.linalg.eigvalsh(C)
    cond = float(ev[-1] / ev[0])
    return dict(L=L, Linv=Linv, Cinv=Cinv, cond=cond, C=C)


def spectra(a, atilde, nfft, dtype=np.float64):
    """Filter spectra ah = rfft(a, nfft), bh = rfft(atilde, nfft).

    `dtype` is the REAL dtype the filters are cast to before the transform;
    the transform runs in that precision (numpy >= 2.0 keeps float32 input in
    single precision and returns complex64), so dtype=np.float32 gives the
    spectra a float32 production path (or the gs_pr_ascoded variant's
    in-model rfft) would compute, while the default float64 gives complex128
    spectra that the caller may cast down (variants.build_consts
    spectra_from='f64', the kit's default).
    """
    a = np.asarray(a, dtype=dtype)
    atilde = np.asarray(atilde, dtype=dtype)
    ah = np.fft.rfft(a, n=nfft)
    bh = np.fft.rfft(atilde, n=nfft)
    if ah.dtype != np.result_type(dtype, np.complex64):
        # older numpy computes in double regardless of the input dtype
        ah = ah.astype(np.result_type(dtype, np.complex64))
        bh = bh.astype(np.result_type(dtype, np.complex64))
    return ah, bh


def nfft_for(N, mode):
    """FFT length for linear convolution of two length-N sequences.

    'pow2': smallest power of two >= 2N-1. This is what PR #141's
            next_fast_len(n_time + P - 1) with P = N returns
            (1 << (2N-2).bit_length()).
    'fast': scipy.fft.next_fast_len(2N-1, real=True), the minimal
            5-smooth length at or above the bound (5.2).
    """
    target = 2 * N - 1
    if mode == "pow2":
        return 1 << (int(target) - 1).bit_length()
    if mode == "fast":
        return int(scipy.fft.next_fast_len(target, real=True))
    raise ValueError(f"unknown nfft mode {mode!r}")


def lower_toeplitz(x):
    """Dense L(x): lower-triangular Toeplitz with first column x, eq. (4.1)."""
    x = np.asarray(x)
    return scipy.linalg.toeplitz(x, np.concatenate([[x[0]], np.zeros(x.shape[0] - 1, dtype=x.dtype)]))


def np_dense_gs_cinv(a, atilde, sigma2):
    """Dense C^{-1} from the GS formula (4.4), for diagnostics.

    Isolates GS-formula error (Levinson roundoff) from FFT roundoff.
    """
    La = lower_toeplitz(a)
    Lb = lower_toeplitz(atilde)
    return (La @ La.T - Lb @ Lb.T) / sigma2


# =============================================================================
# JAX appliers
# =============================================================================
#
# Conventions shared by all appliers:
#   M       (N, k) real array, dtype float32 or float64
#   ah, bh  (nfft//2 + 1,) complex spectra of a and atilde, already cast by
#           the caller (complex64 when M is float32, complex128 when float64)
#   nfft    python int (static), >= 2N-1
#   sigma2  scalar; a python float or a 0-d array of M's dtype
# The output dtype follows M. Nothing here casts.


def _pr_apply_column(v, ah, bh, nfft, sigma2):
    """Transcription of PR #141 apply_cinv_gs_fast for one column v (N,).

    C^{-1} v = [A(A^T v) - B(B^T v)] / sigma2, A = L(a), B = L(atilde);
    A^T v = flip(A flip(v)) per (5.3). Identical op sequence to the PR
    except that the PR divides by sigma**2 and we receive sigma2 directly.
    """
    N = v.shape[0]

    def apply(filt, x):
        # apply_matrix_fft_precomputed: rfft, multiply, irfft, truncate
        return jnp.fft.irfft(jnp.fft.rfft(x, n=nfft) * filt, n=nfft)[:N]

    def apply_t(filt, x):
        return jnp.flip(apply(filt, jnp.flip(x)))

    t1 = apply(ah, apply_t(ah, v))
    t2 = apply(bh, apply_t(bh, v))
    return (t1 - t2) / sigma2


def gs_pr_cinv(M, ah, bh, nfft, sigma2):
    """C^{-1} M as PR #141 codes it: vmap of the per-column routine over
    the k columns (in_axes=1, out_axes=1). Returns (N, k)."""
    f = lambda v: _pr_apply_column(v, ah, bh, nfft, sigma2)  # noqa: E731
    return jax.vmap(f, in_axes=1, out_axes=1)(M)


def gs_half_grams_T(M, ah, bh, nfft):
    """Batched half application, time axis last.

    Returns T of shape (2, k, N) with T[0] = (L(a)^T M)^T and
    T[1] = (L(atilde)^T M)^T. The transpose L(x)^T is a cross-correlation,
    computed as irfft(conj(xh) * rfft(M^T)); with nfft >= 2N-1 the circular
    wrap-around lands only on zero-padded filter entries (same bound as for
    the convolution, see the module docstring). One rfft, one batched irfft.
    """
    N = M.shape[0]
    F = jnp.fft.rfft(M.T, n=nfft, axis=-1)  # (k, nh)
    spec = jnp.stack([ah, bh])  # (2, nh)
    T = jnp.fft.irfft(jnp.conj(spec)[:, None, :] * F[None, :, :], n=nfft, axis=-1)
    return T[..., :N]


def gs_full_cinv(M, ah, bh, nfft, sigma2):
    """C^{-1} M by batched GS (variant V4 of the plan). Returns (N, k).

    Four batched FFT passes: rfft(M^T); irfft with conjugate spectra
    (the two transposed factors, truncated to N); rfft of those; irfft with
    the plain spectra (the two forward factors, truncated to N). The
    truncation between the two rounds is load-bearing and is not fused.
    """
    N = M.shape[0]
    spec = jnp.stack([ah, bh])  # (2, nh)
    T = gs_half_grams_T(M, ah, bh, nfft)  # (2, k, N)
    U = jnp.fft.irfft(spec[:, None, :] * jnp.fft.rfft(T, n=nfft, axis=-1), n=nfft, axis=-1)
    U = U[..., :N]
    CiM_T = (U[0] - U[1]) / sigma2  # (k, N)
    return CiM_T.T


def gs_half_grams(M, ah, bh, nfft, sigma2):
    """Half-Gram factors (variant V6): P = L(a)^T M, R = L(atilde)^T M,
    each (N, k). The caller forms M^T C^{-1} M = (P^T P - R^T R) / sigma2.
    `sigma2` is accepted for signature uniformity and not used here."""
    T = gs_half_grams_T(M, ah, bh, nfft)  # (2, k, N)
    return T[0].T, T[1].T


# =============================================================================
# numpy twins (same signatures; numpy.fft; used in tests / ref / diagnostics)
# =============================================================================


def _np_pr_apply_column(v, ah, bh, nfft, sigma2):
    N = v.shape[0]

    def apply(filt, x):
        return np.fft.irfft(np.fft.rfft(x, n=nfft) * filt, n=nfft)[:N]

    def apply_t(filt, x):
        return np.flip(apply(filt, np.flip(x)))

    t1 = apply(ah, apply_t(ah, v))
    t2 = apply(bh, apply_t(bh, v))
    return (t1 - t2) / sigma2


def np_gs_pr_cinv(M, ah, bh, nfft, sigma2):
    """numpy twin of gs_pr_cinv (explicit loop over columns)."""
    M = np.asarray(M)
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        out[:, j] = _np_pr_apply_column(M[:, j], ah, bh, nfft, sigma2)
    return out


def np_gs_half_grams_T(M, ah, bh, nfft):
    N = M.shape[0]
    F = np.fft.rfft(np.asarray(M).T, n=nfft, axis=-1)
    spec = np.stack([ah, bh])
    T = np.fft.irfft(np.conj(spec)[:, None, :] * F[None, :, :], n=nfft, axis=-1)
    return T[..., :N]


def np_gs_full_terms(M, ah, bh, nfft):
    """The two GS terms separately, each (N, k): L(a) L(a)^T M and
    L(atilde) L(atilde)^T M (not divided by sigma2). For the cancellation
    ratio diagnostic ||term1|| / ||term1 - term2||."""
    N = M.shape[0]
    spec = np.stack([ah, bh])
    T = np_gs_half_grams_T(M, ah, bh, nfft)
    U = np.fft.irfft(spec[:, None, :] * np.fft.rfft(T, n=nfft, axis=-1), n=nfft, axis=-1)
    U = U[..., :N]
    return U[0].T, U[1].T


def np_gs_full_cinv(M, ah, bh, nfft, sigma2):
    """numpy twin of gs_full_cinv."""
    U0, U1 = np_gs_full_terms(M, ah, bh, nfft)
    return (U0 - U1) / sigma2


def np_gs_half_grams(M, ah, bh, nfft, sigma2):
    """numpy twin of gs_half_grams; returns (P, R) each (N, k)."""
    T = np_gs_half_grams_T(M, ah, bh, nfft)
    return T[0].T, T[1].T
