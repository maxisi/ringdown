"""Extended-precision (numpy longdouble) reference for the GS benchmark.

Numpy only. On x86-64 Linux longdouble is the 80-bit x87 format
(eps ~ 1.1e-19), which is enough to make the reference's own error
negligible against the float64 quantities it is used to judge.

Contents:
  refine_solve        mixed-precision iterative refinement of C^{-1} B
  levinson_ld         longdouble Levinson-Durbin: the Yule-Walker filter a,
                      the innovation variances sigma_m^2 and the reflection
                      coefficients (note eq. 3.9); a cast to float64 is the
                      refined filter the GS variants can be fed
  logdet_levinson_ld  exact log|C| = sum_m log sigma_m^2 by that recursion
                      (note eq. 3.10); weakly stable, so its own error grows
                      with cond (2e-11 nats at cond 5e8, 2e-8 at 3.5e11)
  logdet_cholesky_ld  log|C| = 2 sum log diag L from a longdouble dense
                      Cholesky (O(N^3), ~1 s at N=1024, tens of s at N=4096):
                      the reference's log-det
  oneshot_core_ld     one-shot marginalized log-likelihood (note eq. 6.2) and
                      its closed-form gradient wrt each design matrix M_i
  fd_selftest         central finite-difference check of that gradient on a
                      tiny problem

The reference is independent of both routes under test: it never uses the
GS formula and it never trusts a float64 Cholesky solve as more than a
preconditioner.
"""

import numpy as np
import scipy.linalg

LD = np.longdouble
EPS_LD = float(np.finfo(LD).eps)


# =============================================================================
# C^{-1} B by iterative refinement
# =============================================================================


def refine_solve(L, C, B, rounds=2):
    """C^{-1} B in longdouble by mixed-precision iterative refinement.

    L: float64 lower Cholesky factor of C (the preconditioner).
    C: float64 (N, N) dense SPD matrix (its entries are exact in longdouble).
    B: (N,) or (N, m) right-hand side in float64 or longdouble. A longdouble
       B is honoured exactly in the residual, which matters for finite
       differences with steps below float64 resolution.

    Each round: r = B - C X in longdouble, X += cho_solve(r) in float64.
    With cond(C) eps64 < 1 the forward error shrinks by ~cond(C) eps64 per
    round and the longdouble residual ||B - C X|| / (||C|| ||X||) settles
    at the eps_ld level (that is what tests_kernels.py asserts). Returns a
    longdouble array of B's shape.
    """
    B = np.asarray(B)
    squeeze = B.ndim == 1
    if squeeze:
        B = B[:, None]
    B_ld = B.astype(LD)
    C_ld = np.asarray(C).astype(LD)
    X = scipy.linalg.cho_solve((L, True), B.astype(np.float64)).astype(LD)
    for _ in range(rounds):
        res = B_ld - C_ld @ X
        dX = scipy.linalg.cho_solve((L, True), res.astype(np.float64))
        X = X + dX.astype(LD)
    return X[:, 0] if squeeze else X


def ld_residual(C, B, X):
    """Relative residual ||B - C X||_F / (||C||_F ||X||_F) in longdouble."""
    C_ld = np.asarray(C).astype(LD)
    B_ld = np.asarray(B).astype(LD)
    X = np.asarray(X).astype(LD)
    r = B_ld - C_ld @ X
    return float(np.sqrt(np.sum(r * r)) / (np.sqrt(np.sum(C_ld * C_ld)) * np.sqrt(np.sum(X * X))))


# =============================================================================
# exact log-determinant by longdouble Levinson-Durbin
# =============================================================================


def levinson_ld(acf):
    """Levinson-Durbin recursion (note eq. 3.9) in longdouble.

    Returns (a, sigma2_all, refl) as longdouble arrays: the order-(N-1)
    prediction-error filter, the innovation variances sigma_m^2 for
    m = 0..N-1, and the reflection coefficients. O(N^2) with vectorized
    inner products. Deliberately a separate copy from gs_kernels.levinson_durbin
    so the reference does not import the module under test.
    """
    g = np.asarray(acf, dtype=LD)
    N = g.shape[0]
    a = np.zeros(N, dtype=LD)
    a[0] = 1
    sigma2_all = np.empty(N, dtype=LD)
    refl = np.empty(max(N - 1, 0), dtype=LD)
    s = g[0]
    sigma2_all[0] = s
    for m in range(1, N):
        kappa = -(g[m] + np.dot(a[1:m], g[m - 1:0:-1])) / s
        a[: m + 1] = a[: m + 1] + kappa * a[m::-1]
        s = s * (1 - kappa * kappa)
        sigma2_all[m] = s
        refl[m - 1] = kappa
    return a, sigma2_all, refl


def logdet_levinson_ld(acf):
    """log|C| = sum_m log sigma_m^2 as a longdouble scalar (note eq. 3.10)."""
    _, sigma2_all, _ = levinson_ld(acf)
    return np.sum(np.log(sigma2_all))


def logdet_cholesky_ld(C, blk=64):
    """log|C| = 2 sum_j log L_jj from a right-looking longdouble Cholesky.

    C is a dense SPD matrix (float64 entries are exact in longdouble).  Only
    the lower triangle is updated; the factor itself is not returned.  The
    backward error of a Cholesky log-determinant is ~ N eps_ld in log|C|
    (no cond amplification), which makes this the reference's log-det; the
    Levinson recursion above is kept as a cross-check.
    """
    A = np.array(C, dtype=LD, copy=True)
    N = A.shape[0]
    logdet = LD(0)          # sum_j log(L_jj^2) = 2 sum_j log L_jj
    for j in range(N):
        d = A[j, j]
        if not d > 0:
            raise np.linalg.LinAlgError("longdouble Cholesky: matrix not positive definite")
        logdet = logdet + np.log(d)
        if j + 1 < N:
            col = A[j + 1:, j] / np.sqrt(d)
            # trailing update A22 -= col col^T, lower triangle only, in row
            # blocks of `blk`: N^3/6 longdouble multiplies (plus a small
            # triangular waste per block) with N^2 / (2 blk) numpy calls
            sub = A[j + 1:, j + 1:]
            n = col.shape[0]
            for i0 in range(0, n, blk):
                i1 = min(n, i0 + blk)
                sub[i0:i1, :i1] -= np.multiply.outer(col[i0:i1], col[:i1])
    return logdet


# =============================================================================
# tiny longdouble dense linear algebra (k <= 32)
# =============================================================================


def chol_ld(A):
    """Lower Cholesky factor of a small SPD longdouble matrix."""
    A = np.asarray(A, dtype=LD)
    k = A.shape[0]
    R = np.zeros((k, k), dtype=LD)
    for j in range(k):
        d = A[j, j] - np.dot(R[j, :j], R[j, :j])
        if not d > 0:
            raise np.linalg.LinAlgError("longdouble Cholesky: matrix not positive definite")
        R[j, j] = np.sqrt(d)
        for i in range(j + 1, k):
            R[i, j] = (A[i, j] - np.dot(R[i, :j], R[j, :j])) / R[j, j]
    return R


def solve_lower_ld(R, b):
    """Solve R x = b for lower-triangular R (longdouble). b may be (k,) or (k, m)."""
    R = np.asarray(R, dtype=LD)
    b = np.asarray(b, dtype=LD)
    k = R.shape[0]
    x = np.zeros_like(b)
    for i in range(k):
        x[i] = (b[i] - np.tensordot(R[i, :i], x[:i], axes=(0, 0))) / R[i, i]
    return x


def solve_upper_ld(U, b):
    """Solve U x = b for upper-triangular U (longdouble). b may be (k,) or (k, m)."""
    U = np.asarray(U, dtype=LD)
    b = np.asarray(b, dtype=LD)
    k = U.shape[0]
    x = np.zeros_like(b)
    for i in range(k - 1, -1, -1):
        x[i] = (b[i] - np.tensordot(U[i, i + 1:], x[i + 1:], axes=(0, 0))) / U[i, i]
    return x


# =============================================================================
# one-shot marginalized likelihood and gradient
# =============================================================================


def oneshot_core_ld(M_list, y_list, L_list, C_list, logdet_list, rounds=2):
    """One-shot marginalized log-likelihood (note eq. 6.2) in longdouble.

    Inputs (lists over detectors):
      M_list      (N_i, k) design matrices, float64 or longdouble
      y_list      (N_i,)   strains
      L_list      (N_i, N_i) float64 lower Cholesky factors of C_i
      C_list      (N_i, N_i) float64 dense covariances
      logdet_list scalars log|C_i| (e.g. from logdet_levinson_ld)

    Algebra (all k x k work in longdouble):
      A_inv = I + sum_i M_i^T C_i^{-1} M_i,   v = sum_i M_i^T C_i^{-1} y_i,
      Q = sum_i y_i^T C_i^{-1} y_i,   A_inv = R R^T,   u = R^{-1} v,
      logL = -Q/2 + u^T u/2 - sum_i log|C_i|/2 - sum_j log R_jj.

    Gradient (closed form, alpha = A v with A = A_inv^{-1}):
      dlogL/dM_i = C_i^{-1}(y_i - M_i alpha) alpha^T - C_i^{-1} M_i A
                 = (w_i - CiM_i alpha) alpha^T - CiM_i A,
    with w_i = C_i^{-1} y_i and CiM_i = C_i^{-1} M_i from refine_solve, so
    no further solves are needed.

    Returns dict(loglike, dlogL_dM, A_inv, v, Q, alpha, A, CiM, w).
    """
    n_det = len(M_list)
    k = np.asarray(M_list[0]).shape[1]
    A_inv = np.eye(k, dtype=LD)
    v = np.zeros(k, dtype=LD)
    Q = LD(0)
    CiM_list, w_list, M_ld_list = [], [], []
    for i in range(n_det):
        M = np.asarray(M_list[i])
        M_ld = M.astype(LD)
        y = np.asarray(y_list[i])
        CiM = refine_solve(L_list[i], C_list[i], M, rounds=rounds)  # (N, k) ld
        w = refine_solve(L_list[i], C_list[i], y, rounds=rounds)  # (N,) ld
        S = M_ld.T @ CiM
        S = 0.5 * (S + S.T)
        A_inv = A_inv + S
        v = v + M_ld.T @ w
        Q = Q + np.dot(y.astype(LD), w)
        CiM_list.append(CiM)
        w_list.append(w)
        M_ld_list.append(M_ld)

    R = chol_ld(A_inv)
    u = solve_lower_ld(R, v)
    logdet_sum = np.sum(np.asarray(logdet_list, dtype=LD))
    loglike = -0.5 * Q + 0.5 * np.dot(u, u) - 0.5 * logdet_sum - np.sum(np.log(np.diag(R)))

    # alpha = A v = R^{-T} u ; A = R^{-T} R^{-1}
    alpha = solve_upper_ld(R.T, u)
    Rinv = solve_lower_ld(R, np.eye(k, dtype=LD))
    A = Rinv.T @ Rinv
    dlogL_dM = []
    for i in range(n_det):
        CiM = CiM_list[i]
        g = np.outer(w_list[i] - CiM @ alpha, alpha) - CiM @ A
        dlogL_dM.append(g)

    return dict(
        loglike=loglike,
        dlogL_dM=dlogL_dM,
        A_inv=A_inv,
        v=v,
        Q=Q,
        alpha=alpha,
        A=A,
        CiM=CiM_list,
        w=w_list,
    )


# =============================================================================
# finite-difference self-test
# =============================================================================


def _tiny_problem(N=16, k=4, n_det=2, seed=3):
    """A small well-conditioned problem: damped-cosine ACF, random M and y."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 2048.0
    lags = np.arange(N) * dt
    M_list, y_list, L_list, C_list, logdet_list = [], [], [], [], []
    for i in range(n_det):
        acf = np.exp(-lags / 0.01) * np.cos(2 * np.pi * (120 + 10 * i) * lags) + 1e-2 * (lags == 0)
        C = scipy.linalg.toeplitz(acf)
        L = np.linalg.cholesky(C)
        M_list.append(rng.normal(size=(N, k)))
        y_list.append(L @ rng.normal(size=N) + M_list[-1] @ rng.normal(size=k))
        L_list.append(L)
        C_list.append(C)
        logdet_list.append(logdet_levinson_ld(acf))
    return M_list, y_list, L_list, C_list, logdet_list


def fd_selftest(N=16, k=4, n_det=2, h=1e-6, rounds=3, verbose=True):
    """Compare oneshot_core_ld's closed-form gradient with longdouble central
    finite differences of its own loglike. Returns the worst relative error
    (max abs error over max abs gradient entry)."""
    M_list, y_list, L_list, C_list, logdet_list = _tiny_problem(N, k, n_det)
    base = oneshot_core_ld(M_list, y_list, L_list, C_list, logdet_list, rounds=rounds)
    h = LD(h)
    worst = 0.0
    gmax = max(float(np.max(np.abs(g))) for g in base["dlogL_dM"])
    for i in range(n_det):
        fd = np.zeros_like(base["dlogL_dM"][i])
        for r in range(N):
            for c in range(k):
                Mp = [np.asarray(m).astype(LD) for m in M_list]
                Mm = [np.asarray(m).astype(LD) for m in M_list]
                Mp[i][r, c] += h
                Mm[i][r, c] -= h
                fp = oneshot_core_ld(Mp, y_list, L_list, C_list, logdet_list, rounds=rounds)["loglike"]
                fm = oneshot_core_ld(Mm, y_list, L_list, C_list, logdet_list, rounds=rounds)["loglike"]
                fd[r, c] = (fp - fm) / (2 * h)
        err = float(np.max(np.abs(fd - base["dlogL_dM"][i]))) / gmax
        worst = max(worst, err)
        if verbose:
            print(f"  fd_selftest det {i}: max|fd - closed| / max|grad| = {err:.3e}")
    return worst


if __name__ == "__main__":
    w = fd_selftest()
    print(f"fd_selftest worst relative error {w:.3e} (target 1e-9)")
    assert w < 1e-9
