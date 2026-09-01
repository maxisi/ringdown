#!/usr/bin/env python
"""Self-contained H100-vs-A6000-vs-CPU benchmark for the NumPyro ringdown model.

Runs the FULL grid in BOTH precisions on the GPU (float64 and float32 are both
production GPU modes: `ringdown/cli/ringdown_fit.py` maps a `float32` flag onto
`jax_enable_x64=False`), plus the float64 CPU baseline on the same node.

Methodology is the one that survived independent verification
(CLAIMS_VERIFICATION.md):

  * device-time harness: R gradients inside ONE jit'd fori_loop with a
    numerically inert 1e-30 feedback so XLA cannot CSE or hoist the body;
    timed at two loop counts, slope taken, medians of >=5.  That is exactly
    what NUTS pays per leapfrog step inside its own scan.
  * correctness first: every prototype checked against the *unmodified*
    ringdown.model.make_model on the potential AND every gradient component,
    tolerance 1e-11 (the verifier showed 1e-15 is only reachable with an
    artificially well-conditioned covariance; ~1e-12 is the realistic figure).
  * precision and platform are process-global in JAX, so each leg runs in its
    own subprocess with JAX_PLATFORMS / jax_enable_x64 set BEFORE `import jax`.
    The verifier documented that setting JAX_PLATFORMS after the import fails
    silently and yields GPU numbers labeled "CPU".

Legs
----
  main    GPU float64  env, correctness, devtime, rhs sweep, compile, chains,
                       and the float64 reference dump for the f32 accuracy test
  sub 1   GPU float32  f32 accuracy vs the dump, devtime, rhs, compile, chains
  sub 2   CPU float64  env, devtime          (all allocated cores, ONE CPU device)
  sub 3   CPU float64  env, chains           (set_host_device_count(4), i.e. the
                       production CPU configuration -- kept separate so the
                       4-way device split cannot perturb the devtime numbers)
  sub 4   CPU float64  devtime at the production point only, OMP_NUM_THREADS=1
                       (what ringdown_fit.py sets in production)

Usage
-----
    python bench.py --out results.json          # full run, ~25 min on one H100
    python bench.py --smoke --out smoke.json    # ~6 min, reduced reps

No repository file is written; `ringdown` is only imported.
"""

# ---------------------------------------------------------------------------
# PHASE 0: everything that must happen BEFORE `import jax`
# ---------------------------------------------------------------------------
import argparse
import os
import sys

_P = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_P.add_argument("--platform", default="gpu", choices=["gpu", "cpu"],
                help="backend for THIS process (default gpu)")
_P.add_argument("--x64", type=int, default=1, choices=[0, 1],
                help="jax_enable_x64: 1 = float64 (default), 0 = float32")
_P.add_argument("--smoke", action="store_true",
                help="reduced repetitions / configs, for validating the kit")
_P.add_argument("--out", default="results.json", help="output JSON path")
_P.add_argument("--sections", default="all",
                help="comma-separated subset of: env,correctness,devtime,rhs,"
                     "compile,chains,f32ref,f32acc")
_P.add_argument("--no-sub", action="store_true",
                help="do not spawn the CPU / float32 legs (set on the legs)")
_P.add_argument("--ref", default="",
                help="path to the float64 reference .npz (f32 accuracy leg)")
_P.add_argument("--deadline", type=float, default=2000.0,
                help="soft wall-clock budget in s for the WHOLE run (this "
                     "process plus its child legs); sections are skipped once "
                     "the share for the current process is exceeded")
_P.add_argument("--configs", default="",
                help="restrict the size sweep, e.g. '2,205,2;3,1024,8'")
_P.add_argument("--omp", default="",
                help="set OMP/MKL/OPENBLAS_NUM_THREADS before jax import")
_P.add_argument("--tag", default="",
                help="free-form label recorded in the JSON")
ARGS = _P.parse_args()

if ARGS.omp:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NPROC"):
        os.environ[_v] = ARGS.omp

if ARGS.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"          # hard pin, before import jax
else:
    # 'cuda' makes JAX raise instead of silently falling back to CPU when the
    # plugin fails to initialize -- the failure mode we must never mistake for
    # a measurement.
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json          # noqa: E402
import platform      # noqa: E402
import re            # noqa: E402
import socket        # noqa: E402
import statistics as stats  # noqa: E402
import subprocess    # noqa: E402
import time          # noqa: E402
import traceback     # noqa: E402
import warnings      # noqa: E402

T_START = time.time()
LEG = "%s-f%d" % (ARGS.platform, 64 if ARGS.x64 else 32)


def _die(msg, code=2):
    print("\n" + "=" * 78, file=sys.stderr)
    print("FATAL: " + msg, file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    sys.stderr.flush()
    sys.exit(code)


def _diagnostics():
    """Everything a human needs to debug a failed CUDA init."""
    out = ["python      : %s" % sys.executable,
           "version     : %s" % sys.version.replace("\n", " "),
           "hostname    : %s" % socket.gethostname(),
           "JAX_PLATFORMS = %r" % os.environ.get("JAX_PLATFORMS"),
           "CUDA_VISIBLE_DEVICES = %r" % os.environ.get("CUDA_VISIBLE_DEVICES")]
    for k in sorted(os.environ):
        if k.startswith("SLURM_"):
            out.append("%-26s = %s" % (k, os.environ[k]))
    for cmd in (["nvidia-smi"],
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
                 "compute_cap", "--format=csv"]):
        try:
            out.append("$ %s\n%s" % (" ".join(cmd),
                                     subprocess.run(cmd, capture_output=True,
                                                    text=True, timeout=60).stdout))
        except Exception as e:                     # pragma: no cover
            out.append("$ %s -> %r" % (" ".join(cmd), e))
    try:
        import importlib.metadata as md
        for p in ("jax", "jaxlib", "numpyro", "numpy", "scipy",
                  "jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cublas-cu12",
                  "nvidia-cusolver-cu12", "nvidia-cuda-runtime-cu12"):
            try:
                out.append("pkg %-26s %s" % (p, md.version(p)))
            except Exception:
                out.append("pkg %-26s MISSING" % p)
    except Exception as e:                         # pragma: no cover
        out.append("importlib.metadata failed: %r" % e)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# PHASE 1: import jax and verify we are on the requested backend AND precision
# ---------------------------------------------------------------------------
try:
    from jax import config as _jc
    _jc.update("jax_enable_x64", bool(ARGS.x64))
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import initialize_model, potential_energy
    from scipy.linalg import toeplitz
    import ringdown.model as rdm
    from ringdown import qnms
    from ringdown.model import chi_factors, rd_design_matrix
except Exception:
    print(_diagnostics(), file=sys.stderr)
    traceback.print_exc()
    _die("import failed (see diagnostics above)")

# Mirror ringdown/cli/ringdown_fit.py:104-122, which sets device_count=4 on CPU
# and calls numpyro.set_host_device_count(4).  Without it the shipped
# `num_chains=4, chain_method='parallel'` degrades to SEQUENTIAL on CPU too and
# the CPU baseline would not be the production baseline.  Applied ONLY to the
# chain-scaling leg: splitting XLA:CPU into 4 logical devices could perturb the
# per-gradient timings, which the A6000 study took on a single CPU device.
# Must run before jax.devices() initializes the backend.
if ARGS.platform == "cpu" and "chains" in ARGS.sections and \
        "devtime" not in ARGS.sections:
    try:
        _nd = 4
        try:
            _nd = min(4, len(os.sched_getaffinity(0)))
        except Exception:
            pass
        numpyro.set_host_device_count(_nd)
    except Exception as _e:                        # pragma: no cover
        print("warning: set_host_device_count failed: %r" % (_e,))

_DEVS = jax.devices()
_BACKEND = jax.default_backend()
_IS_GPU = any(getattr(d, "platform", "") in ("gpu", "cuda") for d in _DEVS)
if ARGS.platform == "gpu" and not _IS_GPU:
    print(_diagnostics(), file=sys.stderr)
    _die("--platform gpu requested but jax.devices() = %r (backend %r). The "
         "CUDA plugin did not initialize." % (_DEVS, _BACKEND))
if ARGS.platform == "cpu" and _IS_GPU:
    print(_diagnostics(), file=sys.stderr)
    _die("--platform cpu requested but a GPU device is visible: %r. "
         "JAX_PLATFORMS was not honored; refusing to mislabel GPU numbers as "
         "CPU numbers." % (_DEVS,))
if bool(jax.config.jax_enable_x64) != bool(ARGS.x64):
    _die("jax_enable_x64 is %r but --x64 %d was requested"
         % (jax.config.jax_enable_x64, ARGS.x64))

DT = np.float64 if ARGS.x64 else np.float32
ALL_SECTIONS = {"env", "correctness", "devtime", "rhs", "compile", "chains",
                "f32ref", "f32acc"}
SECTIONS = (ALL_SECTIONS if ARGS.sections == "all"
            else set(s.strip() for s in ARGS.sections.split(",")))


# The top-level GPU-float64 process gets ~55% of the total budget; the rest is
# reserved for the float32 and CPU legs, which are launched afterwards.
MY_BUDGET = ARGS.deadline * (0.60 if (not ARGS.no_sub and ARGS.platform == "gpu"
                                      and ARGS.x64) else 1.0)


def want(name):
    if name not in SECTIONS:
        return False
    if time.time() - T_START > MY_BUDGET:
        print("  [deadline exceeded -- skipping section %r]" %
              name, flush=True)
        return False
    return True


# ---------------------------------------------------------------------------
# PHASE 2: synthetic data + the prototype models
#
# Data generation is the construction used in GPU_BENCHMARKS.md (exp-decay x
# cosine ACF -> Toeplitz -> Cholesky) so the numbers are directly comparable to
# the A6000 tables.  scipy.linalg.toeplitz replaces the original O(n^2) Python
# double loop; it builds the same matrix.
# ---------------------------------------------------------------------------
KW = dict(a_scale_max=1e-20, m_min=50.0,
          m_max=150.0, chi_min=0.0, chi_max=0.99)
KW_MCMC = dict(KW, a_scale_max=1.0)


def make_args(n_det=2, ntime=205, seed=0, dt=1 / 2048.0):
    rng = np.random.default_rng(seed)
    times = np.tile(np.arange(ntime) * dt, (n_det, 1))
    lags = np.arange(ntime) * dt
    acf = np.exp(-lags / 0.01) * np.cos(2 * np.pi *
                                        120 * lags) + 1e-3 * (lags == 0)
    C = toeplitz(acf) + ntime * 1e-9 * np.eye(ntime)
    L1 = np.linalg.cholesky(C)
    ls = np.stack([L1] * n_det)
    strains = np.einsum("ij,dj->di", L1, rng.normal(size=(n_det, ntime)))
    fps = rng.normal(size=n_det)
    fcs = rng.normal(size=n_det)
    return tuple(jnp.asarray(a, dtype=DT)
                 for a in (times, strains, ls, fps, fcs))


def modes_of(n):
    return [(1, -2, 2, 2, i) for i in range(n)]


DEGENERATE = [(1, -2, 2, 2, 0), (1, -2, 2, 2, 0)]


def _coeffs(modes):
    fc, gc = [], []
    for mo in modes:
        c = qnms.KerrMode(mo).coefficients
        fc.append(c[0])
        gc.append(c[1])
    return jnp.array(fc), jnp.array(gc)


def _head(modes, a_scale_max, m_min, m_max, chi_min, chi_max):
    """Shared non-linear front end: priors -> frequencies -> design matrix."""
    n_modes = len(modes)
    fco, gco = _coeffs(modes)

    def head(times, fps, fcs):
        m = numpyro.sample("m", dist.Uniform(m_min, m_max))
        chi = numpyro.sample("chi", dist.Uniform(chi_min, chi_max))
        f0 = 1 / (m * qnms.T_MSUN)
        f = f0 * chi_factors(chi, fco)
        g = f0 * chi_factors(chi, gco)
        a_scale = numpyro.sample("a_scale", dist.Uniform(0, a_scale_max),
                                 sample_shape=(n_modes,))
        return rd_design_matrix(times, f, g, fps, fcs, a_scale)
    return head


def make_current(modes, a_scale_max, m_min, m_max, chi_min, chi_max):
    """Faithful transcription of the shipped sequential marginalized branch."""
    n_modes = len(modes)
    fco, gco = _coeffs(modes)

    def model(times, strains, ls, fps, fcs):
        times, strains, ls, fps, fcs = map(
            jnp.array, (times, strains, ls, fps, fcs))
        n_det = times.shape[0]
        m = numpyro.sample("m", dist.Uniform(m_min, m_max))
        chi = numpyro.sample("chi", dist.Uniform(chi_min, chi_max))
        f0 = 1 / (m * qnms.T_MSUN)
        f = f0 * chi_factors(chi, fco)
        g = f0 * chi_factors(chi, gco)
        a_scale = numpyro.sample("a_scale", dist.Uniform(0, a_scale_max),
                                 sample_shape=(n_modes,))
        dms = rd_design_matrix(times, f, g, fps, fcs, a_scale)
        k = dms.shape[2]
        mu = jnp.zeros(k)
        Lambda_inv = jnp.eye(k)
        Lambda_inv_chol = jnp.eye(k)
        for i in range(n_det):
            M, L, y = dms[i, :, :], ls[i, :, :], strains[i, :]
            A_inv = Lambda_inv + \
                jnp.dot(M.T, jsp.linalg.cho_solve((L, True), M))
            A_inv_chol = jsp.linalg.cholesky(A_inv, lower=True)
            a = jsp.linalg.cho_solve(
                (A_inv_chol, True),
                jnp.dot(Lambda_inv, mu)
                + jnp.dot(M.T, jsp.linalg.cho_solve((L, True), y)))
            r = y - jnp.dot(M, mu)
            Cinv_r = jsp.linalg.cho_solve((L, True), r)
            MAMt = jnp.dot(M, jsp.linalg.cho_solve((A_inv_chol, True),
                                                   jnp.dot(M.T, Cinv_r)))
            Cinv_MAMt = jsp.linalg.cho_solve((L, True), MAMt)
            lsd = (jnp.sum(jnp.log(jnp.diag(L)))
                   - jnp.sum(jnp.log(jnp.diag(Lambda_inv_chol)))
                   + jnp.sum(jnp.log(jnp.diag(A_inv_chol))))
            numpyro.factor("logl_%d" % i,
                           -0.5 * jnp.dot(r, Cinv_r - Cinv_MAMt) - lsd)
            mu, Lambda_inv, Lambda_inv_chol = a, A_inv, A_inv_chol
    return model


def make_whiten_seq(modes, **kw):
    """R1.7: sequential recursion, but whitened once per detector."""
    head = _head(modes, **kw)
    def st(A, B): return jsp.linalg.solve_triangular(A, B, lower=True)  # noqa: E731

    def model(times, strains, ls, fps, fcs):
        times, strains, ls, fps, fcs = map(
            jnp.array, (times, strains, ls, fps, fcs))
        n_det = times.shape[0]
        dms = head(times, fps, fcs)
        k = dms.shape[2]
        mu, Lam, Lam_chol = jnp.zeros(k), jnp.eye(k), jnp.eye(k)
        for i in range(n_det):
            L = ls[i]
            W, z = st(L, dms[i]), st(L, strains[i])
            A_inv = Lam + W.T @ W
            Ac = jsp.linalg.cholesky(A_inv, lower=True)
            rw = z - W @ mu
            t = st(Ac, W.T @ rw)
            lsd = (jnp.sum(jnp.log(jnp.diagonal(L)))
                   - jnp.sum(jnp.log(jnp.diag(Lam_chol)))
                   + jnp.sum(jnp.log(jnp.diag(Ac))))
            numpyro.factor("logl_%d" % i,
                           -0.5 * (jnp.dot(rw, rw) - jnp.dot(t, t)) - lsd)
            mu = jsp.linalg.cho_solve((Ac, True), jnp.dot(Lam, mu) + W.T @ z)
            Lam, Lam_chol = A_inv, Ac
    return model


def make_oneshot_unroll_concat(modes, **kw):
    """R1 exactly as recommended in MODEL_OPTIMIZATIONS.md: Python-unrolled
    detector loop, ONE triangular solve against the concatenated [M | y]."""
    head = _head(modes, **kw)

    def model(times, strains, ls, fps, fcs):
        times, strains, ls, fps, fcs = map(
            jnp.array, (times, strains, ls, fps, fcs))
        dms = head(times, fps, fcs)
        k, n_det = dms.shape[2], dms.shape[0]
        G, v, yy, ld = jnp.eye(k), jnp.zeros(k), 0.0, 0.0
        for i in range(n_det):
            L = ls[i]
            My = jnp.concatenate([dms[i], strains[i][:, None]], axis=1)
            Wz = jsp.linalg.solve_triangular(L, My, lower=True)
            W, z = Wz[:, :k], Wz[:, k]
            G = G + W.T @ W
            v = v + W.T @ z
            yy = yy + jnp.dot(z, z)
            ld = ld + jnp.sum(jnp.log(jnp.diagonal(L)))
        Lc = jsp.linalg.cholesky(G, lower=True)
        u = jsp.linalg.solve_triangular(Lc, v, lower=True)
        numpyro.factor("logl_total", -0.5 * yy + 0.5 * jnp.dot(u, u) - ld
                       - jnp.sum(jnp.log(jnp.diag(Lc))))
    return model


def make_oneshot_unroll_sep(modes, **kw):
    """R1 unrolled with TWO separate triangular solves (no [M|y] concat) --
    GPU_BENCHMARKS.md R2b, which avoids pushing the RHS count to k+1 and onto
    the cuBLAS trsm cliff."""
    head = _head(modes, **kw)
    def st(A, B): return jsp.linalg.solve_triangular(A, B, lower=True)  # noqa: E731

    def model(times, strains, ls, fps, fcs):
        times, strains, ls, fps, fcs = map(
            jnp.array, (times, strains, ls, fps, fcs))
        dms = head(times, fps, fcs)
        k, n_det = dms.shape[2], dms.shape[0]
        G, v, yy, ld = jnp.eye(k), jnp.zeros(k), 0.0, 0.0
        for i in range(n_det):
            L = ls[i]
            W, z = st(L, dms[i]), st(L, strains[i])
            G = G + W.T @ W
            v = v + W.T @ z
            yy = yy + jnp.dot(z, z)
            ld = ld + jnp.sum(jnp.log(jnp.diagonal(L)))
        Lc = jsp.linalg.cholesky(G, lower=True)
        u = st(Lc, v)
        numpyro.factor("logl_total", -0.5 * yy + 0.5 * jnp.dot(u, u) - ld
                       - jnp.sum(jnp.log(jnp.diag(Lc))))
    return model


def make_oneshot_vmap(modes, **kw):
    """R1 with the detector loop vmapped (batched trsm + batched potrf)."""
    head = _head(modes, **kw)

    def model(times, strains, ls, fps, fcs):
        times, strains, ls, fps, fcs = map(
            jnp.array, (times, strains, ls, fps, fcs))
        dms = head(times, fps, fcs)
        k = dms.shape[2]

        def per(L, M, y):
            W = jsp.linalg.solve_triangular(L, M, lower=True)
            z = jsp.linalg.solve_triangular(L, y, lower=True)
            return W.T @ W, W.T @ z, jnp.dot(z, z), jnp.sum(jnp.log(jnp.diagonal(L)))
        G, v, yy, ld = jax.vmap(per)(ls, dms, strains)
        A_inv = jnp.eye(k) + jnp.sum(G, axis=0)
        Lc = jsp.linalg.cholesky(A_inv, lower=True)
        u = jsp.linalg.solve_triangular(Lc, jnp.sum(v, axis=0), lower=True)
        numpyro.factor("logl_total", -0.5 * jnp.sum(yy) + 0.5 * jnp.dot(u, u)
                       - jnp.sum(ld) - jnp.sum(jnp.log(jnp.diag(Lc))))
    return model


def make_prioronly(modes, **kw):
    """No likelihood at all: the fixed-cost / kernel-launch floor."""
    head = _head(modes, **kw)

    def model(times, strains, ls, fps, fcs):
        dms = head(jnp.array(times), fps, fcs)
        numpyro.factor("logl_total", jnp.sum(dms[0, 0, :]))
    return model


VARIANTS = [
    ("current", make_current),
    ("whiten_seq", make_whiten_seq),
    ("R1_unroll_concat", make_oneshot_unroll_concat),
    ("R1_unroll_sep", make_oneshot_unroll_sep),
    ("R1_vmap", make_oneshot_vmap),
    ("floor_no_likelihood", make_prioronly),
]
VMAP = dict(VARIANTS)
CHAIN_VARIANTS = ["current", "R1_unroll_sep", "R1_vmap"]

# The size grid.  (2,1024,2) was dropped from the A6000 sweep to make room for
# the second precision -- see README, "What was trimmed and why".
CONFIGS = [(2, 205, 2), (2, 1024, 4), (3, 1024, 8)]
PROD = (2, 205, 2)
if ARGS.smoke:
    CONFIGS = [(2, 205, 2), (2, 1024, 4)]
if ARGS.configs:
    CONFIGS = [tuple(int(x) for x in c.split(","))
               for c in ARGS.configs.split(";") if c.strip()]


# ---------------------------------------------------------------------------
# PHASE 3: timing harnesses
# ---------------------------------------------------------------------------
def _grad_of(model, args):
    return jax.jit(jax.grad(lambda q: potential_energy(model, args, {}, q)))


def _looped(gradfn, p0, R):
    @jax.jit
    def go(p):
        def body(i, q):
            g = gradfn(q)
            return jax.tree.map(lambda a, b: a + 1e-30 * b, p0, g)
        return jax.lax.fori_loop(0, R, body, p)
    return go


def _tmed(f, p, rep):
    jax.block_until_ready(f(p))                  # compile + warm
    T = []
    for _ in range(rep):
        t0 = time.perf_counter()
        jax.block_until_ready(f(p))
        T.append(time.perf_counter() - t0)
    return stats.median(T)


def device_us_per_grad(gradfn, p, target_s=0.15, rep=5, rmin=8, rmax=400):
    """Per-gradient DEVICE time in us (slope over two fori_loop counts)."""
    jax.block_until_ready(gradfn(p))
    t0 = time.perf_counter()
    for _ in range(3):
        jax.block_until_ready(gradfn(p))
    # crude; includes host dispatch
    t_call = (time.perf_counter() - t0) / 3.0
    R1 = int(min(rmax, max(rmin, target_s / max(t_call, 1e-6))))
    R2 = 3 * R1
    t1 = _tmed(_looped(gradfn, p, R1), p, rep)
    t2 = _tmed(_looped(gradfn, p, R2), p, rep)
    return (t2 - t1) / (R2 - R1) * 1e6, R1, R2


SHORT = {"__cublas$triangularSolve": "trsm", "cusolver_potrf_ffi": "potrf",
         "lapack_dtrsm_ffi": "trsm", "lapack_dpotrf_ffi": "potrf",
         "lapack_strsm_ffi": "trsm", "lapack_spotrf_ffi": "potrf"}


def count_custom_calls(gradfn, p):
    """Post-optimization HLO custom-call census (the structural evidence)."""
    try:
        txt = gradfn.lower(p).compile().as_text()
    except Exception:
        return {}
    keys = ("__cublas$triangularSolve", "cusolver_potrf_ffi",
            "lapack_dtrsm_ffi", "lapack_dpotrf_ffi",
            "lapack_strsm_ffi", "lapack_spotrf_ffi")
    out = {}
    for k in keys:                       # several keys share a short name
        n = txt.count('custom_call_target="%s"' % k)
        sk = SHORT.get(k, k)
        out[sk] = out.get(sk, 0) + n
    out["hlo_lines"] = txt.count("\n") + 1
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# SECTION: environment / contention
# ---------------------------------------------------------------------------
def gpu_contention():
    """Who else is on this card?  The A6000 reference numbers were taken on a
    verified-idle card; any comparison must know whether this one was."""
    info = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,"
                            "used_gpu_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        procs = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        info["compute_apps"] = procs
        info["n_foreign_compute_apps"] = sum(
            1 for ln in procs if str(os.getpid()) != ln.split(",")[0].strip())
    except Exception as e:
        info["compute_apps"] = "unavailable: %r" % e
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,"
                            "clocks.max.sm,temperature.gpu,power.draw,memory.used",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        info["gpu_state"] = r.stdout.strip()
    except Exception:
        pass
    try:
        info["loadavg"] = open("/proc/loadavg").read().strip()
    except Exception:
        pass
    return info


def section_env():
    import importlib.metadata as md
    env = {
        "leg": LEG,
        "hostname": socket.gethostname(),
        "platform_requested": ARGS.platform,
        "jax_backend": _BACKEND,
        "jax_devices": [str(d) for d in _DEVS],
        "jax_device_kinds": [getattr(d, "device_kind", "?") for d in _DEVS],
        "jax_device_count": jax.device_count(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "dtype": str(np.dtype(DT)),
        "python": sys.version.split()[0],
        "uname": platform.platform(),
        "smoke": ARGS.smoke,
        "configs": [list(c) for c in CONFIGS],
    }
    try:
        cpu = [ln.split(":", 1)[1].strip()
               for ln in open("/proc/cpuinfo") if ln.startswith("model name")]
        env["cpu_model"] = cpu[0] if cpu else "?"
        env["cpu_logical_count"] = len(cpu)
    except Exception:
        env["cpu_model"] = "?"
    try:
        env["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    env["thread_env"] = {k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                          "OPENBLAS_NUM_THREADS", "XLA_FLAGS", "JAX_PLATFORMS")}
    env["slurm"] = {k: v for k, v in os.environ.items()
                    if k.startswith("SLURM_")}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,"
                            "memory.total,compute_cap,clocks.max.sm",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        env["nvidia_smi"] = r.stdout.strip()
    except Exception as e:
        env["nvidia_smi"] = "unavailable: %r" % e
    pkgs = {}
    for p in ("jax", "jaxlib", "numpyro", "numpy", "scipy",
              "jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cublas-cu12",
              "nvidia-cusolver-cu12", "nvidia-cuda-runtime-cu12", "ringdown"):
        try:
            pkgs[p] = md.version(p)
        except Exception:
            pkgs[p] = None
    env["packages"] = pkgs
    env["contention_before"] = gpu_contention() if _IS_GPU else {}
    print("\n=== ENVIRONMENT [leg %s] ===" % LEG)
    for k in ("hostname", "jax_backend", "jax_devices", "jax_device_kinds",
              "jax_enable_x64", "dtype", "cpu_model", "cpu_affinity_count",
              "nvidia_smi"):
        print("  %-20s %s" % (k, env.get(k)))
    print("  %-20s %s" % ("threads", env["thread_env"]))
    print("  %-20s jax=%s jaxlib=%s numpyro=%s cublas=%s cusolver=%s"
          % ("versions", pkgs.get("jax"), pkgs.get("jaxlib"), pkgs.get("numpyro"),
             pkgs.get("nvidia-cublas-cu12"), pkgs.get("nvidia-cusolver-cu12")))
    if _IS_GPU:
        print("  %-20s %s" %
              ("gpu_state", env["contention_before"].get("gpu_state")))
        nfor = env["contention_before"].get("n_foreign_compute_apps")
        if nfor:
            print("  *** WARNING: %d OTHER compute process(es) on this GPU; "
                  "absolute timings will be contended. ***" % nfor)
    return env


# ---------------------------------------------------------------------------
# SECTION: correctness (float64 -- an algebra check, not a precision check;
# the float32 accuracy question is sections f32ref/f32acc)
# ---------------------------------------------------------------------------
TOL = 1e-11


def section_correctness():
    cfgs = ([(2, 205, 2)] if ARGS.smoke
            else [(1, 205, 1), (2, 205, 2), (3, 205, 3), (2, 1024, 4)])
    npts = 2 if ARGS.smoke else 3
    print("\n=== CORRECTNESS: prototypes vs unmodified ringdown.model.make_model ===")
    print("    tolerance %.0e on the potential and on EVERY gradient component" % TOL)
    res, ok_all = {}, True
    for n_det, ntime, nmodes in cfgs:
        modes = modes_of(nmodes)
        args = make_args(n_det, ntime)
        ref = rdm.make_model(modes=modes, marginalized=True, **KW)
        cands = [(nm, f(modes, **KW)) for nm, f in VARIANTS
                 if nm != "floor_no_likelihood"]
        init = initialize_model(jax.random.PRNGKey(1), ref, model_args=args)
        rng = np.random.default_rng(7)
        worst = {nm: 0.0 for nm, _ in cands}
        worstp = {nm: 0.0 for nm, _ in cands}
        for _ in range(npts):
            p = {k: jnp.asarray(rng.normal(size=v.shape), dtype=DT)
                 for k, v in init.param_info.z.items()}
            pr = potential_energy(ref, args, {}, p)
            gr = jax.grad(lambda q: potential_energy(ref, args, {}, q))(p)
            for nm, m in cands:
                pc = potential_energy(m, args, {}, p)
                gc = jax.grad(
                    lambda q, _m=m: potential_energy(_m, args, {}, q))(p)
                dp = float(abs(pc - pr) / abs(pr))
                d = dp
                for k in gr:
                    den = float(jnp.max(jnp.abs(gr[k]))) + 1e-300
                    d = max(d, float(jnp.max(jnp.abs(gc[k] - gr[k]))) / den)
                worst[nm] = max(worst[nm], d)
                worstp[nm] = max(worstp[nm], dp)
        key = "%d,%d,%d" % (n_det, ntime, nmodes)
        res[key] = {nm: {"pot": worstp[nm], "pot_and_grad": worst[nm],
                         "pass": worst[nm] < TOL} for nm in worst}
        line = "  (%d,%4d,%d) " % (n_det, ntime, nmodes)
        for nm in worst:
            ok_all &= worst[nm] < TOL
            line += " %s=%.1e[%s]" % (nm, worst[nm],
                                      "OK " if worst[nm] < TOL else "FAIL")
        print(line, flush=True)
    res["_all_pass"] = ok_all
    if not ok_all:
        print("  *** CORRECTNESS FAILURE -- the timings below are meaningless. ***")
    return res


# ---------------------------------------------------------------------------
# SECTION: per-gradient device time
# ---------------------------------------------------------------------------
def section_devtime():
    rep = 3 if ARGS.smoke else 5
    target = 0.10 if ARGS.smoke else 0.15
    print("\n=== PER-GRADIENT DEVICE TIME (us/grad) [leg %s] ===" % LEG)
    res = {}
    for n_det, ntime, nmodes in CONFIGS:
        modes = modes_of(nmodes)
        args = make_args(n_det, ntime)
        key = "%d,%d,%d" % (n_det, ntime, nmodes)
        res[key] = {}
        print("  --- (n_det=%d, n_analyze=%d, n_modes=%d)  k=%d ---"
              % (n_det, ntime, nmodes, 4 * nmodes))
        base = None
        variants = (VARIANTS if (n_det, ntime, nmodes) == PROD
                    else [v for v in VARIANTS if v[0] != "whiten_seq"])
        for nm, f in variants:
            model = f(modes, **KW)
            init = initialize_model(
                jax.random.PRNGKey(1), model, model_args=args)
            p = init.param_info.z
            g = _grad_of(model, args)
            tc0 = time.perf_counter()
            cc = count_custom_calls(g, p)
            t_compile = time.perf_counter() - tc0
            us, R1, R2 = device_us_per_grad(g, p, target_s=target, rep=rep)
            if base is None:
                base = us
            res[key][nm] = {"us_per_grad": us, "speedup_vs_current": base / us,
                            "R": [R1, R2], "compile_s": t_compile,
                            "custom_calls": cc}
            print("    %-22s %10.1f us  %6.2fx   %s"
                  % (nm, us, base / us,
                     " ".join("%s=%d" % (k, v) for k, v in cc.items()
                              if k != "hlo_lines")),
                  flush=True)
    return res


# ---------------------------------------------------------------------------
# SECTION: cuBLAS trsm RHS sweep (the >=17 threshold)
# ---------------------------------------------------------------------------
def section_rhs():
    print(
        "\n=== TRIANGULAR-SOLVE RHS SWEEP (forward only, unbatched) [%s] ===" % LEG)
    print("    A6000/f64 reference: flat ~75-132 us for RHS 1..16 at n=205, then")
    print("    a 3.5x THRESHOLD at RHS>=17 (not an odd/even effect -- verifier),")
    print("    with an anomalously fast point at exactly 32.")
    rng = np.random.default_rng(0)
    res = {}
    if ARGS.smoke:
        plan = [(205, [8, 16, 17, 18, 32])]
    elif ARGS.x64:
        plan = [(205, list(range(8, 41))),
                (1024, [8, 9, 12, 16, 17, 18, 24, 32, 33, 40])]
    else:
        plan = [(205, [8, 9, 12, 14, 16, 17, 18, 20, 24, 28, 32, 33, 36, 40])]
    fn = jax.jit(lambda A, X: jsp.linalg.solve_triangular(
        A, X, lower=True).sum())
    for n, ks in plan:
        C = rng.normal(size=(n, n))
        C = C @ C.T / n + n * np.eye(n)
        L = jnp.asarray(np.linalg.cholesky(C), dtype=DT)
        res[str(n)] = {}
        for k in ks:
            B = jnp.asarray(rng.normal(size=(n, k)), dtype=DT)
            jax.block_until_ready(fn(L, B))
            reps, inner = (3, 30) if ARGS.smoke else (7, 50)
            T = []
            for _ in range(reps):
                t0 = time.perf_counter()
                for _ in range(inner):
                    r = fn(L, B)
                jax.block_until_ready(r)
                T.append((time.perf_counter() - t0) / inner * 1e6)
            res[str(n)][str(k)] = stats.median(T)
        ordered = sorted(res[str(n)].items(), key=lambda kv: int(kv[0]))
        print("  n=%d:" % n)
        for i in range(0, len(ordered), 8):
            print("     " + "  ".join("k=%s:%8.1f" % (k, v)
                                      for k, v in ordered[i:i + 8]))
        best = None
        for (k0, v0), (k1, v1) in zip(ordered[:-1], ordered[1:]):
            if best is None or v1 / max(v0, 1e-9) > best[2]:
                best = (int(k0), int(k1), v1 / max(v0, 1e-9))
        res[str(n)]["_max_step"] = {"from": best[0],
                                    "to": best[1], "ratio": best[2]}
        print("     largest single-step jump: k=%d -> k=%d  x%.2f" %
              best, flush=True)
    return res


# ---------------------------------------------------------------------------
# SECTION: vectorized-chain scaling
# ---------------------------------------------------------------------------
def _run_mcmc(model, args, chains, method, N, key=0):
    # numpyro validates chain_method against the literal set, so the "shipped
    # default" case must pass 'parallel' explicitly (which is what MCMC's own
    # default is) -- passing None raises.
    method = "parallel" if method is None else method
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mc = MCMC(NUTS(model, dense_mass=True), num_warmup=N, num_samples=N,
                  num_chains=chains, chain_method=method, progress_bar=False)
        mc.run(jax.random.PRNGKey(key), *args, extra_fields=("num_steps",))
        jax.block_until_ready(mc.get_samples()["m"])
        msgs = [str(x.message)[:100]
                for x in w if "device" in str(x.message).lower()]
    wall = time.perf_counter() - t0
    try:
        nsteps = int(np.sum(np.asarray(mc.get_extra_fields()["num_steps"])))
    except Exception:
        nsteps = None
    return wall, nsteps, msgs


def section_chains():
    N = 100 if ARGS.smoke else 250
    counts_full = [1, 4] if ARGS.smoke else [1, 4, 16, 64]
    modes = modes_of(PROD[2])
    args = make_args(PROD[0], PROD[1])
    print("\n=== VECTORIZED-CHAIN SCALING [%s], point %s, %d+%d per chain ==="
          % (LEG, str(PROD), N, N))
    print("  %-18s %-11s %6s %9s %10s %11s %13s"
          % ("variant", "method", "chains", "wall_s", "s/chain", "vs_1chain",
             "ms/chain-iter"))
    res = {}
    for nm in CHAIN_VARIANTS:
        model = VMAP[nm](modes, **KW_MCMC)
        res[nm] = {}
        base = None
        if ARGS.platform == "cpu":
            # CPU: the 4-chain default is the row
            counts = [1]
        elif nm == "current":
            # the shipped model is 3-4x slower and is only the status-quo
            # anchor; 1 chain plus the 4-chain default is enough for that
            counts = [1]
        elif nm == "R1_unroll_sep" and not ARGS.x64:
            counts = [c for c in counts_full if c != 4]   # f32 budget trim
        else:
            counts = counts_full
        for C in counts:
            try:
                wall, nsteps, msgs = _run_mcmc(model, args, C, "vectorized", N)
            except Exception as e:
                print("    %-18s vectorized %6d  FAILED: %r" % (nm, C, e))
                res[nm][str(C)] = {"error": repr(e)}
                continue
            if base is None:
                base = wall
            res[nm][str(C)] = {
                "wall_s": wall, "num_steps": nsteps,
                "throughput_vs_1chain": base * C / wall,
                "us_per_leapfrog_incl_compile":
                    (wall / nsteps * 1e6) if nsteps else None,
                "ms_per_chain_iteration": wall / (C * 2 * N) * 1e3,
                "warnings": msgs}
            print("  %-18s %-11s %6d %9.2f %10.3f %10.2fx %13.3f"
                  % (nm, "vectorized", C, wall, wall / C, base * C / wall,
                     wall / (C * 2 * N) * 1e3), flush=True)
        # The shipped default: num_chains=4, chain_method unset.  On one GPU it
        # silently degrades to sequential; on CPU it is real parallelism and is
        # the production baseline.
        if ARGS.platform == "cpu" or nm == "current":
            try:
                wall, nsteps, msgs = _run_mcmc(model, args, 4, None, N)
                res[nm]["4_default"] = {
                    "wall_s": wall, "num_steps": nsteps, "warnings": msgs,
                    "ms_per_chain_iteration": wall / (4 * 2 * N) * 1e3}
                print("  %-18s %-11s %6d %9.2f %10.3f            %13.3f  %s"
                      % (nm, "default", 4, wall, wall / 4,
                         wall / (4 * 2 * N) * 1e3,
                         msgs[0][:55] if msgs else ""), flush=True)
            except Exception as e:
                res[nm]["4_default"] = {"error": repr(e)}
    return res


# ---------------------------------------------------------------------------
# SECTION: compile time
# ---------------------------------------------------------------------------
def section_compile():
    print("\n=== COMPILE TIME [%s] ===" % LEG)
    res = {"isolated_grad": {}, "mcmc": {}}
    cfgs = [PROD] if (ARGS.smoke or not ARGS.x64) else [PROD, (3, 1024, 8)]
    for n_det, ntime, nmodes in cfgs:
        modes = modes_of(nmodes)
        args = make_args(n_det, ntime)
        key = "%d,%d,%d" % (n_det, ntime, nmodes)
        res["isolated_grad"][key] = {}
        for nm, f in VARIANTS:
            model = f(modes, **KW)
            init = initialize_model(
                jax.random.PRNGKey(1), model, model_args=args)
            g = jax.jit(
                jax.grad(lambda q: potential_energy(model, args, {}, q)))
            t0 = time.perf_counter()
            g.lower(init.param_info.z).compile()
            dt = time.perf_counter() - t0
            res["isolated_grad"][key][nm] = dt
            print("  isolated jit(grad) %-12s %-22s %6.2f s" % (key, nm, dt),
                  flush=True)
    # Full-NUTS compile, by the affine fit over N in {250,1000} at 1 chain.
    # CLAIMS_VERIFICATION flags this decomposition as fragile (chains
    # decorrelate between variants), so the raw wall times and the leapfrog
    # counts are recorded alongside it.  R1_vmap only, to stay in budget.
    if not ARGS.smoke and ARGS.platform == "gpu":
        modes = modes_of(PROD[2])
        args = make_args(PROD[0], PROD[1])
        for nm in ["R1_vmap"]:
            model = VMAP[nm](modes, **KW_MCMC)
            try:
                w1, s1, _ = _run_mcmc(model, args, 1, "vectorized", 250)
                w2, s2, _ = _run_mcmc(model, args, 1, "vectorized", 1000)
                slope = (w2 - w1) / (2 * 750)
                res["mcmc"][nm] = {
                    "wall_250": w1, "wall_1000": w2, "steps_250": s1,
                    "steps_1000": s2, "ms_per_iter": slope * 1e3,
                    "compile_setup_s": w1 - slope * 500,
                    "us_per_leapfrog": ((w2 - w1) / (s2 - s1) * 1e6)
                    if (s1 and s2 and s2 != s1) else None}
                print("  MCMC %-14s N=250 %6.2fs N=1000 %6.2fs -> compile ~%5.2fs "
                      " %6.3f ms/iter" % (nm, w1, w2, w1 - slope * 500,
                                          slope * 1e3), flush=True)
            except Exception as e:
                res["mcmc"][nm] = {"error": repr(e)}
    return res


# ---------------------------------------------------------------------------
# SECTIONS f32ref / f32acc: the float32 accuracy + robustness test
#
# GPU_BENCHMARKS.md section 3.2 found, on the A6000, that at realistic
# conditioning the float32 GRADIENT is ~3 orders of magnitude worse than the
# float32 log density, and that the SHIPPED sequential form returns NaN at
# cond(A) >= 5e6 where R1 still returns a finite value.  Both are precision
# properties, not silicon properties, so they should reproduce on any card --
# which is exactly why it is worth confirming that they do before trusting
# float32 production runs on new hardware.
#
# f32ref (float64 leg) dumps potentials + gradients at fixed unconstrained
# points for a sweep of a_scale_max and mode sets; f32acc (float32 leg) reloads
# the dump and compares at the identical points.
# ---------------------------------------------------------------------------
F32_CASES = [("220_221", 1e-20), ("220_221", 1.0), ("220_221", 1e3),
             ("220_221", 1e6), ("degenerate", 1.0), ("degenerate", 1e3)]
F32_VARIANTS = ["current", "R1_unroll_sep", "R1_vmap"]


def _modes_for(tagname):
    return DEGENERATE if tagname == "degenerate" else modes_of(2)


def _cond_A_inv(p, args, modes, a_scale_max):
    """cond(A^-1) at the given UNCONSTRAINED point, in float64 numpy.

    Replicates numpyro's biject_to(interval) = sigmoid then affine, which is
    what `potential_energy` applies to these Uniform sites.  Diagnostic only.
    """
    try:
        def sg(x): return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))
        m = 50.0 + 100.0 * sg(p["m"])
        chi = 0.99 * sg(p["chi"])
        a_scale = a_scale_max * sg(p["a_scale"])
        fco, gco = _coeffs(modes)
        f0 = 1.0 / (m * qnms.T_MSUN)
        f = f0 * np.asarray(chi_factors(jnp.asarray(chi, dtype=DT), fco),
                            dtype=np.float64)
        g = f0 * np.asarray(chi_factors(jnp.asarray(chi, dtype=DT), gco),
                            dtype=np.float64)
        times, strains, ls, fps, fcs = [np.asarray(a, dtype=np.float64)
                                        for a in args]
        dms = np.asarray(rd_design_matrix(
            jnp.asarray(times, dtype=DT), jnp.asarray(f, dtype=DT),
            jnp.asarray(g, dtype=DT), jnp.asarray(fps, dtype=DT),
            jnp.asarray(fcs, dtype=DT), jnp.asarray(a_scale, dtype=DT)),
            dtype=np.float64)
        k = dms.shape[2]
        A = np.eye(k)
        for i in range(dms.shape[0]):
            W = np.linalg.solve(np.tril(ls[i]), dms[i])
            A = A + W.T @ W
        return float(np.linalg.cond(A))
    except Exception:
        return float("nan")


def section_f32ref(ref_path):
    """float64 leg: dump reference potentials + gradients for the f32 test."""
    print("\n=== FLOAT64 REFERENCE DUMP for the float32 accuracy test ===")
    args = make_args(2, 205)
    rng = np.random.default_rng(11)
    blob, summary = {}, {}
    npts = 2 if ARGS.smoke else 3
    print("  %-12s %-11s %11s   %s" % ("modes", "a_scale_max", "cond(A^-1)",
                                       "float64 potential (pt0)"))
    for tagname, asm in F32_CASES:
        modes = _modes_for(tagname)
        kw = dict(KW, a_scale_max=asm)
        case = "%s|%g" % (tagname, asm)
        try:
            ref = rdm.make_model(modes=modes, marginalized=True, **kw)
            init = initialize_model(
                jax.random.PRNGKey(1), ref, model_args=args)
            shapes = {k: v.shape for k, v in init.param_info.z.items()}
        except Exception as e:
            print("  %-12s %-11.0e  initialize_model FAILED (%r)"
                  % (tagname, asm, e))
            summary[case] = {"init_error": repr(e)}
            continue
        pts = [{k: rng.normal(size=s) for k, s in shapes.items()}
               for _ in range(npts)]
        for j, p in enumerate(pts):
            for k, v in p.items():
                blob["%s|pt%d|z|%s" % (case, j, k)] = np.asarray(
                    v, dtype=np.float64)
        summary[case] = {"cond_A_inv": _cond_A_inv(pts[0], args, modes, asm),
                         "n_pts": npts}
        for nm in F32_VARIANTS:
            model = VMAP[nm](modes, **kw)
            gfn = jax.grad(
                lambda q, _m=model: potential_energy(_m, args, {}, q))
            for j, p in enumerate(pts):
                pj = {k: jnp.asarray(v, dtype=DT) for k, v in p.items()}
                try:
                    U = float(potential_energy(model, args, {}, pj))
                    G = gfn(pj)
                    blob["%s|pt%d|U|%s" % (case, j, nm)] = np.array([U])
                    for k, v in G.items():
                        blob["%s|pt%d|g|%s|%s" % (case, j, nm, k)] = \
                            np.asarray(v, dtype=np.float64)
                except Exception as e:
                    blob["%s|pt%d|U|%s" % (case, j, nm)] = np.array([np.nan])
                    summary[case]["%s_error" % nm] = repr(e)
            summary[case][nm +
                          "_U"] = float(blob["%s|pt0|U|%s" % (case, nm)][0])
        print("  %-12s %-11.0e %11.3e   %s"
              % (tagname, asm, summary[case]["cond_A_inv"],
                 "  ".join("%s=%.6f" % (nm, summary[case].get(nm + "_U", np.nan))
                           for nm in F32_VARIANTS)), flush=True)
    np.savez(ref_path, **blob)
    print("  -> %s (%d arrays)" % (ref_path, len(blob)))
    return {"path": ref_path, "npts": npts, "summary": summary}


def section_f32acc(ref_path):
    """float32 leg: recompute at the same points and compare to the dump."""
    print("\n=== FLOAT32 ACCURACY / ROBUSTNESS vs the float64 reference ===")
    print("    A6000 reference (GPU_BENCHMARKS 3.2): at cond(A)~26 the f32")
    print("    GRADIENT is only good to ~5e-4 (current) / ~7e-5 (R1) while the")
    print("    potential looks fine at ~5e-7; and `current` goes NaN at")
    print("    cond(A) >= 5e6 where R1 still returns a finite value.")
    if not (ref_path and os.path.exists(ref_path)):
        print("  no reference dump at %r -- section skipped" % ref_path)
        return {"error": "no reference dump at %r" % ref_path}
    blob = np.load(ref_path)
    keys = list(blob.keys())
    cases = sorted(set(k.split("|pt")[0] for k in keys),
                   key=lambda c: (c.split("|")[0], float(c.split("|")[1])))
    args = make_args(2, 205)
    res = {}
    print("  %-20s %-18s %11s %11s %8s"
          % ("case", "variant", "rel_err_U", "rel_err_grad", "all_finite"))
    for case in cases:
        tagname, asm = case.split("|")[0], float(case.split("|")[1])
        modes = _modes_for(tagname)
        kw = dict(KW, a_scale_max=asm)
        pts = sorted(set(int(m.group(1)) for m in
                         (re.match(re.escape(case) + r"\|pt(\d+)\|", k)
                          for k in keys) if m))
        res[case] = {}
        for nm in F32_VARIANTS:
            try:
                model = VMAP[nm](modes, **kw)
                gfn = jax.grad(
                    lambda q, _m=model: potential_energy(_m, args, {}, q))
            except Exception as e:
                res[case][nm] = {"error": repr(e)}
                continue
            eU, eG, finite = 0.0, 0.0, True
            for j in pts:
                pre = "%s|pt%d|z|" % (case, j)
                p = {k[len(pre):]: jnp.asarray(blob[k], dtype=DT)
                     for k in keys if k.startswith(pre)}
                if not p:
                    continue
                try:
                    U = float(potential_energy(model, args, {}, p))
                    G = {k: np.asarray(v, dtype=np.float64)
                         for k, v in gfn(p).items()}
                except Exception:
                    finite = False
                    continue
                U0 = float(blob["%s|pt%d|U|%s" % (case, j, nm)][0])
                if not np.isfinite(U):
                    finite = False
                    eU = float("nan")
                elif np.isfinite(U0):
                    eU = max(eU, abs(U - U0) / max(abs(U0), 1e-300))
                for k, v in G.items():
                    g0 = blob["%s|pt%d|g|%s|%s" % (case, j, nm, k)]
                    if not np.all(np.isfinite(v)):
                        finite = False
                        eG = float("nan")
                        continue
                    eG = max(eG, float(np.max(np.abs(v - g0))
                                       / (np.max(np.abs(g0)) + 1e-300)))
            res[case][nm] = {"rel_err_U": eU, "rel_err_grad": eG,
                             "all_finite": finite}
            print("  %-20s %-18s %11.2e %11.2e %8s"
                  % (case, nm, eU, eG, finite), flush=True)
    return res


# ---------------------------------------------------------------------------
# Subprocess launcher
# ---------------------------------------------------------------------------
def _spawn(tag, extra_argv, out_path, timeout=2400):
    cmd = [sys.executable, os.path.abspath(__file__),
           "--no-sub", "--out", out_path] + extra_argv
    if ARGS.smoke:
        cmd.append("--smoke")
    print("\n" + "-" * 78)
    print("=== LEG: %s ===\n  $ %s" % (tag, " ".join(cmd)), flush=True)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  ! leg timed out after %d s" % timeout)
        return {"error": "timeout"}
    sys.stdout.write(re.sub(r"^", "  | ", r.stdout, flags=re.M))
    print("  (leg wall %.1f s, exit %d)" %
          (time.perf_counter() - t0, r.returncode))
    if r.returncode != 0:
        sys.stdout.write(re.sub(r"^", "  ! ", r.stderr[-4000:], flags=re.M))
        return {"error": "exit %d" % r.returncode, "stderr": r.stderr[-4000:]}
    try:
        with open(out_path) as fh:
            return json.load(fh)
    except Exception as e:
        return {"error": repr(e)}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    out = {"schema": 2, "leg": LEG, "argv": sys.argv, "tag": ARGS.tag}
    out["env"] = section_env()
    base = os.path.splitext(ARGS.out)[0]
    ref_path = ARGS.ref or (base + ".f64ref.npz")

    if want("correctness"):
        out["correctness"] = section_correctness()
        if not out["correctness"].get("_all_pass", False):
            print("\nCorrectness failed; continuing so the failure is recorded, "
                  "but DO NOT trust the timings.\n")
    # f32ref belongs to the float64 leg (it *produces* the reference); f32acc
    # belongs to the float32 leg (it *consumes* it).  Running f32acc under
    # float64 would compare the reference against itself and report 0.0.
    if want("f32ref") and ARGS.x64:
        out["f32ref"] = section_f32ref(ref_path)
    if want("f32acc") and not ARGS.x64:
        out["f32acc"] = section_f32acc(ARGS.ref or ref_path)
    # Order matters: if the budget runs out, shed the cheap diagnostics (rhs,
    # compile) rather than the two headline measurements.
    if want("devtime"):
        out["devtime"] = section_devtime()
    if want("chains"):
        out["chains"] = section_chains()
    if want("rhs"):
        out["rhs_sweep"] = section_rhs()
    if want("compile"):
        out["compile"] = section_compile()

    # ---- child legs (only the top-level GPU float64 process spawns these) ----
    if not ARGS.no_sub and ARGS.platform == "gpu" and ARGS.x64:
        def budget(cap):
            """Cap, but never more than what is left of the total budget."""
            left = ARGS.deadline - (time.time() - T_START)
            return "%.0f" % max(60.0, min(cap, left))

        out["gpu_f32"] = _spawn(
            "GPU float32 -- a PRODUCTION mode (ringdown_fit.py `float32` flag)",
            ["--platform", "gpu", "--x64", "0", "--ref", ref_path,
             "--sections", "env,f32acc,devtime,rhs,compile,chains",
             "--deadline", budget(600)],
            base + ".gpu_f32.json")
        out["cpu_f64"] = _spawn(
            "host CPU float64, all allocated cores, per-gradient timings "
            "(JAX_PLATFORMS=cpu set BEFORE `import jax`; ONE CPU device, as in "
            "the A6000 study)",
            ["--platform", "cpu", "--sections", "env,devtime",
             "--deadline", budget(300)],
            base + ".cpu_f64.json")
        out["cpu_f64_chains"] = _spawn(
            "host CPU float64, chain scaling with numpyro.set_host_device_count"
            "(4) -- the production CPU configuration",
            ["--platform", "cpu", "--sections", "env,chains",
             "--deadline", budget(200)],
            base + ".cpu_f64_chains.json")
        out["cpu_f64_omp1"] = _spawn(
            "host CPU float64, OMP_NUM_THREADS=1 (production CLI setting), "
            "production point only",
            ["--platform", "cpu", "--omp", "1", "--sections", "env,devtime",
             "--configs", "2,205,2", "--deadline", budget(240)],
            base + ".cpu_f64_omp1.json")

    if _IS_GPU:
        out["contention_after"] = gpu_contention()
    out["t_total_s"] = time.time() - T_START
    with open(ARGS.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\n=== LEG %s DONE in %.1f s -> %s ==="
          % (LEG, out["t_total_s"], os.path.abspath(ARGS.out)))
    if not ARGS.no_sub:
        print("Next:  python %s/analyze.py %s"
              % (os.path.dirname(os.path.abspath(__file__)),
                 os.path.abspath(ARGS.out)))


if __name__ == "__main__":
    main()
