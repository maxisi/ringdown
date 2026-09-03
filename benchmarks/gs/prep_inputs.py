#!/usr/bin/env python
"""Build the inputs of the Gohberg-Semencul benchmark kit (numpy/scipy, CPU f64).

    python benchmarks/gs/prep_inputs.py --out benchmarks/gs/inputs [--smoke]

Run once from the repo root with PYTHONPATH=<repo>.  Writes one npz per
(family, n_det, N, n_modes) named {family}_d{n_det}_n{N}_m{n_modes}.npz with

  times, strains        (n_det, N)   times = arange(N) * dt, dt = 1/2048 s
  fps, fcs              (n_det,)     antenna patterns (tests/test_model.py _make_data)
  acf                   (n_det, N)   autocovariance (unit-normalized, except gw150914)
  cond                  (n_det,)     2-norm condition number of C = toeplitz(acf)
  L, Linv, Cinv         (n_det, N, N) Cholesky factor, its inverse, C^{-1}
  a, atilde             (n_det, N)   Yule-Walker filter and its reversal (note eq. 3.5, 4.2),
                                     from scipy.linalg.solve_toeplitz as PR #141 does
  sigma2                (n_det,)     innovation variance sigma^2 = acf[0] + a[1:] @ acf[1:]
                                     (note eq. 3.4, from the solve_toeplitz a)
  a_ld, atilde_ld       (n_det, N)   the same filter from a longdouble Levinson-Durbin
                                     recursion, rounded to float64 ("refined" coefficients:
                                     solve_toeplitz is a float64 Levinson solve whose forward
                                     error, 4e-9 relative at cond 5e8, is what limits the GS
                                     variants' gradient accuracy on gw150914; the refined a
                                     is exact to ~1e-12)
  sigma2_ld             (n_det,)     sigma_{N-1}^2 of that recursion, as float64
  sigma2_all            (n_det, N)   sigma_m^2, m = 0..N-1 (float64 Levinson-Durbin)
  refl_max              (n_det,)     max_m |kappa_m|
  logdetC_exact         (n_det,)     sum_m log sigma_m^2 (note eq. 3.10) evaluated in
                                     longdouble (ref_longdouble.levinson_ld), as float64:
                                     the log|C| constant every hoisted variant uses
  logdetC_f64lev        (n_det,)     the same sum in float64 (diagnostic: 2.7e-8 nats off at
                                     cond 5e8, 1.6e-4 at cond 1.5e12; do not use as a constant)
  logdetC_chol          (n_det,)     2 sum log diag L in float64 (what ringdown's `main` uses)
  logdetC_pr            (n_det,)     N log sigma^2 (PR #141's shortcut, note eq. 8.2)
  levinson_resid        (n_det,)     ||C a - sigma^2 e0|| / (||C||_F ||a||)
  w                     (n_det, N)   C^{-1} y by longdouble iterative refinement, as f64
  z                     (n_det, N)   L^{-1} y (triangular solve)
  Q                     (n_det,)     y^T C^{-1} y (longdouble inner product, as f64)
  snr_target            scalar       requested network optimal SNR (--snr)
  snr_achieved          scalar       sqrt(sum_i h_i^T C_i^{-1} h_i) of the stored
                                     (scaled) injection, cho_solve in float64
  snr_recipe            scalar       the SNR the injection had before rescaling
                                     (_make_data's a_true = 0.5, or 1e-21 for gw150914)
  a_true                scalar       injected amplitude scale in the units the model
                                     sees (already divided by `scale`)
  a_true_phys           scalar       a_true before the strain scaling (= a_true
                                     unless gw150914)
  a_scale_max, scale    scalars      prior bound (5 a_true) and strain scale (see below)
  n_modes               scalar
  pts|m, pts|chi, pts|a_scale   (P,) (P,) (P, n_modes)  fixed unconstrained points
  pts_kind              (P,) int     0 = N(0,1) draw, 1 = NUTS warmup sample
  warmup_ok, warmup_wall_s, warmup_num_warmup, warmup_num_samples, warmup_seed,
  warmup_accept_prob, warmup_num_steps_mean, warmup_n_divergent   bookkeeping
  family, f_low, dt, seed_data, seed_points                       bookkeeping

Data convention (tests/test_model.py _make_data, with a realistic SNR): the
ACF is unit-normalized (acf[0] = 1) and the strain is colored noise L xi plus
an injected ringdown (f_true = linspace(150, 300), g_true = linspace(30, 80),
N(0,1) quadratures, seed 42; fps/fcs = _make_data's fixed values).  Unlike
_make_data, whose O(1) injection against the tiny in-band variance of the
unit-normalized aLIGO ACF gives a network SNR ~1e4 and Q ~ 1e8, the noiseless
injection h_i is rescaled so that the network optimal SNR
rho = sqrt(sum_i h_i^T C_i^{-1} h_i) equals --snr (default 20) before the
noise is added; the injected amplitude scale a_true is rescaled accordingly
and the prior bound is a_scale_max = 5 a_true.  With this Q = y^T C^{-1} y is
~ n_det N + snr^2 and the posterior is a genuine, informative one.

ACF families
  aligo02, aligo2, aligo20   tests/test_model.py _aligo_acf with f_low = 0.2, 2,
                             20 Hz plus a per-detector offset 0.05 * det (the
                             analytic aLIGO PSD with its low-frequency wall
                             floored below f_low: cond(C) ~ 1e10 down to ~1e2);
                             n_fft = 4096, raised to 16384 when N > 2048
  expcos                     benchmarks/h100/bench.py make_args:
                             exp(-lag/0.01) cos(2 pi 120 lag) + 1e-3 delta, with
                             make_args' N*1e-9 diagonal jitter folded into acf[0]
  white                      acf = e0 (C = I)
  gw150914s1                 the gw150914 family WITHOUT the strain scaling (scale = 1,
                             a_scale_max = 5 a_true_phys ~ 1e-20): the scale-invariance
                             twin (checklist item 8).  Opt-in: not in the default family
                             list; build with --families gw150914s1 and run bench_gs.py
                             with --families gw150914,gw150914s1 (its correctness
                             section then compares main's gradient across the pair).
  gw150914                   H1/L1 ACFs of ringdown.Fit.from_config(
                             'etc/ringdown_fit_example.ini') -> compute_acfs()
                             (GWOSC 4 kHz data, f_min = 20 Hz, downsampled by 2 to
                             2048 Hz, Welch 'fd' estimate); the first N lags are
                             taken.  Detector i uses ifo i mod 2 (a third detector
                             reuses H1).  Strains follow the recipe above with the
                             physical ACF: the injection starts at a_true = 1e-21
                             and is rescaled to the target SNR (the SNR is
                             invariant under the strain scaling, so the
                             normalization is done once, on the physical data).
                             Mimicking Fit.strain_scale (the float32 branch, max
                             over detectors of std(strain)): scale = max_i
                             std(strain_i); strains /= scale, acf /= scale^2,
                             a_scale_max = 5 a_true_phys / scale (as Fit._make_model
                             divides the config's A_scale_max by the scale).
                             All legs get the same scaled inputs.

Fixed points: two kinds, distinguished by pts_kind.
  kind 0   20 draws of N(0, 1) in numpyro's unconstrained coordinates (seed 7,
           the same draws for every file with the same n_modes).  The site
           names and shapes come from numpyro.infer.util.initialize_model on
           ringdown.model.make_model(marginalized=True) with this file's data:
           {'m': (), 'chi': (), 'a_scale': (n_modes,)}.
  kind 1   10 typical-set points: the post-warmup samples of a numpyro NUTS run
           (num_warmup = 150, num_samples = 10, 1 chain, PRNGKey(0), CPU
           float64) of the same model on this file's data, mapped to the
           unconstrained coordinates with numpyro.infer.util.unconstrain_fn.
           The run happens in a child process bounded by --warmup-timeout
           (default 300 s); on failure or timeout the file keeps the 20 kind-0
           rows and the failure is recorded (warmup_ok = False, prep_log.json).

prep_log.json in the output directory records, per file, cond, snr_achieved,
Q, a_true, a_scale_max, scale and the warmup outcome; it is updated in place
so that a resumed run keeps the entries of the files it skips.

Grid: (2,205,2), (2,410,2), (2,1024,2), (2,2048,4), (3,1024,8), (2,4096,4).
--smoke restricts to (2,205,2) and families aligo02, white.  Existing files
are skipped unless --force is given, so an interrupted run can be resumed.

Cost: the full grid takes ~80 min on 8 cores WITH the default NUTS warmups
(measured 79 min: the warmup children account for ~75 min of it, and the six
(2,4096,4) files ~30 min, most of whose warmups hit the 300 s timeout);
--no-warmup brings it to ~5 min; --smoke ~1 min.

--refresh-precompute re-derives only the cheap precomputed keys (a_ld,
atilde_ld, sigma2_ld, logdetC_exact, logdetC_f64lev, logdetC_chol) in
EXISTING npz files and rewrites them in place (atomically), leaving strains,
factors and fixed points untouched.  It upgrades inputs written before those
keys existed (whose logdetC_exact was the float64 Levinson sum) in a few
minutes instead of the 80 min rebuild; prep_log.json records the refresh.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# BLAS threading for the dense factorizations (N = 4096 Cholesky, inverse,
# eigenvalues); must be set before numpy loads its BLAS.  Importing ringdown
# later sets OMP_NUM_THREADS=1 only if it is still unset, which would be too
# late to matter for numpy anyway but would cap XLA; so pin it here.
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np          # noqa: E402
import scipy.linalg         # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import gs_kernels as gk        # noqa: E402
import ref_longdouble as rl    # noqa: E402

DT = 1.0 / 2048.0
FULL_CONFIGS = [(2, 205, 2), (2, 410, 2), (2, 1024, 2), (2, 2048, 4),
                (3, 1024, 8), (2, 4096, 4)]
ALL_FAMILIES = ["aligo02", "aligo2", "aligo20", "expcos", "white", "gw150914"]
SMOKE_CONFIGS = [(2, 205, 2)]
SMOKE_FAMILIES = ["aligo02", "white"]
SEED_DATA = 42        # tests/test_model.py _make_data
SEED_POINTS = 7
N_POINTS = 20
KW_PRIOR = dict(m_min=50.0, m_max=150.0, chi_min=0.0, chi_max=0.99)   # bench.py KW
GW_A_TRUE = 1e-21          # starting (physical) injection amplitude, gw150914 family
RECIPE_A_TRUE = 0.5        # _make_data's a_true, the starting amplitude elsewhere
SNR_DEFAULT = 20.0
A_MAX_OVER_A_TRUE = 5.0    # a_scale_max = 5 a_true
SITES = ("m", "chi", "a_scale")
# typical-set points: NUTS warmup in a child process
WARMUP_NUM_WARMUP = 150
WARMUP_NUM_SAMPLES = 10
WARMUP_SEED = 0
WARMUP_TIMEOUT = 300.0
PREP_LOG = "prep_log.json"


# ---------------------------------------------------------------------------
# ACF families
# ---------------------------------------------------------------------------
def _aligo_psd(f):
    """Analytic aLIGO-like PSD (copy of tests/test_model.py)."""
    x = f / 215.0
    return 1e-49 * (x ** -4.14 - 5 * x ** -2
                    + 111 * (1 - x ** 2 + 0.5 * x ** 4) / (1 + 0.5 * x ** 2))


def _aligo_acf(n, dt=DT, f_low=0.2, n_fft=4096):
    """Unit-normalized aLIGO-like ACF (copy of tests/test_model.py _aligo_acf).

    The low-frequency wall is floored below f_low rather than removed, which
    sets cond(C).  n_fft is raised to 16384 when n > 2048 so that the lags
    requested never reach the periodic wrap of the inverse FFT.
    """
    if n > 2048 and n_fft < 16384:
        n_fft = 16384
    df = 1.0 / (n_fft * dt)
    f = np.arange(n_fft // 2 + 1) * df
    psd = np.where(f >= f_low, _aligo_psd(np.maximum(f, f_low)), _aligo_psd(f_low))
    rho = np.fft.irfft(psd)[:n_fft] / dt
    return rho[:n] / rho[0]


def expcos_acf(n, dt=DT):
    """benchmarks/h100/bench.py make_args ACF with its diagonal jitter folded in."""
    lags = np.arange(n) * dt
    acf = np.exp(-lags / 0.01) * np.cos(2 * np.pi * 120 * lags) + 1e-3 * (lags == 0)
    acf[0] += n * 1e-9
    return acf


def white_acf(n):
    acf = np.zeros(n)
    acf[0] = 1.0
    return acf


_GW_CACHE = {}


def gw150914_acfs():
    """{'H1': acf (4096,), 'L1': acf (4096,)} from the example config (cached)."""
    if not _GW_CACHE:
        import jax
        jax.config.update("jax_enable_x64", True)
        import ringdown
        fit = ringdown.Fit.from_config(os.path.join(_repo_root(), "etc",
                                                    "ringdown_fit_example.ini"))
        for ifo, acf in fit.acfs.items():
            if not np.isclose(acf.delta_t, DT):
                raise RuntimeError("gw150914 ACF delta_t %r != %r" % (acf.delta_t, DT))
            _GW_CACHE[ifo] = np.asarray(acf.values, dtype=np.float64)
        _GW_CACHE["_ifos"] = list(fit.ifos)
        print("  gw150914 ACFs from %s: ifos %s, %d lags, acf[0] = %s"
              % (fit.__class__.__name__, _GW_CACHE["_ifos"],
                 len(_GW_CACHE[_GW_CACHE["_ifos"][0]]),
                 [_GW_CACHE[i][0] for i in _GW_CACHE["_ifos"]]))
    return _GW_CACHE


def _repo_root():
    return os.path.dirname(os.path.dirname(_HERE))


def family_acfs(fam, n_det, N):
    """(acf (n_det, N), f_low (n_det,) or nan) for the family."""
    f_low = np.full(n_det, np.nan)
    if fam.startswith("aligo"):
        base = {"aligo02": 0.2, "aligo2": 2.0, "aligo20": 20.0}[fam]
        f_low = np.array([base + 0.05 * i for i in range(n_det)])
        acf = np.stack([_aligo_acf(N, f_low=f_low[i]) for i in range(n_det)])
    elif fam == "expcos":
        acf = np.stack([expcos_acf(N) for _ in range(n_det)])
    elif fam == "white":
        acf = np.stack([white_acf(N) for _ in range(n_det)])
    elif fam in ("gw150914", "gw150914s1"):
        g = gw150914_acfs()
        ifos = g["_ifos"]
        if N > len(g[ifos[0]]):
            raise ValueError("gw150914 ACF has only %d lags < N = %d" % (len(g[ifos[0]]), N))
        acf = np.stack([g[ifos[i % len(ifos)]][:N] for i in range(n_det)])
    else:
        raise ValueError("unknown family %r" % fam)
    return acf, f_low


# ---------------------------------------------------------------------------
# data (tests/test_model.py _make_data, SNR-normalized) and fixed points
# ---------------------------------------------------------------------------
def make_signal_noise(times, L, n_modes, fps, fcs, a_true, seed=SEED_DATA):
    """(signal, noise): the injected ringdown at amplitude scale a_true and
    the colored noise L xi, as in _make_data (same rng stream: quadratures
    first, then the noise)."""
    import jax.numpy as jnp
    from ringdown.model import rd_design_matrix
    rng = np.random.default_rng(seed)
    n_det, N = times.shape
    f_true = np.linspace(150.0, 300.0, n_modes)
    g_true = np.linspace(30.0, 80.0, n_modes)
    dms = np.asarray(rd_design_matrix(jnp.array(times), jnp.array(f_true),
                                      jnp.array(g_true), jnp.array(fps),
                                      jnp.array(fcs), jnp.full(n_modes, a_true)))
    quads_true = rng.normal(size=dms.shape[2])
    signal = dms @ quads_true
    noise = np.einsum("ijk,ik->ij", L, rng.normal(size=(n_det, N)))
    return signal, noise


def network_snr(signal, L):
    """Network optimal SNR sqrt(sum_i h_i^T C_i^{-1} h_i), C_i = L_i L_i^T,
    by scipy cho_solve in float64."""
    rho2 = 0.0
    for i in range(signal.shape[0]):
        h = np.asarray(signal[i], dtype=np.float64)
        w = scipy.linalg.cho_solve((np.asarray(L[i], dtype=np.float64), True), h)
        rho2 += float(np.dot(h, w))
    return float(np.sqrt(rho2))


def antenna_patterns(n_det):
    fps = np.array([1.0, 0.7, -0.4][:n_det] + [0.5] * max(0, n_det - 3))
    fcs = np.array([0.3, -0.9, 0.6][:n_det] + [-0.2] * max(0, n_det - 3))
    return fps, fcs


def _modes(n_modes):
    return [(1, -2, 2, 2, i) for i in range(n_modes)]


def fixed_points(times, strains, L, fps, fcs, n_modes, a_scale_max):
    """20 N(0,1) draws per unconstrained site of ringdown's main model.

    Site names and shapes are taken from numpyro's initialize_model on
    ringdown.model.make_model(marginalized=True) with these inputs, so the
    points are exactly what potential_energy expects.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    from numpyro.infer.util import initialize_model
    from ringdown.model import make_model
    model = make_model(modes=_modes(n_modes), marginalized=True,
                       a_scale_max=a_scale_max, **KW_PRIOR)
    init = initialize_model(jax.random.PRNGKey(1), model,
                            model_args=(times, strains, L, fps, fcs))
    shapes = {k: tuple(v.shape) for k, v in init.param_info.z.items()}
    expected = {"m": (), "chi": (), "a_scale": (n_modes,)}
    if shapes != expected:
        raise RuntimeError("unconstrained sites %r != expected %r" % (shapes, expected))
    rng = np.random.default_rng(SEED_POINTS)
    # fixed site order so the draws are reproducible whatever dict order numpyro uses
    return {k: rng.normal(size=(N_POINTS,) + expected[k]) for k in SITES}


# ---------------------------------------------------------------------------
# typical-set points: NUTS warmup in a child process (wall-clock bounded)
# ---------------------------------------------------------------------------
def _warmup_child(in_path, out_path):
    """Child entry point: NUTS (num_warmup, num_samples, 1 chain) on the data
    in in_path, post-warmup samples mapped to unconstrained coordinates,
    written to out_path."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import constrain_fn, unconstrain_fn
    from ringdown.model import make_model
    with np.load(in_path) as z:
        d = {k: z[k] for k in z.files}
    n_modes = int(d["n_modes"])
    model = make_model(modes=_modes(n_modes), marginalized=True,
                       a_scale_max=float(d["a_scale_max"]), **KW_PRIOR)
    args = (d["times"], d["strains"], d["L"], d["fps"], d["fcs"])
    mcmc = MCMC(NUTS(model), num_warmup=WARMUP_NUM_WARMUP,
                num_samples=WARMUP_NUM_SAMPLES, num_chains=1, progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(WARMUP_SEED), *args,
             extra_fields=("accept_prob", "num_steps", "diverging"))
    wall = time.perf_counter() - t0
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    z = {k: [] for k in SITES}
    for j in range(WARMUP_NUM_SAMPLES):
        params = {k: jnp.asarray(samples[k][j]) for k in SITES}
        u = unconstrain_fn(model, args, {}, params)
        back = constrain_fn(model, args, {}, u)
        for k in SITES:
            if not np.allclose(np.asarray(back[k]), np.asarray(params[k]),
                               rtol=1e-10, atol=1e-12):
                raise RuntimeError("unconstrain/constrain round trip failed at "
                                   "site %s, sample %d" % (k, j))
            z[k].append(np.asarray(u[k], dtype=np.float64))
    out = {"pts|" + k: np.stack(z[k]) for k in SITES}
    out.update(
        wall_s=np.float64(wall),
        accept_prob=np.float64(np.mean(np.asarray(extra["accept_prob"]))),
        num_steps_mean=np.float64(np.mean(np.asarray(extra["num_steps"]))),
        n_divergent=np.int64(np.sum(np.asarray(extra["diverging"]))),
        m_mean=np.float64(np.mean(np.asarray(samples["m"]))),
        chi_mean=np.float64(np.mean(np.asarray(samples["chi"]))),
        a_scale_mean=np.asarray(np.mean(np.asarray(samples["a_scale"]), axis=0)),
    )
    np.savez(out_path, **out)


def warmup_points(name, out_dir, times, strains, L, fps, fcs, n_modes,
                  a_scale_max, timeout=WARMUP_TIMEOUT):
    """({site: (10, ...) unconstrained} or None, info dict).

    Runs _warmup_child in a fresh interpreter so a hung or slow NUTS run can
    be killed at `timeout` seconds of wall time without affecting the parent.
    """
    in_path = os.path.join(out_dir, ".warmup_%s_in.npz" % name)
    res_path = os.path.join(out_dir, ".warmup_%s_out.npz" % name)
    np.savez(in_path, times=times, strains=strains, L=L, fps=fps, fcs=fcs,
             n_modes=np.int64(n_modes), a_scale_max=np.float64(a_scale_max))
    info = dict(ok=False, wall_s=None, error=None, num_warmup=WARMUP_NUM_WARMUP,
                num_samples=WARMUP_NUM_SAMPLES, seed=WARMUP_SEED, timeout_s=timeout)
    t0 = time.perf_counter()
    try:
        cp = subprocess.run([sys.executable, os.path.abspath(__file__),
                             "--warmup-child", in_path, res_path],
                            capture_output=True, text=True, timeout=timeout)
        if cp.returncode != 0 or not os.path.exists(res_path):
            tail = (cp.stderr or cp.stdout or "").strip().splitlines()[-8:]
            info["error"] = "child rc=%d: %s" % (cp.returncode, " | ".join(tail))
        else:
            with np.load(res_path) as z:
                r = {k: z[k] for k in z.files}
            pts = {k: np.asarray(r["pts|" + k], dtype=np.float64) for k in SITES}
            if not all(np.all(np.isfinite(v)) for v in pts.values()):
                raise RuntimeError("non-finite unconstrained warmup sample")
            info.update(ok=True, child_wall_s=float(r["wall_s"]),
                        accept_prob=float(r["accept_prob"]),
                        num_steps_mean=float(r["num_steps_mean"]),
                        n_divergent=int(r["n_divergent"]),
                        m_mean=float(r["m_mean"]), chi_mean=float(r["chi_mean"]),
                        a_scale_mean=[float(x) for x in np.atleast_1d(r["a_scale_mean"])])
    except subprocess.TimeoutExpired:
        info["error"] = "timeout after %.0f s" % timeout
    except Exception as e:  # noqa: BLE001
        info["error"] = repr(e)
    info["wall_s"] = time.perf_counter() - t0
    for pth in (in_path, res_path):
        try:
            os.remove(pth)
        except OSError:
            pass
    return (pts if info["ok"] else None), info


# ---------------------------------------------------------------------------
# per-file build
# ---------------------------------------------------------------------------
def build_one(fam, cfg, snr=SNR_DEFAULT, warmup=True, warmup_timeout=WARMUP_TIMEOUT,
              out_dir=None, verbose=True):
    """(npz dict, prep_log entry)."""
    n_det, N, n_modes = cfg
    t0 = time.perf_counter()
    acf, f_low = family_acfs(fam, n_det, N)
    times = np.tile(np.arange(N) * DT, (n_det, 1))
    fps, fcs = antenna_patterns(n_det)

    # dense factors of the (unscaled) covariances; L is needed for the noise
    den = [gk.dense_factors(acf[i]) for i in range(n_det)]
    L = np.stack([d["L"] for d in den])

    # injection at the recipe amplitude, then rescaled to the target network SNR
    a0 = GW_A_TRUE if fam.startswith("gw150914") else RECIPE_A_TRUE
    signal, noise = make_signal_noise(times, L, n_modes, fps, fcs, a_true=a0)
    snr_recipe = network_snr(signal, L)
    fac = snr / snr_recipe
    signal = signal * fac
    a_true_phys = a0 * fac
    strains = signal + noise

    if fam == "gw150914":
        # Fit.strain_scale (float32 branch): max over detectors of std(strain)
        # (gw150914s1 keeps scale = 1: the scale-invariance twin)
        scale = float(max(np.std(s) for s in strains))
        strains = strains / scale
        signal = signal / scale
        acf = acf / scale ** 2
        # refactor the scaled covariances (Fit.run_input divides the ACF by
        # scale^2 before taking the Cholesky factor)
        den = [gk.dense_factors(acf[i]) for i in range(n_det)]
        L = np.stack([d["L"] for d in den])
    else:
        scale = 1.0
    a_true = a_true_phys / scale
    a_scale_max = A_MAX_OVER_A_TRUE * a_true
    snr_achieved = network_snr(signal, L)

    lev = [gk.levinson(acf[i]) for i in range(n_det)]
    out = dict(times=times, strains=strains, fps=fps, fcs=fcs, acf=acf,
               cond=np.array([d["cond"] for d in den]),
               L=L, Linv=np.stack([d["Linv"] for d in den]),
               Cinv=np.stack([d["Cinv"] for d in den]),
               a=np.stack([lv["a"] for lv in lev]),
               atilde=np.stack([lv["atilde"] for lv in lev]),
               sigma2=np.array([lv["sigma2"] for lv in lev]),
               sigma2_all=np.stack([lv["sigma2_all"] for lv in lev]),
               refl_max=np.array([np.max(np.abs(lv["refl"])) for lv in lev]),
               logdetC_pr=np.array([lv["logdetC_pr"] for lv in lev]),
               levinson_resid=np.array([lv["levinson_resid"] for lv in lev]),
               snr_target=np.float64(snr), snr_achieved=np.float64(snr_achieved),
               snr_recipe=np.float64(snr_recipe),
               a_true=np.float64(a_true), a_true_phys=np.float64(a_true_phys),
               a_scale_max=np.float64(a_scale_max), scale=np.float64(scale),
               n_modes=np.int64(n_modes), family=np.array(fam), f_low=f_low,
               dt=np.float64(DT), seed_data=np.int64(SEED_DATA),
               seed_points=np.int64(SEED_POINTS))

    out.update(precompute_ld(acf, L, out["logdetC_pr"]))

    # theta-independent data terms: w = C^{-1} y (refined), z = L^{-1} y, Q = y^T w
    w, z, Q, wres = [], [], [], []
    for i in range(n_det):
        y = strains[i]
        w_ld = rl.refine_solve(L[i], den[i]["C"], y, rounds=2)
        wres.append(rl.ld_residual(den[i]["C"], y[:, None], w_ld[:, None]))
        w.append(np.asarray(w_ld, dtype=np.float64))
        z.append(scipy.linalg.solve_triangular(L[i], y, lower=True))
        Q.append(np.float64(np.dot(y.astype(np.longdouble), w_ld)))
    out.update(w=np.stack(w), z=np.stack(z), Q=np.array(Q),
               w_refine_resid=np.array(wres))

    # fixed points: 20 N(0,1) draws (kind 0) + 10 NUTS warmup samples (kind 1)
    pts = fixed_points(times, strains, L, fps, fcs, n_modes, a_scale_max)
    kind = np.zeros(N_POINTS, dtype=np.int64)
    winfo = dict(ok=False, error="disabled (--no-warmup)", wall_s=0.0)
    if warmup:
        wpts, winfo = warmup_points(out_name(fam, cfg)[:-4], out_dir or _HERE,
                                    times, strains, L, fps, fcs, n_modes,
                                    a_scale_max, timeout=warmup_timeout)
        if wpts is not None:
            pts = {k: np.concatenate([pts[k], wpts[k]], axis=0) for k in SITES}
            kind = np.concatenate([kind, np.ones(WARMUP_NUM_SAMPLES, dtype=np.int64)])
    for k, v in pts.items():
        out["pts|" + k] = v
    out["pts_kind"] = kind
    out.update(warmup_ok=np.bool_(winfo["ok"]),
               warmup_wall_s=np.float64(winfo.get("wall_s") or 0.0),
               warmup_num_warmup=np.int64(WARMUP_NUM_WARMUP),
               warmup_num_samples=np.int64(WARMUP_NUM_SAMPLES),
               warmup_seed=np.int64(WARMUP_SEED),
               warmup_accept_prob=np.float64(winfo.get("accept_prob", np.nan)),
               warmup_num_steps_mean=np.float64(winfo.get("num_steps_mean", np.nan)),
               warmup_n_divergent=np.int64(winfo.get("n_divergent", -1)))

    wall = time.perf_counter() - t0
    entry = dict(family=fam, n_det=n_det, N=N, n_modes=n_modes,
                 cond=[float(c) for c in out["cond"]],
                 snr_target=float(snr), snr_achieved=float(snr_achieved),
                 snr_recipe=float(snr_recipe), a_true=float(a_true),
                 a_true_phys=float(a_true_phys), a_scale_max=float(a_scale_max),
                 scale=float(scale), Q=[float(q) for q in out["Q"]],
                 logdet_exact_minus_pr=[float(v) for v in
                                        out["logdetC_exact"] - out["logdetC_pr"]],
                 logdet_f64lev_minus_exact=[float(v) for v in
                                            out["logdetC_f64lev"] - out["logdetC_exact"]],
                 logdet_chol_minus_exact=[float(v) for v in
                                          out["logdetC_chol"] - out["logdetC_exact"]],
                 a_ld_minus_a_relmax=[float(v) for v in
                                      np.max(np.abs(out["a_ld"] - out["a"]), axis=1)
                                      / np.max(np.abs(out["a"]), axis=1)],
                 levinson_resid_max=float(np.max(out["levinson_resid"])),
                 w_refine_resid_max=float(np.max(wres)),
                 n_pts=int(kind.shape[0]), n_pts_warmup=int(np.sum(kind == 1)),
                 warmup=winfo, wall_s=wall)
    if verbose:
        s2chk = np.max(np.abs(out["sigma2_all"][:, -1] - out["sigma2"]) / out["sigma2"])
        print("  %-9s d%d n%-4d m%d  cond=%s  lev_resid=%.1e  s2_all[-1]~s2=%.1e  "
              "logdet_exact-pr=%s  Q=%s  snr=%.3f(recipe %.3g)  a_true=%.3e  "
              "a_max=%.3e  scale=%.3e  w_resid=%.1e  pts=%d  warmup=%s  %.1fs"
              % (fam, n_det, N, n_modes,
                 "/".join("%.1e" % c for c in out["cond"]),
                 np.max(out["levinson_resid"]), s2chk,
                 "/".join("%.2f" % v for v in out["logdetC_exact"] - out["logdetC_pr"]),
                 "/".join("%.1f" % q for q in out["Q"]), snr_achieved, snr_recipe,
                 a_true, a_scale_max, scale, np.max(wres), kind.shape[0],
                 ("ok acc=%.2f steps=%.0f div=%d m=%.1f chi=%.2f %.0fs"
                  % (winfo["accept_prob"], winfo["num_steps_mean"],
                     winfo["n_divergent"], winfo["m_mean"], winfo["chi_mean"],
                     winfo["wall_s"])) if winfo["ok"]
                 else "FAILED(%s)" % winfo["error"],
                 wall), flush=True)
    return out, entry


def out_name(fam, cfg):
    return "%s_d%d_n%d_m%d.npz" % ((fam,) + tuple(cfg))


def precompute_ld(acf, L, logdetC_pr=None):
    """Longdouble Levinson-Durbin per detector -> the refined float64
    constants (see the module docstring): a_ld, atilde_ld, sigma2_ld,
    logdetC_exact (longdouble sum, as f64), logdetC_f64lev (float64 sum),
    logdetC_chol (2 sum log diag L)."""
    acf = np.asarray(acf, dtype=np.float64)
    n_det = acf.shape[0]
    a_ld, s2_ld, ld_exact, f64lev = [], [], [], []
    for i in range(n_det):
        a_i, s2_all_i, _ = rl.levinson_ld(acf[i])
        a_ld.append(np.asarray(a_i, dtype=np.float64))
        s2_ld.append(np.float64(s2_all_i[-1]))
        ld_exact.append(np.float64(np.sum(np.log(s2_all_i))))
        _, s2_f64, _ = gk.levinson_durbin(acf[i], dtype=np.float64)
        f64lev.append(float(np.sum(np.log(s2_f64))))
    a_ld = np.stack(a_ld)
    return dict(a_ld=a_ld,
                atilde_ld=np.concatenate([np.zeros((n_det, 1)), a_ld[:, 1:][:, ::-1]], axis=1),
                sigma2_ld=np.array(s2_ld),
                logdetC_exact=np.array(ld_exact),
                logdetC_f64lev=np.array(f64lev),
                logdetC_chol=np.array([2.0 * np.sum(np.log(np.diag(np.asarray(L[i]))))
                                       for i in range(n_det)]))


def refresh_precompute(path, verbose=True):
    """Add/replace the precompute_ld keys of an existing npz in place
    (write to <path>.tmp.npz, then os.replace).  Returns the prep_log patch."""
    t0 = time.perf_counter()
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    old_exact = np.asarray(d.get("logdetC_exact"), dtype=np.float64)
    new = precompute_ld(d["acf"], d["L"], d.get("logdetC_pr"))
    d.update(new)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)
    patch = dict(refreshed=time.strftime("%Y-%m-%d %H:%M:%S"),
                 old_logdetC_exact_minus_new=[float(v) for v in old_exact - new["logdetC_exact"]],
                 logdet_f64lev_minus_exact=[float(v) for v in
                                            new["logdetC_f64lev"] - new["logdetC_exact"]],
                 logdet_chol_minus_exact=[float(v) for v in
                                          new["logdetC_chol"] - new["logdetC_exact"]],
                 a_ld_minus_a_relmax=[float(v) for v in
                                      np.max(np.abs(new["a_ld"] - d["a"]), axis=1)
                                      / np.max(np.abs(d["a"]), axis=1)],
                 logdet_exact_minus_pr=[float(v) for v in new["logdetC_exact"] - d["logdetC_pr"]])
    if verbose:
        print("  %-28s old_exact-new=%s  f64lev-exact=%s  chol-exact=%s  |a_ld-a|/|a|=%s  %.1fs"
              % (os.path.basename(path),
                 "/".join("%.1e" % v for v in patch["old_logdetC_exact_minus_new"]),
                 "/".join("%.1e" % v for v in patch["logdet_f64lev_minus_exact"]),
                 "/".join("%.1e" % v for v in patch["logdet_chol_minus_exact"]),
                 "/".join("%.1e" % v for v in patch["a_ld_minus_a_relmax"]),
                 time.perf_counter() - t0), flush=True)
    return patch


def _load_log(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            pass
    return {}


def _save_log(path, log):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(_HERE, "inputs"))
    p.add_argument("--smoke", action="store_true",
                   help="(2,205,2) only, families aligo02 and white")
    p.add_argument("--configs", default="", help="e.g. '2,205,2;3,1024,8'")
    p.add_argument("--families", default="", help="comma-separated subset")
    p.add_argument("--force", action="store_true", help="rebuild existing files")
    p.add_argument("--snr", type=float, default=SNR_DEFAULT,
                   help="network optimal SNR of the injection (default %g)" % SNR_DEFAULT)
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the NUTS warmup points (20 N(0,1) rows only)")
    p.add_argument("--warmup-timeout", type=float, default=WARMUP_TIMEOUT,
                   help="wall-clock bound of one NUTS warmup child (s, default %g)"
                   % WARMUP_TIMEOUT)
    p.add_argument("--refresh-precompute", action="store_true",
                   help="only re-derive a_ld/atilde_ld/sigma2_ld/logdetC_* in the "
                        "EXISTING npz files of the selected grid (in place; minutes)")
    p.add_argument("--warmup-child", nargs=2, metavar=("IN", "OUT"),
                   help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.warmup_child:
        _warmup_child(*args.warmup_child)
        return

    configs = SMOKE_CONFIGS if args.smoke else FULL_CONFIGS
    if args.configs:
        configs = [tuple(int(x) for x in c.split(",")) for c in args.configs.split(";") if c.strip()]
    fams = SMOKE_FAMILIES if args.smoke else ALL_FAMILIES
    if args.families:
        fams = [f.strip() for f in args.families.split(",") if f.strip()]
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, PREP_LOG)
    plog = _load_log(log_path)
    if args.refresh_precompute:
        print("prep_inputs --refresh-precompute: %d configs x %d families in %s"
              % (len(configs), len(fams), os.path.abspath(args.out)))
        n = 0
        for cfg in configs:
            for fam in fams:
                path = os.path.join(args.out, out_name(fam, cfg))
                if not os.path.exists(path):
                    print("  %-28s missing, skipped" % out_name(fam, cfg))
                    continue
                patch = refresh_precompute(path)
                plog.setdefault(out_name(fam, cfg)[:-4], {})["refresh_precompute"] = patch
                _save_log(log_path, plog)
                n += 1
        print("refreshed %d files; %s updated" % (n, log_path))
        return
    plog["_meta"] = dict(argv=sys.argv, snr_target=args.snr,
                         warmup=not args.no_warmup, warmup_num_warmup=WARMUP_NUM_WARMUP,
                         warmup_num_samples=WARMUP_NUM_SAMPLES, warmup_seed=WARMUP_SEED,
                         warmup_timeout_s=args.warmup_timeout, n_points_normal=N_POINTS,
                         seed_data=SEED_DATA, seed_points=SEED_POINTS,
                         omp_num_threads=os.environ["OMP_NUM_THREADS"],
                         date=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("prep_inputs: %d configs x %d families -> %s (OMP_NUM_THREADS=%s, snr=%g, warmup=%s)"
          % (len(configs), len(fams), os.path.abspath(args.out),
             os.environ["OMP_NUM_THREADS"], args.snr, not args.no_warmup))
    t0 = time.perf_counter()
    for cfg in configs:
        for fam in fams:
            path = os.path.join(args.out, out_name(fam, cfg))
            if os.path.exists(path) and not args.force:
                print("  %-9s d%d n%-4d m%d  exists, skipped" % ((fam,) + tuple(cfg)))
                continue
            out, entry = build_one(fam, cfg, snr=args.snr, warmup=not args.no_warmup,
                                   warmup_timeout=args.warmup_timeout, out_dir=args.out)
            np.savez(path, **out)
            plog[out_name(fam, cfg)[:-4]] = entry
            _save_log(log_path, plog)
    n_fail = sum(1 for k, v in plog.items() if not k.startswith("_")
                 and not v.get("warmup", {}).get("ok", False))
    print("done in %.1f s; %s: %d files, %d without warmup points"
          % (time.perf_counter() - t0, log_path, len(plog) - 1, n_fail))


if __name__ == "__main__":
    main()
