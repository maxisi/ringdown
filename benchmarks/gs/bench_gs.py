#!/usr/bin/env python
"""Benchmark the Gohberg-Semencul route against the Cholesky route.

Measures, for several ways of applying C^{-1} inside ringdown's one-shot
marginalized likelihood (see variants.py and docs/gohberg_semencul_likelihood.md
section 11):

  * performance: device time per gradient (fori_loop slope harness copied
    from benchmarks/h100/bench.py), compile time, HLO op census on the plain
    and on the looped gradient (to detect hoisting of constant work);
  * numerical stability: potential and gradient of the likelihood part
    against an extended-precision (longdouble) reference at fixed
    unconstrained points, in float64 and float32, on CPU and GPU, tabulated
    against cond(C).

Inputs (ACF families, strains, precomputed constants, fixed points) come from
prep_inputs.py as one npz per (family, n_det, N, n_modes).

Legs
----
Precision and platform are process-global in JAX, so each leg is a separate
process with JAX_PLATFORMS / OMP_NUM_THREADS / jax_enable_x64 set BEFORE
`import jax`.  The parent (no --no-sub) only orchestrates and spawns:

  cpu_f64_omp8   env, reference, correctness, devtime, compile, f32acc, [nuts]
                 (the only leg that writes the reference; runs the 4096 config;
                 BLAS and XLA:CPU pool both 8 threads; `8` is --cpu-omp)
  cpu_f64_omp1   env, correctness, devtime, compile   (BLAS 1 thread, XLA pool 1)
  cpu_f64_prod   env, devtime, compile   (production: `import ringdown` sets only
                 OMP_NUM_THREADS=1, so BLAS has 1 thread while XLA:CPU's Eigen
                 pool keeps its default of one thread per core)
  cpu_f32_omp8   env, f32acc, devtime, compile
  gpu_f64        env, correctness, f32acc, devtime, compile   (--platform gpu)
  gpu_f32        env, f32acc, devtime, compile                (--platform gpu;
                 JAX's default matmul precision, i.e. TF32 on Ampere+)
  gpu_f32_hi     env, f32acc   (--matmul-precision highest: true float32 dots,
                 the leg the float32 route-accuracy comparison is read from)

Thread environment: --omp N sets OMP/MKL/OPENBLAS_NUM_THREADS (BLAS/LAPACK
custom calls: trsm, potrf, GEMM); --xla-threads N sets NPROC, which is what
sizes XLA:CPU's Eigen intra-op thread pool (FFT, dot, fusions).  The omp legs
set both to the same N; cpu_f64_prod leaves NPROC unset.  section_env records
both and counts the pool's threads (tf_XLAEigen) so the JSON says which
configuration a CPU number belongs to.

Each child writes <out base>.<leg>.json; the parent embeds them in <out>.

Usage
-----
    python benchmarks/gs/prep_inputs.py --out benchmarks/gs/inputs
    python benchmarks/gs/bench_gs.py --platform gpu --inputs benchmarks/gs/inputs \
        --out benchmarks/gs/results/local.json
    python benchmarks/gs/bench_gs.py --platform cpu --smoke --out smoke.json
    python benchmarks/gs/analyze_gs.py smoke.json

Run from the repo root with PYTHONPATH=<repo> so the kit's ringdown is the
one measured.  No repository file outside benchmarks/gs is touched.
"""

# ---------------------------------------------------------------------------
# PHASE 0: everything that must happen BEFORE `import jax`
# ---------------------------------------------------------------------------
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)     # gs_kernels, ref_longdouble, variants, harness

ALL_SECTIONS = ["env", "correctness", "reference", "devtime", "compile",
                "f32acc", "nuts"]
# 'nuts' is never implied by the default: it must be listed explicitly
DEFAULT_SECTIONS = [s for s in ALL_SECTIONS if s != "nuts"]

_P = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_P.add_argument("--platform", default="cpu", choices=["cpu", "gpu"],
                help="backend for THIS process; the parent spawns gpu legs "
                     "only when this is gpu (default cpu)")
_P.add_argument("--x64", type=int, default=1, choices=[0, 1],
                help="jax_enable_x64 for THIS process: 1 = float64 (default)")
_P.add_argument("--omp", default="",
                help="set OMP/MKL/OPENBLAS_NUM_THREADS (BLAS/LAPACK threads) "
                     "before jax import; does NOT touch XLA's pool (see "
                     "--xla-threads)")
_P.add_argument("--xla-threads", default="",
                help="set NPROC before jax import: the size of XLA:CPU's Eigen "
                     "intra-op thread pool (FFT, dot, fusions); default: leave "
                     "unset = one thread per core, which is what production "
                     "ringdown (OMP_NUM_THREADS=1 only) runs with")
_P.add_argument("--cpu-omp", type=int, default=8,
                help="parent: thread count N of the cpu_f64_ompN / cpu_f32_ompN "
                     "legs (BLAS and XLA pool both N; default 8; the sbatch "
                     "passes SLURM_CPUS_PER_TASK)")
_P.add_argument("--inputs", default=os.path.join(_HERE, "inputs"),
                help="directory of prep_inputs.py npz files")
_P.add_argument("--sections", default="all",
                help="comma-separated subset of: %s (default: all except nuts)"
                     % ",".join(ALL_SECTIONS))
_P.add_argument("--configs", default="",
                help="restrict the size grid, e.g. '2,205,2;3,1024,8'")
_P.add_argument("--families", default="",
                help="comma-separated ACF families (default: all present)")
_P.add_argument("--timing-families", default="aligo02,expcos",
                help="families used for devtime/compile (default aligo02,expcos)")
_P.add_argument("--variants", default="",
                help="comma-separated variants (default: all)")
_P.add_argument("--nfft", default="pow2", choices=["pow2", "fast", "both"],
                help="FFT padding for the GS variants (default pow2 = PR #141)")
_P.add_argument("--out", default="results_gs.json", help="output JSON path")
_P.add_argument("--no-sub", action="store_true",
                help="run the sections in THIS process; do not spawn legs")
_P.add_argument("--leg", default="",
                help="leg name (set by the parent on child legs)")
_P.add_argument("--smoke", action="store_true",
                help="(2,205,2) only, families aligo02+white, 3 points, "
                     "small rep counts, no nuts")
_P.add_argument("--ref", default="",
                help="path of the reference npz (default <out base>.ref.npz)")
_P.add_argument("--tag", default="", help="free-form label recorded in the JSON")
_P.add_argument("--npts", type=int, default=0,
                help="fixed points to use PER KIND (kind 0 = N(0,1) draws, kind 1 = "
                     "NUTS warmup samples; see prep_inputs.py pts_kind); default: "
                     "all stored points (30 = 20 + 10), smoke 3 + 3")
_P.add_argument("--cell-budget", type=float, default=0.0,
                help="soft per-cell devtime budget in s (default 25, smoke 6)")
_P.add_argument("--target-s", type=float, default=0.0,
                help="target wall time of the shorter timing loop (default "
                     "0.15, smoke 0.05)")
_P.add_argument("--nuts-variants", default="main,main_hoisted,gs_half,gemm_linv",
                help="variants for the nuts section")
_P.add_argument("--nuts-n", type=int, default=300,
                help="NUTS warmup and sample counts (default 300)")
_P.add_argument("--nuts-seeds", default="0,1",
                help="PRNG seeds of the WARM NUTS runs (one chain each, run "
                     "after a cold run with seed 0 that pays the compile); "
                     "default 0,1")
_P.add_argument("--gs-coeffs", default="pr", choices=["pr", "refined"],
                help="Yule-Walker filter fed to gs_pr/gs_full/gs_half: 'pr' = "
                     "scipy solve_toeplitz as PR #141 (default); 'refined' = "
                     "longdouble Levinson rounded to f64 (variants.build_consts). "
                     "The correctness/f32acc sections always ALSO evaluate the "
                     "other policy as a diagnostic (same executable)")
_P.add_argument("--spectra-from", default="f64", choices=["f64", "leg"],
                help="float32 legs: spectra from an f64 rfft cast to complex64 "
                     "(default) or from an rfft of the f32-cast filter ('leg', "
                     "as gs_pr_ascoded does in-model); f32acc also evaluates "
                     "the other policy as a diagnostic")
_P.add_argument("--ref-logdet", default="cholesky", choices=["cholesky", "levinson"],
                help="log|C| of the extended-precision reference: longdouble "
                     "dense Cholesky (default; ~1 s at N=1024, ~80 s at N=4096 "
                     "per detector) or the longdouble Levinson sum (weakly "
                     "stable: 2e-8 nats off at cond 3.5e11)")
_P.add_argument("--leg-timeout", type=float, default=3600.0,
                help="wall-clock timeout per child leg in s")
_P.add_argument("--parent-pid", type=int, default=0,
                help="GPU parent PID to ignore when counting foreign processes")
_P.add_argument("--matmul-precision", default="",
                help="jax_default_matmul_precision (e.g. 'highest'); default "
                     "leaves JAX's default, which on Ampere+ GPUs allows TF32 "
                     "for float32 matmuls, as production does")
ARGS = _P.parse_args()

if ARGS.omp:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = ARGS.omp
if ARGS.xla_threads:
    # jaxlib's XLA:CPU client sizes its Eigen intra-op pool from NPROC
    # (checked on jaxlib 0.11.1: 1/8/32 tf_XLAEigen threads for NPROC=1/8/unset
    # on a 32-CPU box); OMP_NUM_THREADS does not affect it
    os.environ["NPROC"] = ARGS.xla_threads

if ARGS.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"          # hard pin, before import jax
else:
    # 'cuda' makes JAX raise instead of silently falling back to CPU when the
    # plugin fails to initialize
    os.environ["JAX_PLATFORMS"] = "cuda"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json          # noqa: E402
import platform      # noqa: E402
import socket        # noqa: E402
import subprocess    # noqa: E402
import time          # noqa: E402
import traceback     # noqa: E402
import warnings      # noqa: E402

T_START = time.time()


def _default_leg_name():
    nm = "%s_f%d" % (ARGS.platform, 64 if ARGS.x64 else 32)
    if ARGS.platform == "cpu" and ARGS.omp:
        nm += "_omp%s" % ARGS.omp
    return nm


LEG = ARGS.leg or _default_leg_name()


def _die(msg, code=2):
    print("\n" + "=" * 78, file=sys.stderr)
    print("FATAL: " + msg, file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    sys.stderr.flush()
    sys.exit(code)


def _diagnostics():
    out = ["python      : %s" % sys.executable,
           "version     : %s" % sys.version.replace("\n", " "),
           "hostname    : %s" % socket.gethostname(),
           "JAX_PLATFORMS = %r" % os.environ.get("JAX_PLATFORMS"),
           "CUDA_VISIBLE_DEVICES = %r" % os.environ.get("CUDA_VISIBLE_DEVICES")]
    try:
        out.append(subprocess.run(["nvidia-smi"], capture_output=True,
                                  text=True, timeout=60).stdout)
    except Exception as e:                        # pragma: no cover
        out.append("nvidia-smi -> %r" % e)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# PHASE 1: import jax and verify backend AND precision
# ---------------------------------------------------------------------------
try:
    from jax import config as _jc
    _jc.update("jax_enable_x64", bool(ARGS.x64))
    if ARGS.matmul_precision:
        _jc.update("jax_default_matmul_precision", ARGS.matmul_precision)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import potential_energy
    from scipy.linalg import toeplitz
    import harness as H
    import variants as V
    import ref_longdouble as RL
except Exception:
    print(_diagnostics(), file=sys.stderr)
    traceback.print_exc()
    _die("import failed (see diagnostics above)")

_DEVS = jax.devices()
_BACKEND = jax.default_backend()
_IS_GPU = any(getattr(d, "platform", "") in ("gpu", "cuda") for d in _DEVS)
if ARGS.platform == "gpu" and not _IS_GPU:
    print(_diagnostics(), file=sys.stderr)
    _die("--platform gpu requested but jax.devices() = %r (backend %r)."
         % (_DEVS, _BACKEND))
if ARGS.platform == "cpu" and _IS_GPU:
    _die("--platform cpu requested but a GPU device is visible: %r" % (_DEVS,))
if bool(jax.config.jax_enable_x64) != bool(ARGS.x64):
    _die("jax_enable_x64 is %r but --x64 %d was requested"
         % (jax.config.jax_enable_x64, ARGS.x64))

DT = np.float64 if ARGS.x64 else np.float32

# ---------------------------------------------------------------------------
# run matrix
# ---------------------------------------------------------------------------
FULL_CONFIGS = [(2, 205, 2), (2, 410, 2), (2, 1024, 2), (2, 2048, 4),
                (3, 1024, 8), (2, 4096, 4)]
BIG_N = 4096            # only the 'big' legs run configs with N >= BIG_N
PROD = (2, 205, 2)
ALL_FAMILIES = ["aligo02", "aligo2", "aligo20", "expcos", "white", "gw150914"]
GATE_FAMILIES = ["white", "expcos"]      # algebra gate 1e-11 vs main
TOL_GATE = 1e-11
TOL_CONCERN, TOL_FAIL = 1e-8, 1e-6      # vs the extended-precision reference

# leg table: argv for the child, its sections, and whether it runs BIG_N.
# N = --cpu-omp is substituted into the multi-thread CPU legs' names and argv.
def leg_table(cpu_omp):
    n = str(int(cpu_omp))
    return {
        "cpu_f64_omp" + n: dict(argv=["--platform", "cpu", "--x64", "1", "--omp", n,
                                      "--xla-threads", n],
                                sections=["env", "reference", "correctness", "devtime",
                                          "compile", "f32acc", "nuts"], big=True),
        "cpu_f64_omp1": dict(argv=["--platform", "cpu", "--x64", "1", "--omp", "1",
                                   "--xla-threads", "1"],
                             sections=["env", "correctness", "devtime", "compile"],
                             big=False),
        # production configuration: BLAS 1 thread, XLA:CPU pool at its default
        "cpu_f64_prod": dict(argv=["--platform", "cpu", "--x64", "1", "--omp", "1"],
                             sections=["env", "devtime", "compile"], big=False),
        "cpu_f32_omp" + n: dict(argv=["--platform", "cpu", "--x64", "0", "--omp", n,
                                      "--xla-threads", n],
                                sections=["env", "f32acc", "devtime", "compile"],
                                big=False),
        "gpu_f64": dict(argv=["--platform", "gpu", "--x64", "1"],
                        sections=["env", "correctness", "f32acc", "devtime",
                                  "compile"], big=True),
        "gpu_f32": dict(argv=["--platform", "gpu", "--x64", "0"],
                        sections=["env", "f32acc", "devtime", "compile"], big=True),
        # same as gpu_f32 with true float32 dots (no TF32): the leg the f32
        # route-accuracy comparison is drawn from; timing is unaffected (<3%)
        # so only the accuracy section is run
        "gpu_f32_hi": dict(argv=["--platform", "gpu", "--x64", "0",
                                 "--matmul-precision", "highest"],
                           sections=["env", "f32acc"], big=True),
    }


LEG_TABLE = leg_table(ARGS.cpu_omp)
LEG_ORDER = list(LEG_TABLE)


def _parse_configs(s):
    return [tuple(int(x) for x in c.split(",")) for c in s.split(";") if c.strip()]


if ARGS.smoke:
    CONFIGS = [PROD]
elif ARGS.configs:
    CONFIGS = _parse_configs(ARGS.configs)
else:
    big = LEG_TABLE.get(LEG, {}).get("big", True)
    CONFIGS = [c for c in FULL_CONFIGS if big or c[1] < BIG_N]

if ARGS.smoke:
    FAMILIES = ["aligo02", "white"]
elif ARGS.families:
    FAMILIES = [f.strip() for f in ARGS.families.split(",") if f.strip()]
else:
    FAMILIES = list(ALL_FAMILIES)
TIMING_FAMILIES = [f for f in ARGS.timing_families.split(",")
                   if f.strip() and f.strip() in FAMILIES] or FAMILIES[:1]
if ARGS.smoke:
    TIMING_FAMILIES = ["aligo02"]

VARIANTS = ([v.strip() for v in ARGS.variants.split(",") if v.strip()]
            if ARGS.variants else list(V.VARIANTS))
for _v in VARIANTS:
    if _v not in V.VARIANTS:
        _die("unknown variant %r (known: %s)" % (_v, ", ".join(V.VARIANTS)))
if "floor" not in VARIANTS:
    VARIANTS.append("floor")     # U_floor is needed for every comparison
NFFT_MODES = ["pow2", "fast"] if ARGS.nfft == "both" else [ARGS.nfft]

NPTS = ARGS.npts or (3 if ARGS.smoke else 0)   # 0 = every stored point
CELL_BUDGET = ARGS.cell_budget or (6.0 if ARGS.smoke else 25.0)
TARGET_S = ARGS.target_s or (0.05 if ARGS.smoke else 0.15)
REP = 3 if ARGS.smoke else 5

if ARGS.sections == "all":
    SECTIONS = list(DEFAULT_SECTIONS)
else:
    SECTIONS = [s.strip() for s in ARGS.sections.split(",") if s.strip()]
for _s in SECTIONS:
    if _s not in ALL_SECTIONS:
        _die("unknown section %r" % _s)


def want(name):
    return name in SECTIONS


def cfg_key(c):
    return "%d,%d,%d" % tuple(c)


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
_INPUT_CACHE = {}


def inputs_path(cfg, fam):
    return os.path.join(ARGS.inputs, "%s_d%d_n%d_m%d.npz" % ((fam,) + tuple(cfg)))


def load_inputs(cfg, fam):
    """Dict of numpy arrays for (config, family), or None if the file is
    missing (logged once)."""
    key = (tuple(cfg), fam)
    if key in _INPUT_CACHE:
        return _INPUT_CACHE[key]
    path = inputs_path(cfg, fam)
    if not os.path.exists(path):
        log("  [missing inputs %s -- cell skipped]" % path)
        _INPUT_CACHE[key] = None
        return None
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    _INPUT_CACHE[key] = d
    return d


def kw_for(inp):
    kw = dict(V.KW_DEFAULT)
    if "a_scale_max" in inp:
        kw["a_scale_max"] = float(inp["a_scale_max"])
    return kw


def modes_for(cfg):
    return V.modes_of(cfg[2])


KIND_NAMES = {0: "normal", 1: "warmup"}   # prep_inputs.py pts_kind


def point_kinds_all(inp):
    """pts_kind of every stored point (zeros if the inputs predate the key)."""
    sites = [k[len("pts|"):] for k in inp if k.startswith("pts|")]
    if not sites:
        raise KeyError("inputs carry no 'pts|<site>' arrays")
    navail = int(inp["pts|" + sites[0]].shape[0])
    if "pts_kind" in inp:
        kinds = np.asarray(inp["pts_kind"]).astype(int).reshape(-1)
        if kinds.shape[0] != navail:
            raise ValueError("pts_kind has %d entries for %d points" % (kinds.shape[0], navail))
    else:
        kinds = np.zeros(navail, dtype=int)
    return sites, kinds


def point_indices(inp, npts=None):
    """Indices of the stored points to use: the first `npts` of EACH kind
    (npts = 0: all of them), in stored order."""
    npts = NPTS if npts is None else npts
    _, kinds = point_kinds_all(inp)
    if npts <= 0:
        return list(range(kinds.shape[0]))
    idx = []
    for kd in sorted(set(kinds.tolist())):
        idx.extend(np.flatnonzero(kinds == kd)[:npts].tolist())
    return sorted(idx)


def fixed_points(inp, npts=None):
    """The stored unconstrained points, as a list of {site: jnp array (DT)};
    the selection is point_indices (the first npts of each kind)."""
    sites, _ = point_kinds_all(inp)
    return [{s: jnp.asarray(inp["pts|" + s][j], dtype=DT) for s in sites}
            for j in point_indices(inp, npts)]


def fixed_point_kinds(inp, npts=None):
    """pts_kind of the points fixed_points returns (same selection)."""
    _, kinds = point_kinds_all(inp)
    return [int(kinds[j]) for j in point_indices(inp, npts)]


def cond_of(inp):
    try:
        return float(np.max(np.asarray(inp["cond"], dtype=np.float64)))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
_FN_CACHE = {}


def vg_fn(variant, cfg, nfft_mode, kw):
    """jit(value_and_grad) of the potential with data AND constants as
    arguments: fn(q, consts, data) with data = (times, strains, fps, fcs).
    One compile per (variant, config, nfft mode, prior bounds); every family
    of the same config reuses it."""
    key = (variant, tuple(cfg), nfft_mode, tuple(sorted(kw.items())))
    if key not in _FN_CACHE:
        model = V.make_model(variant, modes_for(cfg), nfft_mode=nfft_mode, **kw)

        def U(q, consts, data):
            times, strains, fps, fcs = data
            return potential_energy(model, (times, strains, consts, fps, fcs), {}, q)
        _FN_CACHE[key] = jax.jit(jax.value_and_grad(U))
    return _FN_CACHE[key]


def consts_for(variant, inp, nfft_mode, coeffs=None, spectra_from=None):
    """build_consts with the run's policies (--gs-coeffs, --spectra-from)
    unless overridden."""
    return V.build_consts(variant, inp, DT, nfft_mode,
                          coeffs=coeffs or ARGS.gs_coeffs,
                          spectra_from=spectra_from or ARGS.spectra_from)


def eval_points(variant, cfg, inp, nfft_mode, pts, coeffs=None, spectra_from=None):
    """[(U, {site: grad as f64 numpy}), ...] at the given points.  The
    compiled function is shared; only the constants change with the
    coeffs / spectra_from policy overrides."""
    kw = kw_for(inp)
    fn = vg_fn(variant, cfg, nfft_mode, kw)
    consts = consts_for(variant, inp, nfft_mode, coeffs, spectra_from)
    data = V.data_args(inp, DT)
    out = []
    for p in pts:
        U, g = fn(p, consts, data)
        out.append((float(U), {k: np.asarray(v, dtype=np.float64) for k, v in g.items()}))
    return out


def lik_part(vals, floor_vals):
    """Likelihood part: U_lik = U - U_floor, g_lik = g - g_floor (float64)."""
    out = []
    for (U, g), (Uf, gf) in zip(vals, floor_vals):
        out.append((U - Uf, {k: g[k] - gf[k] for k in g}))
    return out


def rel_grad_err(g, g_ref, scale=None):
    """max over sites of max|g - g_ref| / max|g_ref| (per site), and the
    per-site dict.

    With `scale` = {site: s} the denominator is that fixed per-site scale
    instead of the point's own max|g_ref[site]| ("cloud normalization", see
    `cloud_scale`).  The per-point normalization is ill-conditioned wherever
    a scalar site's likelihood gradient is near zero (the NUTS warmup points
    sit near the mode: |g_chi| down to 0.02 against ~70 over the cloud), which
    is what made the 1e-11 algebra gate fail on white/expcos at k = 32 /
    N = 2048 for pure-roundoff differences of 1e-13 of the gradient scale.
    """
    per = {}
    for k in g_ref:
        den = (float(scale[k]) if scale is not None
               else float(np.max(np.abs(g_ref[k])))) + 1e-300
        per[k] = float(np.max(np.abs(g[k] - g_ref[k])) / den)
    return (max(per.values()) if per else float("nan")), per


def cloud_scale(grads):
    """{site: max over the point cloud of max|g[site]|} for a list of
    per-point gradient dicts: the per-site scale of the cloud-normalized
    errors."""
    out = {}
    for g in grads:
        for k, v in g.items():
            out[k] = max(out.get(k, 0.0), float(np.max(np.abs(v))))
    return out


EPS64 = float(np.finfo(np.float64).eps)


def err_over_eps_cond(err, cond, eps=EPS64):
    """Digits lost beyond conditioning: err / (eps cond(C)).  ~1 means the
    route is as accurate as the conditioning allows; >> 10 means it loses
    accuracy the problem does not force it to lose."""
    if err is None or not np.isfinite(cond) or cond <= 0:
        return None
    return float(err / (eps * cond))


def variant_modes(variant):
    return NFFT_MODES if variant in V.GS_VARIANTS else ["pow2"]


def vkey(variant, mode):
    """Result key: GS variants are suffixed with the nfft mode when the run
    uses more than one mode."""
    if variant in V.GS_VARIANTS and len(NFFT_MODES) > 1:
        return "%s@%s" % (variant, mode)
    return variant


# ---------------------------------------------------------------------------
# SECTION: environment
# ---------------------------------------------------------------------------
def xla_cpu_pool_threads():
    """Number of XLA:CPU Eigen pool threads (tf_XLAEigen*) in this process
    after a small warm-up op, from /proc/self/task; None if unavailable."""
    try:
        jax.block_until_ready(jnp.fft.rfft(jnp.ones(64, dtype=DT)))
        n = 0
        for t in os.listdir("/proc/self/task"):
            with open("/proc/self/task/%s/comm" % t) as fh:
                if fh.read().startswith("tf_XLAEigen"):
                    n += 1
        return n
    except Exception:
        return None


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
        # None means JAX's default: TF32 is allowed for f32 matmuls on
        # Ampere+ GPUs, which matters for the f32 accuracy tables
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "matmul_note": ("TF32 allowed for float32 dots (JAX default on Ampere+ GPUs)"
                        if _IS_GPU and not ARGS.x64 and not ARGS.matmul_precision
                        else "true float32/float64 dots"),
        "dtype": str(np.dtype(DT)),
        "gs_coeffs": ARGS.gs_coeffs,
        "spectra_from": ARGS.spectra_from,
        "ref_logdet": ARGS.ref_logdet,
        # jax lowers jnp.fft to the HLO fft op; XLA:CPU implements it with
        # DUCC (pocketfft's successor) and XLA:GPU with cuFFT
        "fft_backend_hint": "cufft" if _IS_GPU else "ducc",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "sys_prefix": sys.prefix,
        "uname": platform.platform(),
        "smoke": ARGS.smoke,
        "configs": [list(c) for c in CONFIGS],
        "families": FAMILIES,
        "timing_families": TIMING_FAMILIES,
        "variants": VARIANTS,
        "nfft_modes": NFFT_MODES,
        "npts": NPTS,
        "sections": SECTIONS,
        "inputs_dir": os.path.abspath(ARGS.inputs),
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
                          "OPENBLAS_NUM_THREADS", "NPROC", "XLA_FLAGS",
                          "JAX_PLATFORMS")}
    # XLA:CPU's Eigen intra-op pool (sized by NPROC, default one per core):
    # the thread count that FFT / dot / fusion parallelism actually sees
    env["xla_cpu_pool_threads"] = xla_cpu_pool_threads()
    # which evaluation of log|C| the inputs carry (variants.logdetC_source)
    env["logdetC_source"] = None
    for cfg in CONFIGS:
        for fam in FAMILIES:
            inp = load_inputs(cfg, fam)
            if inp is not None:
                env["logdetC_source"] = V.logdetC_source(inp)
                break
        if env["logdetC_source"]:
            break
    import ringdown as _rd
    _rd_dir = os.path.dirname(os.path.dirname(os.path.abspath(_rd.__file__)))
    _kit_repo = os.path.dirname(os.path.dirname(_HERE))
    env["ringdown_path"] = os.path.abspath(_rd.__file__)
    env["ringdown_root"] = _rd_dir
    env["kit_root"] = _kit_repo
    env["ringdown_matches_kit"] = os.path.realpath(_rd_dir) == \
        os.path.realpath(_kit_repo)
    try:
        _g = subprocess.run(["git", "-C", _rd_dir, "describe", "--always",
                             "--dirty", "--tags"],
                            capture_output=True, text=True, timeout=30)
        env["ringdown_git"] = _g.stdout.strip() or None
    except Exception:
        env["ringdown_git"] = None
    env["slurm"] = {k: os.environ[k] for k in sorted(os.environ)
                    if k.startswith("SLURM_") and k in (
                        "SLURM_JOB_ID", "SLURM_JOBID", "SLURM_CPUS_ON_NODE",
                        "SLURM_CPUS_PER_TASK", "SLURM_GPUS_ON_NODE",
                        "SLURM_JOB_GPUS", "SLURM_JOB_PARTITION",
                        "SLURM_NODELIST", "SLURM_MEM_PER_NODE")}
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
              "nvidia-cusolver-cu12", "nvidia-cufft-cu12",
              "nvidia-cuda-runtime-cu12", "ringdown"):
        try:
            pkgs[p] = md.version(p)
        except Exception:
            pkgs[p] = None
    env["packages"] = pkgs
    env["contention_before"] = (H.gpu_contention(ARGS.parent_pid or None)
                                if _IS_GPU else {})
    try:
        env["loadavg"] = open("/proc/loadavg").read().strip()
    except Exception:
        pass
    log("\n=== ENVIRONMENT [leg %s] ===" % LEG)
    for k in ("hostname", "jax_backend", "jax_devices", "jax_device_kinds",
              "jax_enable_x64", "jax_default_matmul_precision", "dtype",
              "fft_backend_hint", "cpu_model",
              "cpu_affinity_count", "nvidia_smi", "configs", "families"):
        log("  %-20s %s" % (k, env.get(k)))
    log("  %-20s %s" % ("threads", env["thread_env"]))
    log("  %-20s %s" % ("xla_cpu_pool", env["xla_cpu_pool_threads"]))
    log("  %-20s %s" % ("matmul", env["matmul_note"]))
    log("  %-20s gs_coeffs=%s spectra_from=%s logdetC_source=%s"
        % ("policies", ARGS.gs_coeffs, ARGS.spectra_from, env["logdetC_source"]))
    if env["logdetC_source"] == "f64lev":
        log("  *** WARNING: inputs predate the longdouble log|C|; run "
            "prep_inputs.py --refresh-precompute (nats columns of the hoisted "
            "variants are biased by ~eps cond N) ***")
    log("  %-20s %s (%s)" % ("ringdown", env["ringdown_root"], env["ringdown_git"]))
    if not env["ringdown_matches_kit"]:
        log("  *** WARNING: the imported ringdown is NOT the tree this kit lives in ***")
    if _IS_GPU:
        nfor = env["contention_before"].get("n_foreign_compute_apps")
        log("  %-20s %s" % ("gpu_state", env["contention_before"].get("gpu_state")))
        if nfor:
            log("  *** WARNING: %d OTHER compute process(es) on this GPU ***" % nfor)
    return env


# ---------------------------------------------------------------------------
# SECTION: extended-precision reference (cpu float64 leg only)
# ---------------------------------------------------------------------------
def ref_keys(cfg, fam, j):
    return "%s|%s|pt%d|" % (cfg_key(cfg), fam, j)


def section_reference(ref_path):
    """Longdouble one-shot likelihood and closed-form gradient at the fixed
    points, pulled back through jax.vjp of the design-matrix head; plus this
    leg's float64 U_lik / grad per variant (the 'twin' for the f32 legs).

    Stored keys (npz):  <cfg>|<fam>|pt<j>|z|<site>      the point
                        <cfg>|<fam>|pt<j>|refU          [U_lik_ref, lo] where
                              U_lik_ref = -loglike (float64) and lo the
                              longdouble remainder
                        <cfg>|<fam>|pt<j>|refg|<site>   grad of U_lik_ref
                        <cfg>|<fam>|pt<j>|U|<variant>   [U_lik] this leg
                        <cfg>|<fam>|pt<j>|g|<variant>|<site>
                        meta|leg                        producer leg name
    """
    log("\n=== EXTENDED-PRECISION REFERENCE [leg %s] -> %s ===" % (LEG, ref_path))
    if DT != np.float64 or _IS_GPU:
        log("  reference is produced by the cpu float64 leg only -- skipped")
        return {"skipped": "not the cpu float64 leg"}
    blob = {"meta|leg": np.array([LEG])}
    summary = {}
    for cfg in CONFIGS:
        ck = cfg_key(cfg)
        summary[ck] = {}
        for fam in FAMILIES:
            t0 = time.perf_counter()
            inp = load_inputs(cfg, fam)
            if inp is None:
                summary[ck][fam] = {"error": "missing inputs"}
                continue
            try:
                s = _reference_cell(cfg, fam, inp, blob)
                s["wall_s"] = time.perf_counter() - t0
                summary[ck][fam] = s
                log("  %-12s %-9s cond=%.2e  n_pts=%d  refU(pt0)=%.6f  "
                    "logdet_ld-exact=%.2e  ld_chol-ld_lev=%.1e  %.1fs (logdet %.1fs)"
                    % (ck, fam, s["cond"], s["n_pts"], s["refU_pt0"],
                       s["logdet_ld_minus_exact"],
                       max(abs(x) for x in s["logdet_ld_chol_minus_lev"]),
                       s["wall_s"], s["logdet_ld_wall_s"]))
            except Exception as e:
                traceback.print_exc()
                summary[ck][fam] = {"error": repr(e)}
                log("  %-12s %-9s FAILED: %r" % (ck, fam, e))
    np.savez(ref_path, **blob)
    log("  -> %s (%d arrays)" % (ref_path, len(blob)))
    return {"path": os.path.abspath(ref_path), "npts_per_kind": NPTS, "summary": summary}


def _reference_cell(cfg, fam, inp, blob):
    n_det, N, n_modes = cfg
    kw = kw_for(inp)
    modes = modes_for(cfg)
    pts = fixed_points(inp)
    times, strains, fps, fcs = V.data_args(inp, np.float64)
    acf = np.asarray(inp["acf"], dtype=np.float64)
    L_list = [np.asarray(inp["L"][i], dtype=np.float64) for i in range(n_det)]
    C_list = [toeplitz(acf[i]) for i in range(n_det)]
    y_list = [np.asarray(inp["strains"][i], dtype=np.float64) for i in range(n_det)]
    # exact log|C| in longdouble: dense Cholesky by default (backward error
    # ~ N eps_ld, no cond amplification), the Levinson sum as a cross-check
    # (weakly stable: -2e-11 nats vs Cholesky at cond 5e8, -2e-8 at 3.5e11);
    # both compared with the inputs' constant and with 2 sum log diag L (f64)
    logdet_lev = [RL.logdet_levinson_ld(acf[i]) for i in range(n_det)]
    if ARGS.ref_logdet == "cholesky":
        t_ld = time.perf_counter()
        logdet_ld = [RL.logdet_cholesky_ld(C_list[i]) for i in range(n_det)]
        t_ld = time.perf_counter() - t_ld
    else:
        logdet_ld, t_ld = logdet_lev, 0.0
    logdet_exact = np.asarray(inp["logdetC_exact"], dtype=np.float64)
    logdet_chol = np.array([2.0 * np.sum(np.log(np.diag(L))) for L in L_list])
    head = V.head_design_matrices(modes, **kw)

    def dms_only(z):
        return head(z, times, fps, fcs)[0]

    refU0 = None
    for j, z in enumerate(pts):
        dms, _ = head(z, times, fps, fcs)
        M_list = [np.asarray(dms[i], dtype=np.float64) for i in range(n_det)]
        r = RL.oneshot_core_ld(M_list, y_list, L_list, C_list, logdet_ld)
        ll = r["loglike"]
        hi = np.float64(ll)
        lo = np.float64(ll - np.longdouble(hi))
        # U_lik = -loglike; gradient of U_lik = -(dlogL/dM pulled back)
        G = np.stack([np.asarray(g, dtype=np.float64) for g in r["dlogL_dM"]])
        _, vjp = jax.vjp(dms_only, z)
        (gz,) = vjp(jnp.asarray(G, dtype=np.float64))
        pre = ref_keys(cfg, fam, j)
        for site, val in z.items():
            blob[pre + "z|" + site] = np.asarray(val, dtype=np.float64)
        blob[pre + "refU"] = np.array([-hi, -lo])
        for site, val in gz.items():
            blob[pre + "refg|" + site] = -np.asarray(val, dtype=np.float64)
        if refU0 is None:
            refU0 = float(-hi)
    # this leg's float64 values per variant (the f32 twin)
    floor_vals = eval_points("floor", cfg, inp, "pow2", pts)
    for variant in VARIANTS:
        if variant == "floor":
            continue
        for mode in variant_modes(variant):
            vals = lik_part(eval_points(variant, cfg, inp, mode, pts), floor_vals)
            vk = vkey(variant, mode)
            for j, (U, g) in enumerate(vals):
                pre = ref_keys(cfg, fam, j)
                blob[pre + "U|" + vk] = np.array([U])
                for site, val in g.items():
                    blob[pre + "g|" + vk + "|" + site] = val
    kinds = fixed_point_kinds(inp)
    return {"cond": cond_of(inp), "n_pts": len(pts),
            "n_pts_by_kind": {KIND_NAMES.get(k, "kind%d" % k): kinds.count(k)
                              for k in sorted(set(kinds))},
            "refU_pt0": refU0,
            "logdet_ld": [float(x) for x in logdet_ld],
            "logdet_ld_method": ARGS.ref_logdet, "logdet_ld_wall_s": t_ld,
            "logdet_ld_levinson": [float(x) for x in logdet_lev],
            # the reference's own log-det uncertainty: Cholesky vs Levinson in longdouble
            "logdet_ld_chol_minus_lev": [float(a - b) for a, b in zip(logdet_ld, logdet_lev)],
            "logdetC_source": V.logdetC_source(inp),
            "logdet_ld_minus_exact": float(np.max(np.abs(
                np.array([float(x) for x in logdet_ld]) - logdet_exact))),
            "logdet_ld_minus_chol": float(np.max(np.abs(
                np.array([float(x) for x in logdet_ld]) - logdet_chol))),
            "logdet_ld_minus_f64lev": float(np.max(np.abs(
                np.array([float(x) for x in logdet_ld])
                - np.asarray(inp["logdetC_f64lev"], dtype=np.float64))))
            if "logdetC_f64lev" in inp else None,
            "logdet_exact_minus_pr": [float(x) for x in
                                      (logdet_exact - np.asarray(inp["logdetC_pr"]))]
            if "logdetC_pr" in inp else None}


def load_ref(ref_path):
    if not (ref_path and os.path.exists(ref_path)):
        return None
    with np.load(ref_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def ref_values(blob, cfg, fam, j):
    """(U_ref, {site: g_ref}) or None when the point is not in the blob."""
    pre = ref_keys(cfg, fam, j)
    if pre + "refU" not in blob:
        return None
    U = float(blob[pre + "refU"][0])
    g = {k[len(pre + "refg|"):]: blob[k] for k in blob if k.startswith(pre + "refg|")}
    return U, g


def twin_values(blob, cfg, fam, j, vk):
    pre = ref_keys(cfg, fam, j)
    if pre + "U|" + vk not in blob:
        return None
    U = float(blob[pre + "U|" + vk][0])
    p2 = pre + "g|" + vk + "|"
    g = {k[len(p2):]: blob[k] for k in blob if k.startswith(p2)}
    return U, g


def check_points_match(blob, cfg, fam, pts):
    """The fixed points must be bit-identical to the ones the reference
    used (both come from the inputs npz; this guards against a stale ref)."""
    for j, z in enumerate(pts):
        pre = ref_keys(cfg, fam, j)
        for site, val in z.items():
            k = pre + "z|" + site
            if k in blob and not np.array_equal(
                    np.asarray(val, dtype=np.float64),
                    blob[k].astype(np.float64)):
                # in float32 legs the point is the f32 cast of the stored f64
                if DT == np.float32 and np.array_equal(
                        np.asarray(val, dtype=np.float32),
                        blob[k].astype(np.float32)):
                    continue
                return False
    return True


# ---------------------------------------------------------------------------
# SECTION: correctness (float64: algebra gate vs main, flags vs reference)
# ---------------------------------------------------------------------------
def section_correctness(ref_path):
    log("\n=== CORRECTNESS [leg %s]: likelihood part U_var - U_floor ===" % LEG)
    log("    gate %.0e (relative potential and CLOUD-normalized gradient) vs 'main' on %s;"
        % (TOL_GATE, "/".join(GATE_FAMILIES)))
    log("    vs reference: concerning > %.0e, fail > %.0e (cloud-normalized max rel. "
        "gradient error; per-point numbers kept as *_perpoint)" % (TOL_CONCERN, TOL_FAIL))
    log("    table cells: vs main / vs ref, both cloud-normalized")
    if DT != np.float64:
        log("  correctness is a float64 algebra check -- skipped in this leg")
        return {"skipped": "float32 leg"}
    blob = load_ref(ref_path)
    if blob is None:
        log("  (no reference npz at %r: comparing against main only)" % ref_path)
    res, gate_all = {}, True
    for cfg in CONFIGS:
        ck = cfg_key(cfg)
        res[ck] = {}
        for fam in FAMILIES:
            inp = load_inputs(cfg, fam)
            if inp is None:
                res[ck][fam] = {"error": "missing inputs"}
                continue
            try:
                cell = _correctness_cell(cfg, fam, inp, blob)
            except Exception as e:
                traceback.print_exc()
                cell = {"error": repr(e)}
            res[ck][fam] = cell
            if "error" in cell:
                log("  %-12s %-9s FAILED: %s" % (ck, fam, cell["error"]))
                continue
            line = "  %-12s %-9s cond=%.1e" % (ck, fam, cell["_cond"])
            for vk, r in cell.items():
                if vk.startswith("_"):
                    continue
                if "error" in r:
                    line += "  %s=ERR" % vk
                    continue
                tag = ""
                if r.get("gate") is not None:
                    tag = "OK" if r["gate"] else "FAIL"
                    gate_all &= bool(r["gate"])
                elif r.get("flag"):
                    tag = r["flag"]
                line += "  %s=%.1e/%.1e[%s]" % (vk, r["rel_grad_vs_main_cloud"],
                                                r.get("rel_grad_vs_ref_cloud")
                                                if r.get("rel_grad_vs_ref_cloud") is not None
                                                else float("nan"), tag)
            log(line)
            alt = ["%s:%s" % (vk, "%.1e" % r["alt_coeffs"]["rel_grad_vs_ref_cloud"]
                              if r["alt_coeffs"].get("rel_grad_vs_ref_cloud") is not None
                              else "-")
                   for vk, r in cell.items()
                   if not vk.startswith("_") and "error" not in r and r.get("alt_coeffs")]
            if alt:
                pol = [r["alt_coeffs"]["policy"] for vk, r in cell.items()
                       if not vk.startswith("_") and "error" not in r and r.get("alt_coeffs")][0]
                log("  %-12s %-9s same executable with %s coefficients, vs ref (cloud): %s"
                    % ("", "", pol, "  ".join(alt)))
            by_kind = cell.get("_n_pts_by_kind") or {}
            if len(by_kind) > 1:
                # max over the non-main variants, per point kind: vs main / vs ref
                parts = []
                for kn in sorted(by_kind):
                    em = max([r["rel_grad_vs_main_cloud_by_kind"].get(kn, 0.0)
                              for vk, r in cell.items()
                              if not vk.startswith("_") and "error" not in r
                              and vk != "main"] or [float("nan")])
                    er = [r["rel_grad_vs_ref_cloud_by_kind"].get(kn)
                          for vk, r in cell.items()
                          if not vk.startswith("_") and "error" not in r
                          and vk != "main" and r["rel_grad_vs_ref_cloud_by_kind"].get(kn) is not None]
                    parts.append("%s(n=%d): main=%.1e ref=%s" % (
                        kn, by_kind[kn], em, ("%.1e" % max(er)) if er else "-"))
                log("  %-12s %-9s by kind [max over non-main variants, cloud]: %s"
                    % ("", "", "  ".join(parts)))
    res["_all_gate_pass"] = gate_all
    if not gate_all:
        log("  *** ALGEBRA GATE FAILED -- see the table above ***")
    return res


def _compare(vals, main_vals, refs, kind_names, scale_main, scale_ref, cond):
    """Every accuracy metric of one variant's per-point likelihood values
    against main's and the reference's (see rel_grad_err for the two
    normalizations)."""
    r = {"nats_vs_main": 0.0, "rel_pot_vs_main": 0.0,
         # per-point normalization (max|g_ref[site]| at the same point) --
         # kept as a diagnostic; ill-conditioned near the mode
         "rel_grad_vs_main": 0.0, "per_site_vs_main": {},
         "nats_vs_ref": None, "rel_grad_vs_ref": None, "per_site_vs_ref": {},
         # cloud normalization: |g - g_ref| / max over ALL points of max|g_ref[site]|
         "rel_grad_vs_main_cloud": 0.0, "per_site_vs_main_cloud": {},
         "rel_grad_vs_ref_cloud": None, "per_site_vs_ref_cloud": {},
         "all_finite": True,
         # per point kind (prep_inputs pts_kind: normal = N(0,1) draws,
         # warmup = NUTS typical-set samples)
         "rel_grad_vs_main_by_kind": {}, "rel_grad_vs_ref_by_kind": {},
         "rel_grad_vs_main_cloud_by_kind": {}, "rel_grad_vs_ref_cloud_by_kind": {},
         "nats_vs_ref_by_kind": {}}

    def bump(key, kn, val):
        r[key] = max(r[key] or 0.0, val)
        r[key + "_by_kind"][kn] = max(r[key + "_by_kind"].get(kn, 0.0), val)

    for j, ((U, g), (Um, gm)) in enumerate(zip(vals, main_vals)):
        kn = kind_names[j]
        if not (np.isfinite(U) and all(np.all(np.isfinite(v)) for v in g.values())):
            r["all_finite"] = False
        dU = abs(U - Um)
        r["nats_vs_main"] = max(r["nats_vs_main"], dU)
        r["rel_pot_vs_main"] = max(r["rel_pot_vs_main"], dU / (abs(Um) + 1e-300))
        e, per = rel_grad_err(g, gm)
        bump("rel_grad_vs_main", kn, e)
        for k, v in per.items():
            r["per_site_vs_main"][k] = max(r["per_site_vs_main"].get(k, 0.0), v)
        e, per = rel_grad_err(g, gm, scale_main)
        bump("rel_grad_vs_main_cloud", kn, e)
        for k, v in per.items():
            r["per_site_vs_main_cloud"][k] = max(r["per_site_vs_main_cloud"].get(k, 0.0), v)
        if refs[j] is not None:
            Ur, gr = refs[j]
            dUr = abs(U - Ur)
            r["nats_vs_ref"] = max(r["nats_vs_ref"] or 0.0, dUr)
            r["nats_vs_ref_by_kind"][kn] = max(r["nats_vs_ref_by_kind"].get(kn, 0.0), dUr)
            e2, per2 = rel_grad_err(g, gr)
            bump("rel_grad_vs_ref", kn, e2)
            for k, v in per2.items():
                r["per_site_vs_ref"][k] = max(r["per_site_vs_ref"].get(k, 0.0), v)
            e2, per2 = rel_grad_err(g, gr, scale_ref)
            bump("rel_grad_vs_ref_cloud", kn, e2)
            for k, v in per2.items():
                r["per_site_vs_ref_cloud"][k] = max(r["per_site_vs_ref_cloud"].get(k, 0.0), v)
    # digits lost beyond conditioning (checklist item 6), cloud-normalized
    r["err_over_eps_cond"] = err_over_eps_cond(r["rel_grad_vs_ref_cloud"], cond)
    return r


def _flag(e, finite):
    if e is None:
        return None
    return ("fail" if (e > TOL_FAIL or not finite)
            else "concerning" if e > TOL_CONCERN else "ok")


ALT_COEFFS_VARIANTS = ("gs_pr", "gs_full", "gs_half")   # gs_pr_ascoded: PR filter only


def _correctness_cell(cfg, fam, inp, blob):
    pts = fixed_points(inp)
    kinds = fixed_point_kinds(inp)
    kind_names = [KIND_NAMES.get(k, "kind%d" % k) for k in kinds]
    cond = cond_of(inp)
    floor_vals = eval_points("floor", cfg, inp, "pow2", pts)
    main_vals = lik_part(eval_points("main", cfg, inp, "pow2", pts), floor_vals)
    refs = [ref_values(blob, cfg, fam, j) if blob is not None else None
            for j in range(len(pts))]
    if blob is not None and not check_points_match(blob, cfg, fam, pts):
        refs = [None] * len(pts)
        log("  [reference points differ from inputs for %s/%s: ref ignored]"
            % (cfg_key(cfg), fam))
    scale_main = cloud_scale([g for _, g in main_vals])
    scale_ref = cloud_scale([r[1] for r in refs if r is not None]) or None
    cell = {"_cond": cond, "_n_pts": len(pts),
            "_n_pts_by_kind": {kn: kind_names.count(kn) for kn in sorted(set(kind_names))},
            "_has_ref": any(r is not None for r in refs),
            "_U_lik_main_by_kind": {},
            # per-site scales of the cloud normalization
            "_cloud_scale_main": scale_main, "_cloud_scale_ref": scale_ref,
            "_gate_normalization": "cloud",
            "_logdetC_source": V.logdetC_source(inp),
            "_gs_coeffs": ARGS.gs_coeffs}
    for kn in sorted(set(kind_names)):
        Us = [U for (U, _), k in zip(main_vals, kind_names) if k == kn]
        cell["_U_lik_main_by_kind"][kn] = {"min": float(min(Us)), "max": float(max(Us)),
                                           "pt0": float(Us[0])}
    for variant in VARIANTS:
        if variant == "floor":
            continue
        for mode in variant_modes(variant):
            vk = vkey(variant, mode)
            try:
                vals = (main_vals if variant == "main"
                        else lik_part(eval_points(variant, cfg, inp, mode, pts),
                                      floor_vals))
            except Exception as e:
                cell[vk] = {"error": repr(e)}
                continue
            r = _compare(vals, main_vals, refs, kind_names, scale_main, scale_ref, cond)
            # the algebra gate: cloud-normalized gradient and relative potential
            worst_main = max(r["rel_pot_vs_main"], r["rel_grad_vs_main_cloud"])
            worst_main_pp = max(r["rel_pot_vs_main"], r["rel_grad_vs_main"])
            if fam in GATE_FAMILIES and variant != "main":
                r["gate"] = bool(worst_main < TOL_GATE and r["all_finite"])
                r["gate_perpoint"] = bool(worst_main_pp < TOL_GATE and r["all_finite"])
            else:
                r["gate"] = r["gate_perpoint"] = None
            # flags vs the reference: cloud-normalized (headline) and per-point
            r["flag"] = _flag(r["rel_grad_vs_ref_cloud"], r["all_finite"])
            r["flag_perpoint"] = _flag(r["rel_grad_vs_ref"], r["all_finite"])
            # F1 diagnostic: the SAME executable fed the other Yule-Walker filter
            # (solve_toeplitz vs longdouble-refined), isolating the precompute's
            # forward error from the FFT route
            if variant in ALT_COEFFS_VARIANTS and "a_ld" in inp:
                other = "refined" if ARGS.gs_coeffs == "pr" else "pr"
                try:
                    vals2 = lik_part(eval_points(variant, cfg, inp, mode, pts, coeffs=other),
                                     floor_vals)
                    r2 = _compare(vals2, main_vals, refs, kind_names, scale_main, scale_ref, cond)
                    r["alt_coeffs"] = {"policy": other}
                    for k in ("rel_grad_vs_main", "rel_grad_vs_main_cloud", "rel_grad_vs_ref",
                              "rel_grad_vs_ref_cloud", "nats_vs_ref", "err_over_eps_cond",
                              "rel_grad_vs_ref_by_kind"):
                        r["alt_coeffs"][k] = r2[k]
                    r["alt_coeffs"]["flag"] = _flag(r2["rel_grad_vs_ref_cloud"], r2["all_finite"])
                except Exception as e:
                    r["alt_coeffs"] = {"policy": other, "error": repr(e)}
            cell[vk] = r
    # checklist item 8: scale invariance.  gw150914s1 is the gw150914 family
    # with scale = 1 (prep_inputs.py); at the same unconstrained point the
    # likelihood gradients must agree and U_lik must differ by exactly
    # n_det N log(scale) (log|C| shifts by -2 N log scale per detector, Q and
    # M^T C^{-1} M are invariant).  Evaluated with `main` on both files.
    if fam.endswith("s1") and fam[:-2] in FAMILIES:
        inp0 = load_inputs(cfg, fam[:-2])
        if inp0 is not None:
            try:
                f0 = eval_points("floor", cfg, inp0, "pow2", pts)
                m0 = lik_part(eval_points("main", cfg, inp0, "pow2", pts), f0)
                sc = float(inp0["scale"])
                n_det, N = int(inp0["strains"].shape[0]), int(inp0["strains"].shape[1])
                expected = n_det * N * np.log(sc)
                g_rel = max(rel_grad_err(g1, g0, scale_main)[0]
                            for (_, g1), (_, g0) in zip(main_vals, m0))
                dU = max(abs((U1 - U0) - expected) for (U1, _), (U0, _) in zip(main_vals, m0))
                cell["_scale_twin"] = {"twin": fam[:-2], "scale": sc,
                                       "expected_dU_nats": float(expected),
                                       "rel_grad_cloud": float(g_rel),
                                       "nats_minus_expected": float(dU)}
                log("  %-12s %-9s scale twin vs %s (scale %.3e): grad rel %.1e (cloud), "
                    "|dU - n_det N log scale| = %.1e nats"
                    % (cfg_key(cfg), fam, fam[:-2], sc, g_rel, dU))
            except Exception as e:
                cell["_scale_twin"] = {"twin": fam[:-2], "error": repr(e)}
    return cell


# ---------------------------------------------------------------------------
# SECTION: per-gradient device time + census
# ---------------------------------------------------------------------------
def section_devtime():
    log("\n=== PER-GRADIENT DEVICE TIME (us/grad) [leg %s] ===" % LEG)
    log("    slope method over two fori_loop counts, medians of %d; per-cell "
        "budget %.0f s; constants are jit ARGUMENTS" % (REP, CELL_BUDGET))
    res = {"_dropped": [],
           # the thread configuration these numbers belong to
           "_threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                        "NPROC": os.environ.get("NPROC"),
                        "xla_cpu_pool_threads": xla_cpu_pool_threads()},
           "_rep": REP, "_slope_note": "us_per_grad = slope over two fori_loop "
           "counts (medians of rep timings); us_spread = slope range over the "
           "individual repetitions (repeatability, not a confidence interval)"}
    log("    threads: %s" % res["_threads"])
    for cfg in CONFIGS:
        ck = cfg_key(cfg)
        res[ck] = {}
        for fi, fam in enumerate(TIMING_FAMILIES):
            inp = load_inputs(cfg, fam)
            if inp is None:
                res["_dropped"].append({"cell": "%s/%s" % (ck, fam),
                                        "reason": "missing inputs"})
                continue
            res[ck][fam] = {}
            log("  --- (n_det=%d, N=%d, n_modes=%d) k=%d  family %s  cond=%.1e ---"
                % (cfg + (4 * cfg[2], fam, cond_of(inp))))
            kw = kw_for(inp)
            times, strains, fps, fcs = V.data_args(inp, DT)
            p = fixed_points(inp, 1)[0]
            base = {}
            for variant in VARIANTS:
                for mode in variant_modes(variant):
                    vk = vkey(variant, mode)
                    cell_name = "%s/%s/%s" % (ck, fam, vk)
                    t_cell = time.perf_counter()
                    try:
                        model = V.make_model(variant, modes_for(cfg), nfft_mode=mode, **kw)
                        consts = consts_for(variant, inp, mode)
                        g = H.grad_of_args(model, times, strains, fps, fcs)
                        # the resolved FFT length: for N >= 1024 'fast' and
                        # 'pow2' coincide, so those columns time one executable
                        r = {"nfft": (V.nfft_of(inp, mode)
                                      if variant in V.GS_VARIANTS else None)}
                        # census only once per config (it does not depend on
                        # the constant VALUES); the compile time is recorded
                        if fi == 0:
                            tc0 = time.perf_counter()
                            r["census_grad"] = H.count_custom_calls_args(g, p, (consts,))
                            r["compile_s"] = time.perf_counter() - tc0
                            r["census_looped"] = H.count_custom_calls_looped_args(
                                g, p, (consts,), R=3)
                            r["hoisted"] = H.hoisting_report(r["census_grad"],
                                                             r["census_looped"])
                        # time-box: a first call tells us the per-call cost
                        jax.block_until_ready(g(p, consts))
                        t0 = time.perf_counter()
                        jax.block_until_ready(g(p, consts))
                        t_one = time.perf_counter() - t0
                        # the smallest run the slope method can do is rep=3
                        # over R1=2 and R2=6, i.e. 24 gradients; drop the
                        # cell if even that is far beyond the budget
                        if 3 * (2 + 6) * t_one > 4 * CELL_BUDGET:
                            res["_dropped"].append(
                                {"cell": cell_name, "reason":
                                 "single gradient %.2f s exceeds cell budget" % t_one})
                            log("    %-22s DROPPED (%.2f s per gradient)" % (vk, t_one))
                            continue
                        us, R1, R2, rep_used, t_call, spread = H.device_us_per_grad_args(
                            g, p, (consts,), target_s=TARGET_S, rep=REP,
                            budget_s=CELL_BUDGET)
                        r.update({"us_per_grad": us, "R": [R1, R2], "rep": rep_used,
                                  "t_call_s": t_call, "nfft_mode": mode,
                                  "us_spread": spread,
                                  "cell_wall_s": time.perf_counter() - t_cell})
                        if variant in ("main", "main_hoisted"):
                            base[variant] = us
                        if "main" in base:
                            r["speedup_vs_main"] = base["main"] / us
                        if "main_hoisted" in base:
                            r["speedup_vs_main_hoisted"] = base["main_hoisted"] / us
                        res[ck][fam][vk] = r
                        cen = r.get("census_grad", {})
                        log("    %-22s %10.1f us [%.0f..%.0f]  %6.2fx main  R=%s rep=%d  %s%s"
                            % (vk, us, spread["us_min"], spread["us_max"],
                               r.get("speedup_vs_main", float("nan")),
                               r["R"], rep_used,
                               " ".join("%s=%s" % (k, v) for k, v in cen.items()
                                        if k in ("trsm", "trsm_big", "potrf", "gemm", "dot", "fft")),
                               ("  hoisted:%s" % r["hoisted"]) if r.get("hoisted") else ""))
                    except Exception as e:
                        traceback.print_exc()
                        res[ck][fam][vk] = {"error": repr(e)}
                        log("    %-22s FAILED: %r" % (vk, e))
    return res


# ---------------------------------------------------------------------------
# SECTION: compile time
# ---------------------------------------------------------------------------
def section_compile():
    log("\n=== COMPILE TIME of jit(grad) [leg %s] ===" % LEG)
    res = {}
    fam = TIMING_FAMILIES[0]
    for cfg in CONFIGS:
        ck = cfg_key(cfg)
        inp = load_inputs(cfg, fam)
        if inp is None:
            res[ck] = {"error": "missing inputs %s" % fam}
            continue
        res[ck] = {"_family": fam}
        kw = kw_for(inp)
        times, strains, fps, fcs = V.data_args(inp, DT)
        p = fixed_points(inp, 1)[0]
        for variant in VARIANTS:
            for mode in variant_modes(variant):
                vk = vkey(variant, mode)
                try:
                    model = V.make_model(variant, modes_for(cfg), nfft_mode=mode, **kw)
                    consts = consts_for(variant, inp, mode)
                    g = H.grad_of_args(model, times, strains, fps, fcs)
                    t0 = time.perf_counter()
                    lowered = g.lower(p, consts)
                    t1 = time.perf_counter()
                    lowered.compile()
                    t2 = time.perf_counter()
                    res[ck][vk] = {"lower_s": t1 - t0, "compile_s": t2 - t1,
                                   "total_s": t2 - t0}
                    log("  %-12s %-22s lower %5.2f s  compile %5.2f s"
                        % (ck, vk, t1 - t0, t2 - t1))
                except Exception as e:
                    res[ck][vk] = {"error": repr(e)}
                    log("  %-12s %-22s FAILED: %r" % (ck, vk, e))
    return res


# ---------------------------------------------------------------------------
# SECTION: accuracy vs reference and vs the float64 twin (every leg)
# ---------------------------------------------------------------------------
def section_f32acc(ref_path):
    """Recompute U_lik and its gradient at the reference points in THIS
    leg's precision and compare with (a) the extended-precision reference and
    (b) the cpu float64 twin stored in the same npz.  In float64 legs (b) is
    a cross-platform check; in the producing leg it is skipped.

    Every error is reported per-point-normalized (rel_grad_vs_*) AND
    cloud-normalized (rel_grad_vs_*_cloud, see rel_grad_err), overall and per
    point kind (*_by_kind).  In float32 legs the GS variants gs_pr/gs_full/
    gs_half are additionally evaluated with the other spectra policy
    (--spectra-from) as `alt_spectra` (checklist item 1)."""
    log("\n=== ACCURACY vs reference and vs float64 twin [leg %s, %s] ==="
        % (LEG, np.dtype(DT)))
    log("    matmul: %s" % ("TF32 allowed for float32 dots (JAX default on Ampere+)"
                            if _IS_GPU and not ARGS.x64 and not ARGS.matmul_precision
                            else "jax_default_matmul_precision=%r"
                            % str(jax.config.jax_default_matmul_precision)))
    blob = load_ref(ref_path)
    if blob is None:
        log("  no reference npz at %r -- section skipped" % ref_path)
        return {"error": "no reference npz at %r" % ref_path}
    producer = str(blob.get("meta|leg", np.array(["?"]))[0])
    is_producer = (producer == LEG)
    res = {"_producer_leg": producer, "_dtype": str(np.dtype(DT)),
           "_matmul_precision": str(jax.config.jax_default_matmul_precision),
           "_tf32_possible": bool(_IS_GPU and not ARGS.x64 and not ARGS.matmul_precision),
           "_spectra_from": ARGS.spectra_from, "_gs_coeffs": ARGS.gs_coeffs}
    log("  %-12s %-9s %-22s %9s %9s %9s %9s %9s %6s"
        % ("config", "family", "variant", "nats_ref", "relg_ref", "relg_ref*",
           "relg_twn", "relg_twn*", "finite"))
    log("  (* = cloud-normalized; by-kind numbers in the JSON)")

    def compare(vals, refs, twins, kind_names, scale_ref, scale_twin):
        r = {"nats_vs_ref": 0.0, "rel_grad_vs_ref": 0.0, "rel_grad_vs_ref_cloud": 0.0,
             "nats_vs_twin": None, "rel_grad_vs_twin": None, "rel_grad_vs_twin_cloud": None,
             "all_finite": True, "per_site_vs_ref": {}, "per_site_vs_ref_cloud": {},
             "rel_grad_vs_ref_by_kind": {}, "rel_grad_vs_ref_cloud_by_kind": {},
             "nats_vs_ref_by_kind": {},
             "rel_grad_vs_twin_by_kind": {}, "rel_grad_vs_twin_cloud_by_kind": {}}

        def bump(key, kn, val):
            r[key] = max(r[key] or 0.0, val)
            r[key + "_by_kind"][kn] = max(r[key + "_by_kind"].get(kn, 0.0), val)

        for j, (U, g) in enumerate(vals):
            kn = kind_names[j]
            fin = np.isfinite(U) and all(np.all(np.isfinite(v)) for v in g.values())
            if not fin:
                r["all_finite"] = False
                r["nats_vs_ref"] = r["rel_grad_vs_ref"] = float("nan")
                r["rel_grad_vs_ref_cloud"] = float("nan")
                continue
            if refs[j] is not None:
                Ur, gr = refs[j]
                r["nats_vs_ref"] = max(r["nats_vs_ref"], abs(U - Ur))
                r["nats_vs_ref_by_kind"][kn] = max(r["nats_vs_ref_by_kind"].get(kn, 0.0),
                                                   abs(U - Ur))
                e, per = rel_grad_err(g, gr)
                bump("rel_grad_vs_ref", kn, e)
                for k, v in per.items():
                    r["per_site_vs_ref"][k] = max(r["per_site_vs_ref"].get(k, 0.0), v)
                e, per = rel_grad_err(g, gr, scale_ref)
                bump("rel_grad_vs_ref_cloud", kn, e)
                for k, v in per.items():
                    r["per_site_vs_ref_cloud"][k] = max(r["per_site_vs_ref_cloud"].get(k, 0.0), v)
            if twins is not None and twins[j] is not None:
                Ut, gt = twins[j]
                r["nats_vs_twin"] = max(r["nats_vs_twin"] or 0.0, abs(U - Ut))
                e, _ = rel_grad_err(g, gt)
                bump("rel_grad_vs_twin", kn, e)
                e, _ = rel_grad_err(g, gt, scale_twin)
                bump("rel_grad_vs_twin_cloud", kn, e)
        return r

    for cfg in CONFIGS:
        ck = cfg_key(cfg)
        res[ck] = {}
        for fam in FAMILIES:
            inp = load_inputs(cfg, fam)
            if inp is None:
                res[ck][fam] = {"error": "missing inputs"}
                continue
            if ref_keys(cfg, fam, 0) + "refU" not in blob:
                res[ck][fam] = {"error": "not in reference"}
                continue
            pts = fixed_points(inp)
            if not check_points_match(blob, cfg, fam, pts):
                res[ck][fam] = {"error": "reference points differ from inputs"}
                continue
            kinds = fixed_point_kinds(inp)
            kind_names = [KIND_NAMES.get(k, "kind%d" % k) for k in kinds]
            refs = [ref_values(blob, cfg, fam, j) for j in range(len(pts))]
            scale_ref = cloud_scale([g for g in (r[1] for r in refs if r is not None)]) or None
            cell = {"_cond": cond_of(inp), "_n_pts": len(pts),
                    "_n_pts_by_kind": {kn: kind_names.count(kn) for kn in sorted(set(kind_names))},
                    "_cloud_scale_ref": scale_ref}
            try:
                floor_vals = eval_points("floor", cfg, inp, "pow2", pts)
            except Exception as e:
                res[ck][fam] = {"error": "floor: %r" % (e,)}
                continue
            for variant in VARIANTS:
                if variant == "floor":
                    continue
                for mode in variant_modes(variant):
                    vk = vkey(variant, mode)
                    try:
                        vals = lik_part(eval_points(variant, cfg, inp, mode, pts),
                                        floor_vals)
                    except Exception as e:
                        cell[vk] = {"error": repr(e), "all_finite": False}
                        continue
                    twins = None
                    scale_twin = None
                    if not is_producer:
                        twins = [twin_values(blob, cfg, fam, j, vk) for j in range(len(pts))]
                        if all(t is None for t in twins):
                            twins = None
                        else:
                            scale_twin = cloud_scale([t[1] for t in twins if t is not None]) or None
                    r = compare(vals, refs, twins, kind_names, scale_ref, scale_twin)
                    # checklist item 1: the other spectra policy, same executable
                    if DT == np.float32 and variant in ALT_COEFFS_VARIANTS:
                        other = "leg" if ARGS.spectra_from == "f64" else "f64"
                        try:
                            vals2 = lik_part(eval_points(variant, cfg, inp, mode, pts,
                                                         spectra_from=other), floor_vals)
                            r2 = compare(vals2, refs, twins, kind_names, scale_ref, scale_twin)
                            r["alt_spectra"] = {"policy": other}
                            for k in ("rel_grad_vs_ref", "rel_grad_vs_ref_cloud",
                                      "rel_grad_vs_twin", "rel_grad_vs_twin_cloud",
                                      "nats_vs_ref", "rel_grad_vs_ref_by_kind"):
                                r["alt_spectra"][k] = r2[k]
                        except Exception as e:
                            r["alt_spectra"] = {"policy": other, "error": repr(e)}
                    cell[vk] = r

                    def fx(v):
                        return ("%.2e" % v) if v is not None else "-"
                    log("  %-12s %-9s %-22s %9s %9s %9s %9s %9s %6s"
                        % (ck, fam, vk, fx(r["nats_vs_ref"]), fx(r["rel_grad_vs_ref"]),
                           fx(r["rel_grad_vs_ref_cloud"]), fx(r["rel_grad_vs_twin"]),
                           fx(r["rel_grad_vs_twin_cloud"]), r["all_finite"]))
            res[ck][fam] = cell
    return res


# ---------------------------------------------------------------------------
# SECTION: short NUTS runs (cpu float64 leg, explicit --sections nuts)
# ---------------------------------------------------------------------------
def section_nuts():
    """Short NUTS runs at the production config.

    Per variant: one COLD run (seed 0; its wall includes the sampler's JIT
    compile, which differs by variant: ~10 s for main vs ~0.2 s for
    gemm_linv at 300+300) and then one WARM run per --nuts-seeds on the same
    MCMC object (compiled kernel reused).  Throughput numbers (us/leapfrog,
    ESS/s) come from the warm runs; ESS is reported per seed so the seed-to-
    seed spread (which exceeds the variant-to-variant differences at this
    size) is visible.  1 chain per run: at (2,205,2) on CPU the gradient is
    only ~25-40% of the per-leapfrog wall (compare us_per_leapfrog with the
    devtime section's us/grad), so a gradient speedup cannot move NUTS wall
    by more than that share.
    """
    n = ARGS.nuts_n
    cfg, fam = PROD, "aligo02"
    seeds = [int(x) for x in ARGS.nuts_seeds.split(",") if x.strip()]
    log("\n=== NUTS %d+%d on %s %s [leg %s]: cold run (seed 0, incl. compile) then "
        "warm runs, seeds %s ===" % (n, n, cfg_key(cfg), fam, LEG, seeds))
    if ARGS.smoke:
        log("  skipped in smoke mode")
        return {"skipped": "smoke"}
    if DT != np.float64:
        log("  nuts runs in the float64 leg only -- skipped")
        return {"skipped": "float32 leg"}
    inp = load_inputs(cfg, fam)
    if inp is None:
        return {"error": "missing inputs"}
    kw = kw_for(inp)
    times, strains, fps, fcs = V.data_args(inp, DT)
    res = {"config": list(cfg), "family": fam, "num_warmup": n, "num_samples": n,
           "num_chains": 1, "seeds_warm": seeds,
           "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                       "NPROC": os.environ.get("NPROC"),
                       "xla_cpu_pool_threads": xla_cpu_pool_threads()}}
    from numpyro.diagnostics import effective_sample_size
    variants = [v.strip() for v in ARGS.nuts_variants.split(",") if v.strip()]

    def one_run(mc, seed):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mc.run(jax.random.PRNGKey(seed), times, strains, consts, fps, fcs,
                   extra_fields=("num_steps",))
            samples = mc.get_samples()
            jax.block_until_ready(samples["m"])
        wall = time.perf_counter() - t0
        nsteps = int(np.sum(np.asarray(mc.get_extra_fields()["num_steps"])))
        out = {"seed": seed, "wall_s": wall, "num_steps": nsteps,
               "us_per_leapfrog": wall / max(nsteps, 1) * 1e6,
               "ess": {}, "ess_per_s": {}, "mean": {}, "sd": {}}
        for site in ("m", "chi", "a_scale"):
            x = np.asarray(samples[site], dtype=np.float64)
            if x.ndim == 1:
                x = x[:, None]
            ess = np.asarray(effective_sample_size(x[None, ...]))
            out["ess"][site] = [float(e) for e in np.atleast_1d(ess)]
            out["ess_per_s"][site] = [float(e) / wall for e in np.atleast_1d(ess)]
            out["mean"][site] = [float(v) for v in x.mean(axis=0)]
            out["sd"][site] = [float(v) for v in x.std(axis=0, ddof=1)]
        return out

    for variant in variants:
        if variant not in V.VARIANTS or variant == "floor":
            res[variant] = {"error": "unknown variant"}
            continue
        try:
            model = V.make_model(variant, modes_for(cfg), nfft_mode=NFFT_MODES[0], **kw)
            consts = consts_for(variant, inp, NFFT_MODES[0])
            # production settings (ringdown/fit.py: dense_mass=True), 1 chain
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mc = MCMC(NUTS(model, dense_mass=True), num_warmup=n, num_samples=n,
                          num_chains=1, progress_bar=False)
            cold = one_run(mc, 0)
            warm = [one_run(mc, sd) for sd in seeds]
            walls = [w["wall_s"] for w in warm]
            uspl = [w["us_per_leapfrog"] for w in warm]
            r = {"cold": cold, "warm": warm,
                 # legacy fields (the cold run) kept for older analyzers
                 "wall_s": cold["wall_s"], "num_steps": cold["num_steps"],
                 "us_per_leapfrog_incl_compile": cold["us_per_leapfrog"],
                 "ess": cold["ess"], "ess_per_s": cold["ess_per_s"],
                 "mean": cold["mean"], "sd": cold["sd"],
                 # headline: warm numbers
                 "compile_s_est": cold["wall_s"] - warm[0]["wall_s"] if warm else None,
                 "wall_warm_s": float(np.median(walls)) if walls else None,
                 "us_per_leapfrog_warm": float(np.median(uspl)) if uspl else None,
                 "us_per_leapfrog_warm_range": [min(uspl), max(uspl)] if uspl else None,
                 "ess_m_warm_range": [min(w["ess"]["m"][0] for w in warm),
                                      max(w["ess"]["m"][0] for w in warm)] if warm else None,
                 "ess_m_per_s_warm_range": [min(w["ess_per_s"]["m"][0] for w in warm),
                                            max(w["ess_per_s"]["m"][0] for w in warm)]
                 if warm else None,
                 "mean_m_warm": [w["mean"]["m"][0] for w in warm],
                 "sd_m_warm": [w["sd"]["m"][0] for w in warm]}
            res[variant] = r
            log("  %-14s cold %6.1f s (%d steps, %.0f us/leapfrog incl. compile; "
                "compile ~%.1f s)  warm %s s  %s us/leapfrog  ESS(m) %s  ESS(m)/s %s  "
                "mean m %s" % (
                    variant, cold["wall_s"], cold["num_steps"], cold["us_per_leapfrog"],
                    r["compile_s_est"] or float("nan"),
                    "/".join("%.1f" % w for w in walls),
                    "/".join("%.0f" % u for u in uspl),
                    "/".join("%.0f" % w["ess"]["m"][0] for w in warm),
                    "/".join("%.1f" % w["ess_per_s"]["m"][0] for w in warm),
                    "/".join("%.2f" % w["mean"]["m"][0] for w in warm)))
        except Exception as e:
            traceback.print_exc()
            res[variant] = {"error": repr(e)}
            log("  %-14s FAILED: %r" % (variant, e))
    return res


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_leg(ref_path):
    """Run the sections applicable to THIS process and return the JSON."""
    out = {"schema": 1, "kind": "gs-bench-leg", "leg": LEG, "argv": sys.argv,
           "tag": ARGS.tag, "dtype": str(np.dtype(DT)), "platform": ARGS.platform,
           "omp": ARGS.omp}
    out["env"] = section_env()
    order = [("reference", lambda: section_reference(ref_path)),
             ("correctness", lambda: section_correctness(ref_path)),
             ("devtime", section_devtime),
             ("compile", section_compile),
             ("f32acc", lambda: section_f32acc(ref_path)),
             ("nuts", section_nuts)]
    for name, fn in order:
        if not want(name):
            continue
        t0 = time.perf_counter()
        try:
            out[name] = fn()
        except Exception as e:
            traceback.print_exc()
            out[name] = {"error": repr(e), "traceback": traceback.format_exc()[-4000:]}
        out.setdefault("section_wall_s", {})[name] = time.perf_counter() - t0
    if _IS_GPU:
        out["contention_after"] = H.gpu_contention(ARGS.parent_pid or None)
    out["t_total_s"] = time.time() - T_START
    return out


def main():
    base = os.path.splitext(ARGS.out)[0]
    ref_path = ARGS.ref or (base + ".ref.npz")
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)

    if ARGS.no_sub:
        out = run_leg(ref_path)
        with open(ARGS.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        log("\n=== LEG %s DONE in %.1f s -> %s ===" % (LEG, out["t_total_s"],
                                                      os.path.abspath(ARGS.out)))
        return

    # ---- parent: orchestrate the legs ----
    out = {"schema": 1, "kind": "gs-bench", "argv": sys.argv, "tag": ARGS.tag,
           "parent_platform": ARGS.platform, "env": section_env(), "legs": {}}
    legs = [nm for nm in LEG_ORDER if ARGS.platform == "gpu" or not nm.startswith("gpu")]
    for nm in legs:
        spec = LEG_TABLE[nm]
        secs = [s for s in spec["sections"] if s in SECTIONS]
        if not secs:
            continue
        argv = list(spec["argv"]) + ["--leg", nm, "--sections", ",".join(secs),
                                     "--inputs", ARGS.inputs, "--ref", ref_path,
                                     "--nfft", ARGS.nfft,
                                     "--timing-families", ARGS.timing_families,
                                     "--nuts-variants", ARGS.nuts_variants,
                                     "--nuts-n", str(ARGS.nuts_n),
                                     "--nuts-seeds", ARGS.nuts_seeds,
                                     "--cpu-omp", str(ARGS.cpu_omp),
                                     "--gs-coeffs", ARGS.gs_coeffs,
                                     "--spectra-from", ARGS.spectra_from,
                                     "--ref-logdet", ARGS.ref_logdet]
        for flag, val in (("--configs", ARGS.configs), ("--families", ARGS.families),
                          ("--variants", ARGS.variants), ("--tag", ARGS.tag),
                          ("--matmul-precision", ARGS.matmul_precision)):
            if val and flag not in spec["argv"]:     # a leg's own setting wins
                argv += [flag, val]
        for flag, val in (("--npts", ARGS.npts), ("--cell-budget", ARGS.cell_budget),
                          ("--target-s", ARGS.target_s)):
            if val:
                argv += [flag, str(val)]
        t0 = time.perf_counter()
        out["legs"][nm] = H._spawn(__file__, nm, argv, base + ".%s.json" % nm,
                                   smoke=ARGS.smoke, timeout=ARGS.leg_timeout)
        out["legs"][nm]["_spawn_wall_s"] = time.perf_counter() - t0
        # write after every leg so a crash later still leaves a usable file
        out["t_total_s"] = time.time() - T_START
        with open(ARGS.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    if _IS_GPU:
        out["contention_after"] = H.gpu_contention()
    out["t_total_s"] = time.time() - T_START
    with open(ARGS.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    log("\n=== ALL LEGS DONE in %.1f s -> %s ===" % (out["t_total_s"],
                                                    os.path.abspath(ARGS.out)))
    log("Next:  python %s %s" % (os.path.join(_HERE, "analyze_gs.py"),
                                 os.path.abspath(ARGS.out)))


if __name__ == "__main__":
    main()
