"""Timing and HLO-census harness for the Gohberg-Semencul benchmark kit.

The functions `_grad_of`, `_looped`, `_tmed`, `device_us_per_grad`,
`gpu_contention` and `_spawn` are verbatim copies of the same-named functions
in benchmarks/h100/bench.py at commit 0246591eb32f5eac3d7fb8ed21190c92193bc7e0
(branch main / eval-pr-141).  They are copied rather than imported because
bench.py parses sys.argv at import time.  `count_custom_calls` is the bench.py
census extended with GEMM and FFT op counts, and `count_custom_calls_looped`
runs the same census on the fori_loop form so that loop-invariant code motion
(hoisting of theta-independent work out of the while body) can be detected.

The `*_args` twins (`looped_args`, `tmed_args`, `device_us_per_grad_args`,
`count_custom_calls_args`, `count_custom_calls_looped_args`) generalize the
verbatim functions to gradient functions that take EXTRA positional arguments
(the per-detector constants) so that those constants are entry parameters of
the compiled executable rather than embedded literals.  With jax 0.11 a
closed-over array becomes an HLO constant literal, but XLA does NOT
constant-fold work on it: jit(grad(sum(irfft(rfft(x) * rfft(a_closed)))))
compiles to three fft ops on CPU and on GPU, one of them literally
`fft(%constant), fft_type=RFFT`, the same census as the as-argument form.
What does remove theta-independent work is loop-invariant code motion out of
`while` loops: XLA:CPU hoists it (the timing fori_loop here, NUTS tree
building in production: `main`'s two N x 1 solves z = L^{-1} y and
gs_pr_ascoded's four constant rffts leave the loop body, see the census
`moved` column), XLA:GPU does not.  Constants are passed as arguments so the
plain-gradient census is unambiguous, and `count_custom_calls_looped_args`
reports the hoisting separately.

This module imports jax; the caller must have set JAX_PLATFORMS and
jax_enable_x64 before importing it.
"""

import json
import os
import re
import statistics as stats
import subprocess
import sys
import time

import jax
from numpyro.infer.util import potential_energy


# ---------------------------------------------------------------------------
# Verbatim copies from benchmarks/h100/bench.py (PHASE 3: timing harnesses)
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


# ---------------------------------------------------------------------------
# Generalized twins: gradient functions of the form gradfn(q, *extra)
# ---------------------------------------------------------------------------
def grad_of_args(model, times, strains, fps, fcs):
    """jit(grad) of the potential with the constants dict as a jit ARGUMENT.

    Returns gradfn(q, consts).  `times`, `strains`, `fps`, `fcs` are closed
    over (they are data, identical for every variant, and would be closed
    over by numpyro's MCMC as well); `consts` is a pytree argument so that
    the census of the plain gradient counts every op the route performs
    (see the module docstring: XLA does not fold ops on closed-over arrays,
    but keeping the constants as entry parameters makes that a non-issue).
    """
    def U(q, consts):
        return potential_energy(model, (times, strains, consts, fps, fcs), {}, q)
    return jax.jit(jax.grad(U))


def value_and_grad_of_args(model, times, strains, fps, fcs):
    """Same as grad_of_args but returns (U, grad)."""
    def U(q, consts):
        return potential_energy(model, (times, strains, consts, fps, fcs), {}, q)
    return jax.jit(jax.value_and_grad(U))


def looped_args(gradfn, p0, R):
    """fori_loop of R gradients; go(p, *extra) with the same inert feedback."""
    @jax.jit
    def go(p, *extra):
        def body(i, q):
            g = gradfn(q, *extra)
            return jax.tree.map(lambda a, b: a + 1e-30 * b, p0, g)
        return jax.lax.fori_loop(0, R, body, p)
    return go


def tmed_args(f, p, extra, rep, all_times=False):
    jax.block_until_ready(f(p, *extra))          # compile + warm
    T = []
    for _ in range(rep):
        t0 = time.perf_counter()
        jax.block_until_ready(f(p, *extra))
        T.append(time.perf_counter() - t0)
    return (stats.median(T), T) if all_times else stats.median(T)


def device_us_per_grad_args(gradfn, p, extra, target_s=0.15, rep=5, rmin=8,
                            rmax=400, budget_s=None):
    """Per-gradient DEVICE time in us for gradfn(q, *extra).

    Identical slope method to `device_us_per_grad`.  If `budget_s` is given
    the loop counts and repetitions are reduced so that the two timed loops
    stay within roughly that many seconds; the return value reports what
    was actually used.  Returns (us_per_grad, R1, R2, rep_used, t_call_s,
    spread) where spread = {us_min, us_max, t1_s, t2_s} is the range of the
    slope over the individual repetitions.
    """
    jax.block_until_ready(gradfn(p, *extra))
    t0 = time.perf_counter()
    for _ in range(3):
        jax.block_until_ready(gradfn(p, *extra))
    t_call = (time.perf_counter() - t0) / 3.0
    R1 = int(min(rmax, max(rmin, target_s / max(t_call, 1e-6))))
    if budget_s is not None:
        # total loop work is about rep * (R1 + 3 R1) gradients plus one warm
        # call each; shrink rep first, then R1 (never below 2)
        while rep > 3 and rep * 4 * R1 * t_call > budget_s:
            rep -= 1
        while R1 > 2 and rep * 4 * R1 * t_call > budget_s:
            R1 = max(2, R1 // 2)
    R2 = 3 * R1
    t1, T1 = tmed_args(looped_args(gradfn, p, R1), p, extra, rep, all_times=True)
    t2, T2 = tmed_args(looped_args(gradfn, p, R2), p, extra, rep, all_times=True)
    us = (t2 - t1) / (R2 - R1) * 1e6
    # spread of the slope over the individual repetitions: the most and the
    # least favourable pairing of the two loops' timings.  A per-cell
    # repeatability estimate; not a confidence interval.
    spread = {"us_min": (min(T2) - max(T1)) / (R2 - R1) * 1e6,
              "us_max": (max(T2) - min(T1)) / (R2 - R1) * 1e6,
              "t1_s": [float(x) for x in T1], "t2_s": [float(x) for x in T2]}
    return us, R1, R2, rep, t_call, spread


# ---------------------------------------------------------------------------
# HLO census
# ---------------------------------------------------------------------------
# custom-call targets grouped under a short name (bench.py SHORT, extended)
SHORT = {"__cublas$triangularSolve": "trsm", "cusolver_potrf_ffi": "potrf",
         "lapack_dtrsm_ffi": "trsm", "lapack_dpotrf_ffi": "potrf",
         "lapack_strsm_ffi": "trsm", "lapack_spotrf_ffi": "potrf",
         "__cublas$gemm": "gemm", "__cublas$lt$matmul": "gemm",
         "__cublas$lt$matmul$f8": "gemm", "__onednn$matmul": "gemm",
         "__cublas$getrf": "getrf", "cusolver_getrf_ffi": "getrf",
         "lapack_dgetrf_ffi": "getrf", "lapack_sgetrf_ffi": "getrf",
         "cudnn$fft": "fft_custom", "cufft": "fft_custom"}
CUSTOM_KEYS = tuple(SHORT.keys())

# an HLO instruction line looks like  `%name = TYPE[...] op(...), attrs`; the
# op name is the token right before the first `(` after the `=`
_OP_RE = re.compile(r"=\s*\S+\s+([a-zA-Z_$-]+)\(")
_FFT_TYPE_RE = re.compile(r"fft_type=(\w+)")


# result shape of an HLO instruction: `%name = f64[205,8]{0,1} ...` (a tuple
# result such as potrf's `(f64[8,8]{0,1}, s32[])` yields its first element)
_SHAPE_RE = re.compile(r"=\s*\(?[a-z]+\d*\[([\d,]*)\]")
BIG_DIM = 64     # k = 4 n_modes <= 32 in the grid, N >= 205: any dim >= 64 is N-sized


def census_of_text(txt):
    """Op census of an HLO text (whole module or one computation block).

    Besides the counts, trsm/potrf custom calls are binned by result shape
    (`trsm_shapes`, e.g. {"205x8": 4, "8x1": 2}) and `trsm_big` counts the
    ones with an N-sized dimension (>= BIG_DIM).  Every hoisted variant keeps
    the k x k Cholesky of A^{-1} and its solves (potrf + 4 small trsm in the
    gradient), so `trsm_big` is the number that tells the routes apart.
    """
    out = {}
    for k in CUSTOM_KEYS:                # several keys share a short name
        n = txt.count('custom_call_target="%s"' % k)
        sk = SHORT.get(k, k)
        out[sk] = out.get(sk, 0) + n
    n_dot = n_fft = 0
    fft_types = {}
    trsm_shapes, potrf_shapes = {}, {}
    n_trsm_big = 0
    for ln in txt.splitlines():
        if "custom_call_target=" in ln:
            for k in CUSTOM_KEYS:
                if 'custom_call_target="%s"' % k not in ln:
                    continue
                sk = SHORT.get(k, k)
                if sk in ("trsm", "potrf"):
                    ms = _SHAPE_RE.search(ln)
                    dims = [int(d) for d in ms.group(1).split(",") if d] if ms else []
                    key = "x".join(str(d) for d in dims) or "?"
                    bins = trsm_shapes if sk == "trsm" else potrf_shapes
                    bins[key] = bins.get(key, 0) + 1
                    if sk == "trsm" and any(d >= BIG_DIM for d in dims):
                        n_trsm_big += 1
            continue
        m = _OP_RE.search(ln)
        if not m:
            continue
        op = m.group(1)
        if op == "dot":
            n_dot += 1
        elif op == "fft":
            n_fft += 1
            mt = _FFT_TYPE_RE.search(ln)
            if mt:
                fft_types[mt.group(1)] = fft_types.get(mt.group(1), 0) + 1
    out["trsm_big"] = n_trsm_big
    if trsm_shapes:
        out["trsm_shapes"] = trsm_shapes
    if potrf_shapes:
        out["potrf_shapes"] = potrf_shapes
    # `dot(` ops are XLA-native matmuls (CPU emitter, or fused GEMMs the GPU
    # backend kept as HLO dots); cuBLAS custom calls are counted under `gemm`
    # already, so report both and a total
    out["dot"] = n_dot
    out["gemm"] = out.get("gemm", 0)
    out["gemm_total"] = out["gemm"] + n_dot
    out["fft"] = n_fft
    if fft_types:
        out["fft_types"] = fft_types
    out["hlo_lines"] = txt.count("\n") + 1
    return {k: v for k, v in out.items() if v}


def _hlo_computations(txt):
    """Split post-optimization HLO text into {computation_name: block_text}.

    A computation block starts with `%name (...) -> ... {` or
    `ENTRY %name (...)` and ends at the matching `}` at column 0.
    """
    blocks, cur, name = {}, [], None
    for ln in txt.splitlines():
        m = re.match(r"^(?:ENTRY\s+)?%([\w.\-$]+)\s*\(", ln)
        if m and ln.rstrip().endswith("{"):
            if name is not None:
                blocks[name] = "\n".join(cur)
            name, cur = m.group(1), [ln]
        elif name is not None:
            cur.append(ln)
            if ln.startswith("}"):
                blocks[name] = "\n".join(cur)
                name, cur = None, []
    if name is not None:
        blocks[name] = "\n".join(cur)
    return blocks


def census_while_body(txt):
    """Census restricted to the body computation(s) of `while` ops.

    Returns (body_census, n_while).  Nested calls (fusions, called
    computations) referenced from the body are followed one level down so
    that fused ops inside the body are counted; XLA's post-optimization
    module keeps fusion computations as separate blocks.
    """
    comps = _hlo_computations(txt)
    bodies = re.findall(r"while\(.*?\),.*?body=%([\w.\-$]+)", txt)
    if not bodies:
        return {}, 0
    seen, stack = set(), list(bodies)
    while stack:
        nm = stack.pop()
        if nm in seen or nm not in comps:
            continue
        seen.add(nm)
        # follow references to other computations (calls=, to_apply=, etc.)
        for ref in re.findall(r"(?:calls|to_apply|body|condition|"
                              r"branch_computations|true_computation|"
                              r"false_computation)=\{?%([\w.\-$]+)",
                              comps[nm]):
            stack.append(ref)
    joined = "\n".join(comps[nm] for nm in seen)
    return census_of_text(joined), len(bodies)


def count_custom_calls(gradfn, p):
    """Post-optimization HLO op census of gradfn(p) (the structural evidence).

    Extends bench.py's census (trsm, potrf, hlo_lines) with gemm (cuBLAS
    custom calls), dot (native HLO dot ops), fft (HLO fft ops, by type).
    """
    try:
        txt = gradfn.lower(p).compile().as_text()
    except Exception as e:
        return {"error": repr(e)}
    return census_of_text(txt)


def count_custom_calls_args(gradfn, p, extra):
    try:
        txt = gradfn.lower(p, *extra).compile().as_text()
    except Exception as e:
        return {"error": repr(e)}
    return census_of_text(txt)


def count_custom_calls_looped(gradfn, p, R=3):
    """Census of the fori_loop form of gradfn: whole module and while body.

    If XLA hoisted theta-independent work (constant FFTs, factorizations of
    constants) out of the loop, the body census has fewer ops than the
    census of the plain gradient while the module census still has them.
    """
    try:
        txt = _looped(gradfn, p, R).lower(p).compile().as_text()
    except Exception as e:
        return {"error": repr(e)}
    body, nwhile = census_while_body(txt)
    return {"module": census_of_text(txt), "body": body, "n_while": nwhile}


def count_custom_calls_looped_args(gradfn, p, extra, R=3):
    try:
        txt = looped_args(gradfn, p, R).lower(p, *extra).compile().as_text()
    except Exception as e:
        return {"error": repr(e)}
    body, nwhile = census_while_body(txt)
    return {"module": census_of_text(txt), "body": body, "n_while": nwhile}


def hoisting_report(grad_census, looped_census):
    """Ops present in the plain gradient but missing from the loop body."""
    if not isinstance(looped_census, dict) or "body" not in looped_census:
        return {}
    body = looped_census["body"]
    moved = {}
    for k in ("trsm", "trsm_big", "potrf", "gemm", "dot", "fft"):
        a, b = grad_census.get(k, 0), body.get(k, 0)
        if a != b:
            moved[k] = {"grad": a, "loop_body": b}
    return moved


# ---------------------------------------------------------------------------
# Verbatim copies from benchmarks/h100/bench.py: contention and spawning
# ---------------------------------------------------------------------------
def gpu_contention(parent_pid=None):
    """Who else is on this card?  The A6000 reference numbers were taken on a
    verified-idle card; any comparison must know whether this one was."""
    info = {}
    ignore = {str(os.getpid())}
    if parent_pid:
        ignore.add(str(parent_pid))
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,"
                           "used_gpu_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        procs = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        info["compute_apps"] = procs
        info["n_foreign_compute_apps"] = sum(
            1 for ln in procs
            if ln.split(",")[0].strip() not in ignore)
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


def _spawn(script, tag, extra_argv, out_path, smoke=False, timeout=2400):
    """Run `script` as a child leg and return its JSON.

    bench.py's `_spawn` with the script path and the smoke flag passed in
    explicitly (bench.py reads them from its module globals); otherwise the
    same command shape: --no-sub, --out, --parent-pid, then extra_argv.
    """
    cmd = [sys.executable, os.path.abspath(script),
           "--no-sub", "--out", out_path,
           "--parent-pid", str(os.getpid())] + extra_argv
    if smoke:
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
