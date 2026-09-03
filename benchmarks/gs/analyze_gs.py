#!/usr/bin/env python
"""Render the tables of a bench_gs.py results JSON.

    python benchmarks/gs/analyze_gs.py results.json          # plain text
    python benchmarks/gs/analyze_gs.py results.json --md     # markdown

Pure stdlib -- no jax, no numpy; run it anywhere.  Accepts either the parent
JSON (with a 'legs' dict) or a single leg JSON.  Older JSONs (without the
cloud-normalized / by-kind / spread keys) render with the fields they have.

Tables
------
  env          one line per leg: backend, devices, dtype, BLAS threads (OMP),
               XLA:CPU pool threads (NPROC / tf_XLAEigen census), matmul
               precision, ringdown git; caveats (TF32, FP64-throttled GPUs)
  correctness  per leg (f64): rel. error vs main and vs reference (cloud-
               normalized when available, per-point otherwise), gate/flags;
               per-kind breakdown; digits lost beyond conditioning
               err/(eps cond C); GS variants with the other Yule-Walker
               filter (solve_toeplitz vs longdouble-refined); scale twins
  devtime      us/grad per variant x config per leg (+- repeatability spread),
               speedup vs main and vs main_hoisted, dropped cells, thread
               configuration and hoisting caveats, spread of identical-
               executable (@fast == @pow2) and same-shape family pairs
  compile      compile seconds per variant x config per leg
  breakdown    f32/f64 accuracy vs cond(C): rows = (config, family) sorted by
               cond, cols = variants, cell = worst rel. gradient error vs the
               reference; labelled with the matmul precision; per-kind rows;
               spectra-policy diagnostic in f32 legs
  census       HLO op counts (trsm/potrf/gemm/dot/fft) on the plain gradient
               and hoisting differences in the looped form
  nuts         cold/warm wall, compile, us/leapfrog, ESS (per seed), ESS/s,
               posterior mean/sd per variant, gradient share of a leapfrog
"""
import json
import math
import re
import sys

MD = "--md" in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
if not ARGS:
    print(__doc__)
    sys.exit(1)

VARIANT_ORDER = ["main", "main_hoisted", "gemm_linv", "gemm_cinv", "gs_pr",
                 "gs_pr_ascoded", "gs_full", "gs_half", "floor"]
GS_BASES = ("gs_pr", "gs_pr_ascoded", "gs_full", "gs_half")
EPS64 = 2.220446049250313e-16
# GPUs whose FP64 rate is 1/32 or 1/64 of FP32 (consumer / workstation parts)
FP64_THROTTLED = re.compile(r"RTX|GeForce|Quadro|A6000|A40|A10|A16|L40|L4\b|T4|A2\b", re.I)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def fnum(x, fmt="%.1f"):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "nan" if math.isnan(x) else "inf"
        return fmt % x
    except Exception:
        return str(x)


def fexp(x):
    return fnum(x, "%.1e")


def table(headers, rows, title=None):
    """Print a fixed-width (or markdown) table."""
    rows = [[str(c) for c in r] for r in rows]
    headers = [str(h) for h in headers]
    if title:
        print(("\n### %s" % title) if MD else ("\n%s" % title))
    if not rows:
        print("  (no data)")
        return
    if MD:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for r in rows:
            print("| " + " | ".join(r) + " |")
        return
    w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
    print("  " + "  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        print("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(headers))))


def heading(txt):
    print(("\n## %s" % txt) if MD else ("\n" + "=" * 78 + "\n" + txt + "\n" + "=" * 78))


def note(txt):
    print(("\n> " + txt) if MD else ("  ! " + txt))


def sort_variants(keys):
    def rank(k):
        base = k.split("@")[0]
        return (VARIANT_ORDER.index(base) if base in VARIANT_ORDER else 99, k)
    return sorted([k for k in keys if not k.startswith("_")], key=rank)


def cfg_parse(k):
    a, b, c = (int(x) for x in k.split(","))
    return a, b, c


def cfg_sort(keys):
    def rank(k):
        try:
            a, b, c = cfg_parse(k)
            return (b, a, c)
        except Exception:
            return (1e9, 0, 0)
    return sorted([k for k in keys if not k.startswith("_")], key=rank)


def legs_of(doc):
    if "legs" in doc:
        return {k: v for k, v in doc["legs"].items() if isinstance(v, dict)}
    return {doc.get("leg", "leg"): doc}


def nfft_pow2(N):
    return 1 << (2 * N - 2).bit_length()


def nfft_fast(N):
    """Smallest 5-smooth integer >= 2N-1 (scipy.fft.next_fast_len(real=True))."""
    target = 2 * N - 1
    best = nfft_pow2(N)
    p5 = 1
    while p5 < best:
        p3 = p5
        while p3 < best:
            p2 = p3
            while p2 < target:
                p2 *= 2
            best = min(best, p2)
            p3 *= 3
        p5 *= 5
    return best


def nfft_of_cell(x, ck, vk):
    """Resolved FFT length of a GS devtime cell (recorded, else derived)."""
    if isinstance(x, dict) and x.get("nfft"):
        return x["nfft"]
    try:
        N = cfg_parse(ck)[1]
    except Exception:
        return None
    mode = vk.split("@")[1] if "@" in vk else (x.get("nfft_mode") if isinstance(x, dict) else None)
    if mode == "fast":
        return nfft_fast(N)
    if mode == "pow2":
        return nfft_pow2(N)
    return None


def leg_threads(leg):
    """(OMP_NUM_THREADS, NPROC, xla pool census) of a leg, from env or devtime.

    JSONs written before NPROC was recorded: bench_gs.py's --omp then ALSO
    exported NPROC=--omp (coupled), so NPROC is inferred from the argv.
    """
    e = leg.get("env") or {}
    te = e.get("thread_env") or {}
    dv = leg.get("devtime") if isinstance(leg.get("devtime"), dict) else {}
    th = dv.get("_threads") or {}
    nproc = te.get("NPROC", th.get("NPROC"))
    if "NPROC" not in te and "NPROC" not in th:
        argv = [str(a) for a in (leg.get("argv") or [])]
        if "--omp" in argv and "--xla-threads" not in argv:
            nproc = "%s (coupled to --omp by the old kit)" % argv[argv.index("--omp") + 1]
    return (te.get("OMP_NUM_THREADS"), nproc,
            e.get("xla_cpu_pool_threads", th.get("xla_cpu_pool_threads")))


def is_gpu_leg(leg):
    return (leg.get("env") or {}).get("jax_backend") in ("gpu", "cuda")


def matmul_label(leg, sec=None):
    e = leg.get("env") or {}
    prec = (sec or {}).get("_matmul_precision", e.get("jax_default_matmul_precision"))
    dtype = (sec or {}).get("_dtype", leg.get("dtype", e.get("dtype", "")))
    if is_gpu_leg(leg) and "32" in str(dtype) and str(prec) in ("None", "", "default", "bfloat16"):
        return "matmul precision %s: TF32 dots (JAX default on Ampere+ GPUs)" % prec
    return "matmul precision %s" % prec


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def show_env(legs, doc):
    heading("ENVIRONMENT")
    if "tag" in doc and doc["tag"]:
        print("tag: %s" % doc["tag"])
    rows = []
    for nm, leg in legs.items():
        if "error" in leg and "env" not in leg:
            rows.append([nm, "ERROR: %s" % leg["error"]] + [""] * 9)
            continue
        e = leg.get("env", {})
        omp, nproc, pool = leg_threads(leg)
        rows.append([nm, e.get("jax_backend"), ",".join(e.get("jax_device_kinds", [])),
                     e.get("dtype"), omp,
                     "%s / %s" % (nproc if nproc is not None else "unset",
                                  pool if pool is not None else "?"),
                     e.get("fft_backend_hint"), e.get("jax_default_matmul_precision"),
                     "%s%s" % (e.get("ringdown_git"),
                               "" if e.get("ringdown_matches_kit", True) else " (NOT KIT)"),
                     fnum(leg.get("t_total_s"), "%.0f")])
    table(["leg", "backend", "device", "dtype", "OMP", "NPROC / XLA pool", "fft",
           "matmul_prec", "ringdown", "wall_s"], rows)
    print("  OMP = BLAS/LAPACK threads (OMP/MKL/OPENBLAS_NUM_THREADS: trsm, potrf, GEMM); "
          "NPROC sizes XLA:CPU's Eigen intra-op pool (FFT, dot, fusions), 'unset' = one "
          "thread per core (production: `import ringdown` sets only OMP_NUM_THREADS=1); "
          "XLA pool = tf_XLAEigen thread census when recorded.")
    for nm, leg in legs.items():
        e = leg.get("env", {})
        c = e.get("contention_before") or {}
        if c.get("n_foreign_compute_apps"):
            print("  ! %s: %d foreign compute process(es) on the GPU" %
                  (nm, c["n_foreign_compute_apps"]))
        if e.get("hostname"):
            print("  %s: host %s, cpu %s (%s cores), configs %s, families %s%s" %
                  (nm, e.get("hostname"), e.get("cpu_model"), e.get("cpu_affinity_count"),
                   ";".join(",".join(str(x) for x in c) for c in e.get("configs", [])),
                   ",".join(e.get("families", [])),
                   ("; gs_coeffs=%s spectra_from=%s logdetC_source=%s"
                    % (e.get("gs_coeffs"), e.get("spectra_from"), e.get("logdetC_source")))
                   if e.get("gs_coeffs") else ""))
        if e.get("logdetC_source") == "f64lev":
            note("%s: inputs carry the float64 Levinson log|C| (biased by ~eps cond N "
                 "nats); the potential (nats) columns of the hoisted variants include "
                 "that constant.  Run prep_inputs.py --refresh-precompute." % nm)
    for nm, leg in legs.items():
        e = leg.get("env", {})
        if is_gpu_leg(leg) and "64" in str(e.get("dtype", "")) and \
                any(FP64_THROTTLED.search(k or "") for k in e.get("jax_device_kinds", [])):
            note("%s: %s has 1/32-1/64 FP64 throughput; its float64 timings are a "
                 "smoke test, not a production GPU number (see submit_h100.sbatch)."
                 % (nm, ",".join(e.get("jax_device_kinds", []))))
        if is_gpu_leg(leg) and "32" in str(e.get("dtype", "")) and \
                str(e.get("jax_default_matmul_precision")) in ("None", "default"):
            note("%s: float32 dots run in TF32 (JAX default on Ampere+); its accuracy "
                 "tables measure TF32 rounding (~1e-2..1e-1), not the C^{-1} route.  "
                 "Read float32 route accuracy from a --matmul-precision highest leg "
                 "(gpu_f32_hi)." % nm)


def _cell_err(x, key):
    """cloud-normalized value when present, else the per-point one."""
    if not isinstance(x, dict):
        return None
    v = x.get(key + "_cloud")
    return v if v is not None else x.get(key)


def show_correctness(legs):
    heading("CORRECTNESS (float64): likelihood part U_var - U_floor")
    print("cell = max rel. gradient error vs main / vs reference; [OK|FAIL] is the "
          "1e-11 algebra gate (white, expcos), [ok|concerning|fail] the reference flag "
          "(> 1e-8 concerning, > 1e-6 fail).")
    print("Normalization: |g - g_ref| / (max over ALL points of max|g_ref[site]|) per site "
          "('cloud') when the JSON has it, else per point (max|g_ref[site]| at the same "
          "point, which blows up near the mode where the warmup points sit).")
    for nm, leg in legs.items():
        sec = leg.get("correctness")
        if not isinstance(sec, dict) or "skipped" in sec or "error" in sec:
            continue
        rows = []
        variants = set()
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                if isinstance(cell, dict):
                    variants |= set(k for k in cell if not k.startswith("_"))
        variants = sort_variants(variants)
        cloud = any(isinstance(cell, dict) and cell.get("_gate_normalization") == "cloud"
                    for ck in cfg_sort(sec) for cell in sec[ck].values())
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                if not isinstance(cell, dict) or "error" in cell:
                    rows.append([ck, fam, "ERR"] + [str((cell or {}).get("error", ""))[:40]]
                                + [""] * (len(variants) - 1))
                    continue
                r = [ck, fam, fexp(cell.get("_cond"))]
                for v in variants:
                    x = cell.get(v)
                    if not isinstance(x, dict):
                        r.append("-")
                    elif "error" in x:
                        r.append("ERR")
                    else:
                        tag = ""
                        if x.get("gate") is not None:
                            tag = "OK" if x["gate"] else "FAIL"
                        elif x.get("flag"):
                            tag = x["flag"]
                        r.append("%s/%s%s" % (fexp(_cell_err(x, "rel_grad_vs_main")),
                                              fexp(_cell_err(x, "rel_grad_vs_ref")),
                                              (" [%s]" % tag) if tag else ""))
                rows.append(r)
        table(["config", "family", "cond(C)"] + variants, rows,
              "leg %s (gate all pass: %s; normalization: %s)"
              % (nm, sec.get("_all_gate_pass"), "cloud" if cloud else "per point"))
        if not cloud:
            note("per-point normalization: the FAIL cells at (3,1024,8) white/expcos and "
                 "(2,2048,4) white come from warmup points where |g_chi|, |g_m| are ~0.02-0.4 "
                 "against a cloud scale of ~70; cloud-normalized the same differences are "
                 "<= 2e-13 (gate passes with 50x margin).  Rerun for the cloud numbers.")
        # nats table vs reference
        rows = []
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                if not isinstance(cell, dict) or "error" in cell or not cell.get("_has_ref"):
                    continue
                r = [ck, fam]
                for v in variants:
                    x = cell.get(v)
                    r.append(fexp(x.get("nats_vs_ref")) if isinstance(x, dict) else "-")
                rows.append(r)
        if rows:
            src = {cell.get("_logdetC_source") for ck in cfg_sort(sec)
                   for cell in sec[ck].values() if isinstance(cell, dict)} - {None}
            table(["config", "family"] + variants, rows,
                  "leg %s: |U_lik - U_ref| in nats (inputs' log|C|: %s)"
                  % (nm, ",".join(sorted(src)) if src else "float64 Levinson sum (pre-refresh "
                     "inputs: the hoisted variants' nats include its ~eps cond N bias)"))
        # digits lost beyond conditioning
        rows = []
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                if not isinstance(cell, dict) or "error" in cell or not cell.get("_has_ref"):
                    continue
                cond = cell.get("_cond")
                r = [ck, fam, fexp(cond)]
                for v in variants:
                    x = cell.get(v)
                    val = None
                    if isinstance(x, dict):
                        val = x.get("err_over_eps_cond")
                        if val is None and cond and x.get("rel_grad_vs_ref") is not None:
                            val = x["rel_grad_vs_ref"] / (EPS64 * cond)
                    r.append(fnum(val, "%.2g"))
                rows.append(r)
        if rows:
            table(["config", "family", "cond(C)"] + variants, rows,
                  "leg %s: err / (eps cond C), digits lost beyond conditioning (~1 = as "
                  "good as the conditioning allows; >> 10 = the route loses accuracy the "
                  "problem does not force)" % nm)
        # per point kind
        rows = []
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                if not isinstance(cell, dict) or "error" in cell:
                    continue
                kinds = sorted((cell.get("_n_pts_by_kind") or {}).keys())
                if len(kinds) < 2:
                    continue
                for kn in kinds:
                    r = [ck, fam, "%s (n=%d)" % (kn, cell["_n_pts_by_kind"][kn])]
                    for v in variants:
                        x = cell.get(v)
                        if not isinstance(x, dict) or "error" in x:
                            r.append("-")
                            continue
                        bk = x.get("rel_grad_vs_ref_cloud_by_kind") or x.get("rel_grad_vs_ref_by_kind") or {}
                        bm = x.get("rel_grad_vs_main_cloud_by_kind") or x.get("rel_grad_vs_main_by_kind") or {}
                        r.append("%s/%s" % (fexp(bm.get(kn)), fexp(bk.get(kn))))
                    rows.append(r)
        if rows:
            table(["config", "family", "kind"] + variants, rows,
                  "leg %s: by point kind (normal = N(0,1) draws, warmup = NUTS typical set): "
                  "vs main / vs ref" % nm)
        # alternative Yule-Walker coefficients
        rows = []
        pol = None
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                if not isinstance(cell, dict) or "error" in cell:
                    continue
                alt = {v: cell[v].get("alt_coeffs") for v in variants
                       if isinstance(cell.get(v), dict) and cell[v].get("alt_coeffs")}
                if not alt:
                    continue
                r = [ck, fam, fexp(cell.get("_cond"))]
                for v in variants:
                    a = alt.get(v)
                    x = cell.get(v)
                    if not a or "error" in a:
                        r.append("-" if not a else "ERR")
                        continue
                    pol = a.get("policy")
                    r.append("%s -> %s [%s]" % (fexp(_cell_err(x, "rel_grad_vs_ref")),
                                                fexp(_cell_err(a, "rel_grad_vs_ref")),
                                                a.get("flag") or "-"))
                rows.append(r)
        if rows:
            table(["config", "family", "cond(C)"] + variants, rows,
                  "leg %s: GS variants, same executable with the %s Yule-Walker filter "
                  "(run policy -> alternative), rel. gradient error vs ref" % (nm, pol))
            note("The two policies differ only in the constants a, atilde, sigma^2 "
                 "(solve_toeplitz as PR #141 vs longdouble Levinson rounded to f64); the "
                 "difference is the precompute's forward error (F1), not the FFT route.")
        # scale twins
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                st = cell.get("_scale_twin") if isinstance(cell, dict) else None
                if st:
                    if "error" in st:
                        print("  scale twin %s/%s vs %s: ERR %s" % (ck, fam, st.get("twin"), st["error"][:60]))
                    else:
                        print("  scale twin %s/%s vs %s (scale %.3e): main gradient rel. diff "
                              "%.1e (cloud), |dU - n_det N log scale| = %.1e nats"
                              % (ck, fam, st["twin"], st["scale"], st["rel_grad_cloud"],
                                 st["nats_minus_expected"]))


def _spread_pct(a, b):
    m = 0.5 * (a + b)
    return abs(a - b) / m * 100.0 if m else float("nan")


def show_devtime(legs):
    heading("DEVICE TIME PER GRADIENT (us/grad)")
    print("Slope method over two fori_loop counts, medians of rep timings; '+-x%' is half "
          "the range of the slope over the individual repetitions (repeatability, not a "
          "confidence interval), when recorded.")
    for nm, leg in legs.items():
        sec = leg.get("devtime")
        if not isinstance(sec, dict) or "error" in sec:
            continue
        omp, nproc, pool = leg_threads(leg)
        gpu = is_gpu_leg(leg)
        if not gpu:
            note("%s thread configuration: BLAS threads (OMP) = %s, XLA:CPU pool: NPROC = %s"
                 "%s.  CPU numbers depend on BOTH: with a multi-thread XLA pool the batched-"
                 "FFT variants (gs_full, gs_half), the Eigen dots (gemm_*) and the prior-"
                 "only floor run 2-3x SLOWER at N >= 1024 than with a 1-thread pool while "
                 "main's trsm gets faster; gs_pr's per-column vmapped FFTs are insensitive.  "
                 "Production (`import ringdown`: OMP_NUM_THREADS=1, NPROC unset) is the "
                 "cpu_f64_prod leg; quote CPU speedups with the thread setting."
                 % (nm, omp, nproc if nproc is not None else "unset (one thread per core)",
                    (", %s tf_XLAEigen threads" % pool) if pool is not None else ""))
        fams = sorted({f for ck in cfg_sort(sec) for f in sec[ck]})
        for fam in fams:
            variants = set()
            for ck in cfg_sort(sec):
                variants |= set(sec[ck].get(fam, {}).keys())
            variants = sort_variants(variants)
            rows = []
            for ck in cfg_sort(sec):
                cell = sec[ck].get(fam)
                if not cell:
                    continue
                r = [ck]
                for v in variants:
                    x = cell.get(v)
                    if not isinstance(x, dict):
                        r.append("-")
                    elif "error" in x:
                        r.append("ERR")
                    else:
                        sp = x.get("us_spread") or {}
                        txt = fnum(x.get("us_per_grad"))
                        if sp.get("us_min") is not None and x.get("us_per_grad"):
                            half = 0.5 * (sp["us_max"] - sp["us_min"]) / x["us_per_grad"] * 100
                            txt += " +-%.0f%%" % half
                        r.append(txt)
                rows.append(r)
            table(["config"] + variants, rows, "leg %s, family %s: us/grad" % (nm, fam))
            rows = []
            for ck in cfg_sort(sec):
                cell = sec[ck].get(fam)
                if not cell:
                    continue
                r = [ck]
                for v in variants:
                    x = cell.get(v)
                    if isinstance(x, dict) and "us_per_grad" in x:
                        r.append("%s / %s" % (fnum(x.get("speedup_vs_main"), "%.2f"),
                                              fnum(x.get("speedup_vs_main_hoisted"), "%.2f")))
                    else:
                        r.append("-")
                rows.append(r)
            table(["config"] + variants, rows,
                  "leg %s, family %s: speedup vs main / vs main_hoisted" % (nm, fam))
        # hoisting caveat: which variants had ops moved out of the timing loop
        moved = {}
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                for v, x in cell.items():
                    if isinstance(x, dict) and x.get("hoisted"):
                        moved.setdefault(v, set()).add(
                            ",".join("%s:%d->%d" % (k, d["grad"], d["loop_body"])
                                     for k, d in x["hoisted"].items()))
        if moved:
            lst = "; ".join("%s (%s)" % (v, "/".join(sorted(m))) for v, m in sorted(moved.items()))
            if gpu:
                note("%s: op counts differ between the plain gradient and the loop body for %s "
                     "(census 'moved' column).  On GPU these are cuBLAS-vs-HLO-dot lowering "
                     "differences inside the body, not hoisting: XLA:GPU keeps main's N x 1 "
                     "solves and gs_pr_ascoded's constant rffts in the loop, which is why "
                     "main_hoisted is 4-6%% (f64) / 8-19%% (f32) faster than main and "
                     "gs_pr_ascoded ~10%% slower than gs_pr here but not on CPU." % (nm, lst))
            else:
                note("%s: XLA:CPU hoisted theta-independent work out of the timing loop for %s "
                     "(census 'moved' column).%s  XLA:GPU hoists nothing, so 'speedup vs main' "
                     "means different things on the two platforms; use 'vs main_hoisted' as the "
                     "route comparison everywhere.  Inside NUTS tree building the same loop-"
                     "invariant code motion applies on CPU (z paid once per trajectory)."
                     % (nm, lst,
                        "  For `main` these are the two N x 1 solves z = L^{-1} y, so on this "
                        "leg main ~= main_hoisted and 'speedup vs main' excludes the per-gradient "
                        "recomputation production pays on GPU." if "main" in moved else ""))
        # identical-executable pairs: @fast and @pow2 with the same nfft
        rows = []
        worst = []
        for ck in cfg_sort(sec):
            for fam in fams:
                cell = sec[ck].get(fam) or {}
                for base in GS_BASES:
                    a, b = cell.get(base + "@fast"), cell.get(base + "@pow2")
                    if not (isinstance(a, dict) and isinstance(b, dict)
                            and "us_per_grad" in a and "us_per_grad" in b):
                        continue
                    na, nb = nfft_of_cell(a, ck, base + "@fast"), nfft_of_cell(b, ck, base + "@pow2")
                    if na is not None and na == nb:
                        pct = _spread_pct(a["us_per_grad"], b["us_per_grad"])
                        worst.append(pct)
                        rows.append([ck, fam, base, na, fnum(a["us_per_grad"]),
                                     fnum(b["us_per_grad"]), "%.1f%%" % pct])
        if rows:
            table(["config", "family", "variant", "nfft", "@fast us", "@pow2 us", "spread"], rows,
                  "leg %s: identical executables timed twice (@fast and @pow2 resolve to the "
                  "same nfft for N >= 1024): per-cell repeatability" % nm)
            worst.sort()
            print("  median spread %.1f%%, max %.1f%% over %d pairs: speedup differences below "
                  "~%.2fx on this leg are not resolved."
                  % (worst[len(worst) // 2], worst[-1], len(worst), 1 + worst[-1] / 100))
        # same-shape family pairs
        if len(fams) >= 2:
            rows = []
            for ck in cfg_sort(sec):
                cells = [sec[ck].get(f) or {} for f in fams]
                vs = sort_variants(set.intersection(*[set(c) for c in cells]) if cells else set())
                for v in vs:
                    us = [c[v].get("us_per_grad") for c in cells
                          if isinstance(c.get(v), dict) and c[v].get("us_per_grad")]
                    if len(us) >= 2:
                        rows.append([ck, v] + [fnum(u) for u in us]
                                    + ["%.1f%%" % _spread_pct(min(us), max(us))])
            if rows:
                table(["config", "variant"] + fams + ["spread"], rows,
                      "leg %s: same shape, different ACF family (identical executable, "
                      "different constant values): spread" % nm)
        dropped = sec.get("_dropped") or []
        if dropped:
            print("  dropped cells (%s):" % nm)
            for d in dropped:
                print("    %s: %s" % (d.get("cell"), d.get("reason")))


def show_compile(legs):
    heading("COMPILE TIME of jit(grad) (s)")
    for nm, leg in legs.items():
        sec = leg.get("compile")
        if not isinstance(sec, dict) or "error" in sec:
            continue
        variants = set()
        for ck in cfg_sort(sec):
            variants |= set(k for k in sec[ck] if not k.startswith("_"))
        variants = sort_variants(variants)
        rows = []
        for ck in cfg_sort(sec):
            r = [ck]
            for v in variants:
                x = sec[ck].get(v)
                r.append(fnum(x.get("total_s"), "%.2f") if isinstance(x, dict)
                         and "total_s" in x else ("ERR" if isinstance(x, dict) else "-"))
            rows.append(r)
        table(["config"] + variants, rows, "leg %s (lower + compile)" % nm)


def show_breakdown(legs):
    heading("ACCURACY vs cond(C): worst rel. gradient error vs the reference")
    print("Rows sorted by cond(C); 'cloud' = normalized by the point cloud's per-site gradient "
          "scale (per point otherwise).  The per-kind table below is the one to read trends "
          "from: the warmup (typical-set) points carry a k x k-tail cancellation that the "
          "N(0,1) points do not, and several cells (e.g. aligo2 (3,1024,8), the non-gw150914 "
          "(2,4096,4) files) have NO warmup points, so the overall column mixes populations.")
    for nm, leg in legs.items():
        sec = leg.get("f32acc")
        if not isinstance(sec, dict) or "error" in sec:
            continue
        dtype = sec.get("_dtype", leg.get("dtype", ""))
        thr = 1e-3 if "32" in str(dtype) else 1e-8
        mm = matmul_label(leg, sec)
        cells = []
        variants = set()
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                if isinstance(cell, dict) and "error" not in cell:
                    cells.append((cell.get("_cond") or float("nan"), ck, fam, cell))
                    variants |= set(k for k in cell if not k.startswith("_"))
        variants = sort_variants(variants)
        cells.sort(key=lambda t: (t[0] if t[0] == t[0] else float("inf"), t[1], t[2]))
        has_cloud = any(isinstance(c[3].get(v), dict) and c[3][v].get("rel_grad_vs_ref_cloud") is not None
                        for c in cells for v in variants)
        metrics = [("rel_grad_vs_ref", "gradient (per point)")]
        if has_cloud:
            metrics.append(("rel_grad_vs_ref_cloud", "gradient (cloud)"))
        metrics += [("nats_vs_ref", "potential (nats)"), ("rel_grad_vs_twin", "gradient vs f64 twin")]
        if has_cloud:
            metrics.append(("rel_grad_vs_twin_cloud", "gradient vs f64 twin (cloud)"))
        for metric, label in metrics:
            rows = []
            any_val = False
            marked = set()
            for cond, ck, fam, cell in cells:
                r = [fexp(cond), ck, fam]
                for v in variants:
                    x = cell.get(v)
                    val = x.get(metric) if isinstance(x, dict) else None
                    if val is None:
                        r.append("-")
                        continue
                    any_val = True
                    s = fexp(val)
                    if not x.get("all_finite", True):
                        s += " !nonfinite"
                    elif metric.startswith("rel_grad_vs_ref") and (v not in marked) and \
                            isinstance(val, float) and val > thr:
                        s += " <-- >%.0e" % thr
                        marked.add(v)
                    r.append(s)
                rows.append(r)
            if any_val:
                table(["cond(C)", "config", "family"] + variants, rows,
                      "leg %s (%s; %s): %s (threshold mark %.0e%s)"
                      % (nm, dtype, mm, label, thr,
                         "; twin = %s" % sec.get("_producer_leg")
                         if "twin" in metric else ""))
        if sec.get("_tf32_possible") or (is_gpu_leg(leg) and "32" in str(dtype)
                                          and "TF32" in mm):
            note("%s: these numbers are set by TF32 matmul rounding in the k x k and N x k "
                 "dots (white noise, cond 1, shows the same 4e-2..1e-1 for every variant); "
                 "with --matmul-precision highest they drop 50-600x to the cpu_f32 level and "
                 "the variant ordering changes.  Not a statement about the C^{-1} routes." % nm)
        # per kind
        rows = []
        for cond, ck, fam, cell in sorted(cells, key=lambda t: (cfg_sort([t[1]]), t[0], t[2])):
            kinds = sorted((cell.get("_n_pts_by_kind") or {}).keys())
            if not kinds:
                continue
            for kn in kinds:
                r = [ck, fam, fexp(cond), "%s (n=%d)" % (kn, cell["_n_pts_by_kind"][kn])]
                for v in variants:
                    x = cell.get(v)
                    if not isinstance(x, dict):
                        r.append("-")
                        continue
                    bk = x.get("rel_grad_vs_ref_cloud_by_kind") or x.get("rel_grad_vs_ref_by_kind") or {}
                    r.append(fexp(bk.get(kn)))
                rows.append(r)
        if rows:
            table(["config", "family", "cond(C)", "kind"] + variants, rows,
                  "leg %s (%s; %s): gradient vs ref by point kind%s, sorted by config then cond"
                  % (nm, dtype, mm, " (cloud)" if has_cloud else ""))
        # spectra policy diagnostic (f32 legs)
        rows = []
        pol = None
        for cond, ck, fam, cell in cells:
            alt = {v: cell[v].get("alt_spectra") for v in variants
                   if isinstance(cell.get(v), dict) and cell[v].get("alt_spectra")}
            if not alt:
                continue
            r = [fexp(cond), ck, fam]
            for v in variants:
                a = alt.get(v)
                x = cell.get(v)
                if not a or "error" in a:
                    r.append("-" if not a else "ERR")
                    continue
                pol = a.get("policy")
                r.append("%s -> %s" % (fexp(_cell_err(x, "rel_grad_vs_ref")),
                                       fexp(_cell_err(a, "rel_grad_vs_ref"))))
            rows.append(r)
        if rows:
            table(["cond(C)", "config", "family"] + variants, rows,
                  "leg %s: GS variants with spectra from policy %s -> %s (run policy -> "
                  "alternative; 'f64' = f64 rfft cast to complex64, 'leg' = rfft of the "
                  "f32-cast filter as gs_pr_ascoded does), gradient vs ref"
                  % (nm, sec.get("_spectra_from"), pol))


def show_census(legs):
    heading("HLO CENSUS on the plain gradient (first timing family) and hoisting")
    for nm, leg in legs.items():
        sec = leg.get("devtime")
        if not isinstance(sec, dict) or "error" in sec:
            continue
        rows = []
        for ck in cfg_sort(sec):
            for fam, cell in sec[ck].items():
                for v in sort_variants(cell):
                    x = cell[v]
                    if not isinstance(x, dict) or "census_grad" not in x:
                        continue
                    c = x["census_grad"]
                    body = (x.get("census_looped") or {}).get("body") or {}
                    hoist = x.get("hoisted") or {}
                    rows.append([ck, v] + [str(c.get(k, 0)) for k in
                                           ("trsm", "trsm_big", "potrf", "gemm", "dot", "fft")]
                                + [str(c.get("fft_types", "")), str(c.get("hlo_lines", "")),
                                   str(body.get("fft", "-")),
                                   ",".join("%s:%d->%d" % (k, d["grad"], d["loop_body"])
                                            for k, d in hoist.items()) or ""])
        table(["config", "variant", "trsm", "trsm_NxN", "potrf", "gemm", "dot", "fft",
               "fft types", "hlo", "fft/loop", "moved (grad->loop body)"], rows,
              "leg %s (trsm_NxN = triangular solves with an N-sized dimension; the rest "
              "is the k x k tail every variant shares; 'moved' = ops hoisted out of the "
              "fori_loop body by XLA, CPU only)" % nm)


def _devtime_us(leg, cfg_list, fam, variant):
    """us/grad of a variant at a config/family from the leg's devtime, or None."""
    sec = leg.get("devtime")
    if not isinstance(sec, dict) or not cfg_list:
        return None
    ck = ",".join(str(c) for c in cfg_list)
    cell = (sec.get(ck) or {}).get(fam) or {}
    for k in (variant, variant + "@pow2", variant + "@fast"):
        x = cell.get(k)
        if isinstance(x, dict) and x.get("us_per_grad"):
            return x["us_per_grad"]
    return None


def show_nuts(legs):
    heading("NUTS")
    for nm, leg in legs.items():
        sec = leg.get("nuts")
        if not isinstance(sec, dict) or "skipped" in sec or "error" in sec:
            continue
        new = any(isinstance(sec[k], dict) and "warm" in sec[k] for k in sec)
        rows = []
        for v in sort_variants([k for k in sec if isinstance(sec[k], dict)
                                and k not in ("threads",)]):
            x = sec[v]
            if "error" in x:
                rows.append([v, "ERR", x["error"][:50]] + [""] * (10 if new else 6))
                continue
            gus = _devtime_us(leg, sec.get("config"), sec.get("family"), v)
            if new:
                warm = x.get("warm") or []
                uspl = x.get("us_per_leapfrog_warm")
                rows.append([v, fnum(x["cold"]["wall_s"]), fnum(x.get("compile_s_est")),
                             "/".join(fnum(w["wall_s"]) for w in warm),
                             "/".join(str(w["num_steps"]) for w in warm),
                             "%s [%s]" % (fnum(uspl, "%.0f"),
                                          "..".join(fnum(u, "%.0f") for u in
                                                    (x.get("us_per_leapfrog_warm_range") or []))),
                             ("%s (%.0f%%)" % (fnum(gus, "%.0f"), 100 * gus / uspl))
                             if gus and uspl else "-",
                             "/".join(fnum(w["ess"]["m"][0], "%.0f") for w in warm),
                             "/".join(fnum(w["ess_per_s"]["m"][0], "%.1f") for w in warm),
                             "/".join("%s+-%s" % (fnum(w["mean"]["m"][0], "%.2f"),
                                                  fnum(w["sd"]["m"][0], "%.2f")) for w in warm),
                             "/".join(fnum(w["mean"]["chi"][0], "%.4f") for w in warm)])
            else:
                rows.append([v, fnum(x["wall_s"]), str(x["num_steps"]),
                             fnum(x.get("us_per_leapfrog_incl_compile"), "%.0f"),
                             fnum(gus, "%.0f"),
                             fnum(x["ess"]["m"][0], "%.0f"), fnum(x["ess"]["chi"][0], "%.0f"),
                             "/".join(fnum(e, "%.0f") for e in x["ess"]["a_scale"]),
                             fnum(x["ess_per_s"]["m"][0], "%.1f"),
                             "%s +- %s" % (fnum(x["mean"]["m"][0], "%.3f"), fnum(x["sd"]["m"][0], "%.3f")),
                             "%s +- %s" % (fnum(x["mean"]["chi"][0], "%.4f"), fnum(x["sd"]["chi"][0], "%.4f"))])
        title = "leg %s: %s %s, %s+%s, %d chain(s)" % (
            nm, ",".join(str(c) for c in sec.get("config", [])), sec.get("family"),
            sec.get("num_warmup"), sec.get("num_samples"), sec.get("num_chains", 1))
        if new:
            table(["variant", "cold wall_s", "compile ~s", "warm wall_s (per seed)",
                   "steps", "us/leapfrog warm [range]", "us/grad devtime (share)",
                   "ESS m", "ESS(m)/s warm", "m", "chi"], rows,
                  title + ", warm seeds %s" % sec.get("seeds_warm"))
            note("Throughput from the WARM runs (compile excluded); ESS per seed, whose spread "
                 "at this size exceeds the variant-to-variant differences, so the ESS/s column "
                 "ranks nothing with < ~4 seeds.  'share' = device us/grad over us/leapfrog: "
                 "the fraction of a leapfrog step a faster gradient can save.")
        else:
            table(["variant", "wall_s", "steps", "us/leapfrog (incl. compile)",
                   "us/grad devtime", "ESS m", "ESS chi", "ESS a_scale",
                   "ESS(m)/s", "m", "chi"], rows, title)
            note("Single cold run per variant: wall_s INCLUDES the sampler's JIT compile "
                 "(~10 s for main vs ~0.2 s for gemm_linv at 300+300), so ESS(m)/s and "
                 "us/leapfrog here rank compile time, not throughput; one chain, one seed: "
                 "seed-to-seed ESS spread (main 111-181) exceeds the variant differences.  "
                 "Posterior means are consistent.  Rerun for cold/warm timings.")


def main():
    with open(ARGS[0]) as fh:
        doc = json.load(fh)
    legs = legs_of(doc)
    if MD:
        print("# GS benchmark results: %s" % ARGS[0])
    show_env(legs, doc)
    show_correctness(legs)
    show_devtime(legs)
    show_compile(legs)
    show_breakdown(legs)
    show_census(legs)
    show_nuts(legs)
    errs = []
    for nm, leg in legs.items():
        for k, v in leg.items():
            if isinstance(v, dict) and "error" in v and k not in ("f32acc",):
                errs.append("%s/%s: %s" % (nm, k, str(v["error"])[:80]))
        if "error" in leg:
            errs.append("%s: %s" % (nm, leg["error"]))
    if errs:
        heading("SECTION ERRORS")
        for e in errs:
            print("  " + e)


if __name__ == "__main__":
    main()
