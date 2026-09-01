#!/usr/bin/env python
"""Score an H100 benchmark JSON against PREDICTIONS.md and the A6000 reference.

    python analyze.py results_<jobid>.json

Pure stdlib -- no jax, no GPU; run it anywhere, including the login node.
Reference numbers from GPU_BENCHMARKS.md / CLAIMS_VERIFICATION.md are embedded
below so the comparison is self-contained.
"""
import json
import math
import os
import sys

# ===========================================================================
# Embedded reference numbers
# ===========================================================================
# RTX A6000, per-gradient DEVICE time, float64, us/gradient (GPU_BENCHMARKS.md B.6)
A6000_F64 = {
    "2,205,2":   {"current": 3072.0, "whiten_seq": 1412.0,
                  "R1_unroll_concat": 1186.8, "R1_unroll_sep": 1206.6,
                  "R1_vmap": 912.9, "floor_no_likelihood": 66.4},
    "2,1024,2":  {"current": 15847.4, "whiten_seq": 6904.1,
                  "R1_unroll_concat": 6633.9, "R1_unroll_sep": 6695.6,
                  "R1_vmap": 4457.3, "floor_no_likelihood": 74.0},
    "3,1024,8":  {"current": 25092.6, "whiten_seq": 11110.3,
                  "R1_unroll_concat": 17070.8, "R1_unroll_sep": 10465.8,
                  "R1_vmap": 5931.2, "floor_no_likelihood": 115.7},
}
# (2,1024,4) was never measured on the A6000; bracketed estimate for context only.
A6000_F64_EST = {"2,1024,4": {"current": 15000.0, "R1_unroll_sep": 6800.0,
                              "R1_vmap": 4600.0}}

# RTX A6000, float32, us/gradient (GPU_BENCHMARKS.md B.9, throughput harness --
# which agreed with the device harness to 0.2% at these sizes)
A6000_F32 = {
    "2,205,2":  {"current": 1101.5, "R1_unroll_concat": 403.2, "R1_vmap": 429.5},
    "3,1024,8": {"current": 7295.7, "R1_unroll_concat": 2072.7, "R1_vmap": 2511.1},
}
A6000_F32_OVER_F64 = {"2,205,2": {"current": 2.73, "R1_unroll_concat": 2.94,
                                  "R1_vmap": 2.19},
                      "3,1024,8": {"current": 3.41, "R1_unroll_concat": 8.18,
                                   "R1_vmap": 2.35}}

# Workstation CPU (2x Xeon Gold 6244, 16 physical cores @ 3.6 GHz), float64,
# device harness, us/gradient (GPU_BENCHMARKS.md B.6)
WS_CPU_F64 = {
    "2,205,2":  {"current": 768.7, "whiten_seq": 290.8, "R1_unroll_concat": 278.3,
                 "R1_unroll_sep": 273.5, "R1_vmap": 747.5,
                 "floor_no_likelihood": 60.1},
    "2,1024,2": {"current": 11349.1, "R1_unroll_sep": 2663.3, "R1_vmap": 3668.2},
    "3,1024,8": {"current": 40750.3, "R1_unroll_sep": 16314.8, "R1_vmap": 18593.5},
}
# verifier, OMP_NUM_THREADS=1, isolated kernel, (2,205,2): R1 sep 356 us
WS_CPU_OMP1_KERNEL = {"current": 1520.7, "R1_unroll_concat": 540.5,
                      "R1_unroll_sep": 356.0, "R1_vmap": 753.2}

# A6000 chain scaling, R1 vmap, f64, 250+250 (GPU_BENCHMARKS.md B.8)
A6000_CHAINS = {
    "current":       {"1": 40.63, "4": 65.94, "16": 103.00, "64": 180.25,
                      "4_default": 72.23},
    "R1_unroll_sep": {"1": 12.08, "4": 26.98, "16": 27.62, "64": 44.71},
    "R1_vmap":       {"1": 12.84, "4": 15.14, "16": 23.65, "64": 36.22},
}
A6000_THROUGHPUT_VS_1 = {"R1_vmap": {"4": 3.39, "16": 8.68, "64": 22.68},
                         "R1_unroll_sep": {"4": 1.79, "16": 7.00, "64": 17.30},
                         "current": {"16": 6.31}}
# ms per chain-iteration, 250+250, compile included (GPU_BENCHMARKS.md 4.3)
WS_CPU_MS_PER_CHAIN_ITER = {"default4_current": 5.21, "default4_R1": 2.84}

A6000_RHS_205_F64 = {8: 81.7, 16: 131.6, 17: 462.1, 18: 439.7, 32: 278.7,
                     33: 461.4, 40: 379.6}
A6000_COMPILE = {"isolated_grad_prod": (0.46, 1.07), "mcmc_R1_vmap": 9.3}


# ===========================================================================
# Prediction table (see PREDICTIONS.md).  (lo, hi, point)
# ===========================================================================
P = {}
P["P1"] = {"desc": "per-gradient us, float64, (2,205,2)",
           "items": {"current": (2000, 2900, 2450),
                     "R1_unroll_sep": (800, 1150, 980),
                     "R1_unroll_concat": (780, 1130, 960),
                     "R1_vmap": (500, 900, 750)}}
P["P2"] = {"desc": "per-gradient us, float64, (3,1024,8)",
           "items": {"current": (10000, 17000, 13000),
                     "R1_unroll_sep": (4200, 7500, 5800),
                     "R1_vmap": (2500, 4500, 3400)}}
P["P2b"] = {"desc": "per-gradient us, float64, (2,1024,4)",
            "items": {"current": (7000, 12000, 9000),
                      "R1_unroll_sep": (3000, 5500, 4000),
                      "R1_vmap": (2200, 3800, 2900)}}
P["P3"] = {"desc": "no-likelihood floor us, float64, (2,205,2)",
           "items": {"floor_no_likelihood": (50, 95, 70)}}
P["P4"] = {"desc": "R1 speedup ratios preserved, float64, (2,205,2)",
           "items": {"R1_vmap/current": (3.0, 4.2, 3.5),
                     "R1_unroll_sep/current": (2.2, 3.2, 2.6),
                     "R1_vmap/R1_unroll_sep": (1.2, 1.9, 1.4)}}
P["P5"] = {"desc": "GPU f64 / CPU f64, best-to-best, (2,205,2) -- "
                   "PREDICTED: GPU still LOSES",
           "items": {"gpu_over_cpu_best": (1.3, 3.6, 2.3),
                     "gpu_over_cpu_current": (2.0, 3.5, 2.6)}}
P["P6"] = {"desc": "CPU f64 / GPU f64, best-to-best, (3,1024,8) -- GPU WINS",
           "items": {"cpu_over_gpu_best": (3.5, 8.0, 5.0)}}
P["P7"] = {"desc": "chain throughput vs 1 chain, f64, R1_vmap",
           "items": {"4": (3.0, 3.9, 3.4), "16": (9.0, 14.0, 11.0),
                     "64": (22.0, 36.0, 28.0)}}
P["P8"] = {"desc": "cuBLAS trsm RHS threshold at n=205, f64 "
                   "(P(exists)=0.7, P(at 16->17)=0.45)",
           "items": {"max_step_ratio": (1.5, 3.0, 2.2)}}
P["P9"] = {"desc": "compile time",
           "items": {"isolated_grad_R1_vmap_s": (0.5, 1.8, 1.0),
                     "mcmc_R1_vmap_compile_s": (7.0, 18.0, 11.0)}}
P["P10"] = {"desc": "f32/f64 speedup on the SAME card -- predicted SMALLER "
                    "than the A6000's",
            "items": {"(2,205,2) current": (1.4, 2.4, 1.8),
                      "(2,205,2) R1_unroll_sep": (1.4, 2.5, 1.8),
                      "(2,205,2) R1_vmap": (1.3, 2.2, 1.7),
                      "(3,1024,8) R1_vmap": (1.5, 3.0, 2.2),
                      "(3,1024,8) R1_unroll_sep": (2.0, 5.0, 3.0)}}
P["P10b"] = {"desc": "absolute float32 us/gradient, (2,205,2)",
             "items": {"current": (700, 1200, 900),
                       "R1_unroll_sep": (280, 480, 380),
                       "R1_vmap": (300, 500, 390)}}
P["P11"] = {"desc": "THE HEADLINE: same-node CPU float64 vs GPU float32, "
                    "best-to-best, (2,205,2) -- predicted a TIE",
            "items": {"gpuF32_over_cpuF64_best": (0.7, 2.0, 1.2)}}
P["P11b"] = {"desc": "GPU f32 ms per chain-iteration (250+250, compile incl.)",
             "items": {"4": (3.0, 7.0, 4.5), "16": (1.0, 2.4, 1.6),
                       "64": (0.15, 0.55, 0.35)}}
P["P12"] = {"desc": "float32 accuracy/robustness is silicon-independent",
            "items": {}}


# ===========================================================================
# helpers
# ===========================================================================
BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    BOLD = DIM = OFF = ""


def hdr(t):
    print("\n" + BOLD + "=" * 78 + "\n" + t + "\n" + "=" * 78 + OFF)


def sub(t):
    print("\n" + BOLD + "-- " + t + OFF)


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def ug(d, cfg, var):
    """us_per_grad for (cfg, variant) from a devtime dict."""
    return g(d, cfg, var, "us_per_grad")


SCORE = {"HIT": 0, "MISS": 0, "n/a": 0}


def check(pid, name, value, note=""):
    """Score `value` against P[pid]['items'][name]."""
    spec = g(P, pid, "items", name)
    if spec is None:
        return
    lo, hi, pt = spec
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        SCORE["n/a"] += 1
        print("   %-4s %-34s %12s   pred %8.4g-%-8.4g  %s"
              % (pid, name, "n/a", lo, hi, note))
        return
    ok = lo <= value <= hi
    SCORE["HIT" if ok else "MISS"] += 1
    off = ""
    if not ok:
        off = "  (%.2fx %s the interval)" % (
            value / hi if value > hi else lo / value,
            "above" if value > hi else "below")
    print("   %-4s %-34s %12.4g   pred %8.4g-%-8.4g [pt %.4g]  %s%s%s"
          % (pid, name, value, lo, hi, pt,
             BOLD + ("HIT " if ok else "MISS") + OFF, off, note))


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


# ===========================================================================
def main():
    if len(sys.argv) < 2:
        sys.exit("usage: analyze.py <results.json>")
    path = sys.argv[1]
    with open(path) as fh:
        R = json.load(fh)

    env = R.get("env", {})
    f32 = R.get("gpu_f32", {}) or {}
    cpu = R.get("cpu_f64", {}) or {}
    cpuch = R.get("cpu_f64_chains", {}) or {}
    cpu1 = R.get("cpu_f64_omp1", {}) or {}

    dt64 = R.get("devtime", {}) or {}
    dt32 = f32.get("devtime", {}) or {}
    dtcpu = cpu.get("devtime", {}) or {}
    dtcpu1 = cpu1.get("devtime", {}) or {}

    # ---------------------------------------------------------------- env --
    hdr("ENVIRONMENT")
    print("  file            %s" % os.path.abspath(path))
    print("  tag             %s" % R.get("tag"))
    print("  hostname        %s" % env.get("hostname"))
    print("  GPU             %s" % (env.get("jax_device_kinds")))
    print("  nvidia-smi      %s" % env.get("nvidia_smi"))
    print("  CPU             %s  (%s cores in affinity mask)"
          % (env.get("cpu_model"), env.get("cpu_affinity_count")))
    print("  jax/jaxlib      %s / %s" % (g(env, "packages", "jax"),
                                         g(env, "packages", "jaxlib")))
    print("  cublas/cusolver %s / %s" % (g(env, "packages", "nvidia-cublas-cu12"),
                                         g(env, "packages", "nvidia-cusolver-cu12")))
    print("  smoke run       %s" % env.get("smoke"))
    kinds = " ".join(env.get("jax_device_kinds") or [])
    warn = []
    if env.get("smoke"):
        warn.append("this is a --smoke run: reduced reps/configs, several "
                    "sections skipped")
    if "H100" not in kinds.upper():
        warn.append("the GPU is %r, not an H100 -- the PREDICTIONS.md scorecard "
                    "below is written for an H100 and does not apply" % kinds)
    for w in warn:
        print("  " + BOLD + "!! " + w + OFF)
    print("  total wall      %.1f s" % (R.get("t_total_s") or float("nan")))
    for phase in ("contention_before",):
        c = g(env, phase) or {}
        if c.get("n_foreign_compute_apps"):
            print("  " + BOLD + "!! %d foreign compute process(es) on the GPU at "
                  "start: %s" % (c["n_foreign_compute_apps"],
                                 c.get("compute_apps")) + OFF)
            print("     -> absolute timings are CONTENDED; ratios are still usable.")
    for leg, d in (("gpu_f32", f32), ("cpu_f64", cpu),
                   ("cpu_f64_chains", cpuch), ("cpu_f64_omp1", cpu1)):
        if d.get("error"):
            print("  " + BOLD + "!! leg %s FAILED: %s" %
                  (leg, d["error"]) + OFF)
        elif d:
            print("  leg %-14s ok  (%s, %s)"
                  % (leg, g(d, "env", "jax_device_kinds"), g(d, "env", "dtype")))

    # -------------------------------------------------------- correctness --
    hdr("CORRECTNESS (must pass before anything below means anything)")
    corr = R.get("correctness", {}) or {}
    if not corr:
        print("  (section absent)")
    else:
        print("  all variants within 1e-11 of ringdown.model.make_model: %s"
              % (BOLD + str(corr.get("_all_pass")) + OFF))
        for cfg, row in sorted(corr.items()):
            if cfg.startswith("_"):
                continue
            print("   (%-9s) " % cfg + "  ".join(
                "%s=%.1e" % (k, v["pot_and_grad"]) for k, v in row.items()))

    # --------------------------------------------- devtime f64 vs A6000 ----
    hdr("PER-GRADIENT DEVICE TIME, float64 -- H100 vs A6000 vs same-node CPU")
    print("  %-11s %-20s %10s %10s %8s %11s %8s %10s"
          % ("config", "variant", "H100 us", "A6000 us", "A6k/H100",
             "nodeCPU us", "GPU/CPU", "wsCPU us"))
    for cfg in sorted(dt64):
        ref = A6000_F64.get(cfg) or A6000_F64_EST.get(cfg, {})
        est = cfg in A6000_F64_EST and cfg not in A6000_F64
        for var in ("current", "whiten_seq", "R1_unroll_concat", "R1_unroll_sep",
                    "R1_vmap", "floor_no_likelihood"):
            h = ug(dt64, cfg, var)
            if h is None:
                continue
            a = ref.get(var)
            c = ug(dtcpu, cfg, var)
            w = g(WS_CPU_F64, cfg, var)
            print("  %-11s %-20s %10.1f %10s %8s %11s %8s %10s%s"
                  % (cfg, var, h,
                     "%.1f" % a if a else "-",
                     "%.2fx" % (a / h) if a else "-",
                     "%.1f" % c if c else "-",
                     "%.2fx" % (h / c) if c else "-",
                     "%.1f" % w if w else "-",
                     "  (A6000 est.)" if est and a else ""))
    print("  " + DIM + "wsCPU = the ORIGINAL workstation CPU (2x Xeon Gold 6244, "
          "16 cores @3.6GHz), for scale only;\n  nodeCPU is the same-node "
          "baseline and is the one the verdict is based on." + OFF)

    # ------------------------------------------------------ devtime f32 ----
    if dt32:
        hdr("PER-GRADIENT DEVICE TIME, float32 -- and the f32/f64 gap")
        print("  %-11s %-20s %10s %10s %9s %11s %12s"
              % ("config", "variant", "H100 f32", "H100 f64", "f32 gain",
                 "A6000 gain", "A6000 f32 us"))
        for cfg in sorted(dt32):
            for var in ("current", "R1_unroll_concat", "R1_unroll_sep",
                        "R1_vmap", "floor_no_likelihood"):
                h32, h64 = ug(dt32, cfg, var), ug(dt64, cfg, var)
                if h32 is None:
                    continue
                a = g(A6000_F32_OVER_F64, cfg, var)
                aabs = g(A6000_F32, cfg, var)
                print("  %-11s %-20s %10.1f %10s %9s %11s %12s"
                      % (cfg, var, h32, "%.1f" % h64 if h64 else "-",
                         "%.2fx" % (h64 / h32) if h64 else "-",
                         "%.2fx" % a if a else "-",
                         "A6000 %.0f" % aabs if aabs else ""))

    # =====================================================================
    # THE HEADLINE COMPARISONS
    # =====================================================================
    hdr("HEADLINE: best-config-to-best-config, SAME NODE, production point")
    cfg = "2,205,2"

    def best(d, c):
        vals = {v: ug(d, c, v) for v in
                ("current", "whiten_seq", "R1_unroll_concat", "R1_unroll_sep",
                 "R1_vmap") if ug(d, c, v) is not None}
        if not vals:
            return None, None
        k = min(vals, key=vals.get)
        return k, vals[k]

    gv64, gt64 = best(dt64, cfg)
    gv32, gt32 = best(dt32, cfg)
    cv, ct = best(dtcpu, cfg)
    cv1, ct1 = best(dtcpu1, cfg)
    rows = [("GPU float64", gv64, gt64), ("GPU float32", gv32, gt32),
            ("CPU float64 (all cores)", cv, ct),
            ("CPU float64 (OMP=1, prod)", cv1, ct1)]
    for lbl, v, t in rows:
        print("  %-28s %-20s %s"
              % (lbl, v or "-", "%9.1f us/gradient" % t if t else "n/a"))
    print()
    r_f64 = ratio(gt64, ct)
    r_f32 = ratio(gt32, ct)
    r_f32_omp1 = ratio(gt32, ct1)
    if r_f64:
        print("  " + BOLD + "GPU f64 / CPU f64  = %.2fx  -> GPU is %s" % (
            r_f64, "SLOWER" if r_f64 > 1 else "FASTER") + OFF
            + "   [A6000 was 3.34-4.00x slower]")
    if r_f32:
        print("  " + BOLD + "GPU f32 / CPU f64  = %.2fx  -> GPU is %s" % (
            r_f32, "SLOWER" if r_f32 > 1 else "FASTER") + OFF
            + "   <<< the comparison you actually run")
    if r_f32_omp1:
        print("       (against the production CPU threading regime, OMP=1: %.2fx)"
              % r_f32_omp1)
    print("\n  Workstation/A6000 reference for scale: CPU f64 best %.1f us "
          "(OMP=1 kernel: %.1f), A6000 f64 best %.1f us, A6000 f32 best %.1f us."
          % (WS_CPU_F64["2,205,2"]["R1_unroll_sep"],
             WS_CPU_OMP1_KERNEL["R1_unroll_sep"],
             A6000_F64["2,205,2"]["R1_vmap"], A6000_F32["2,205,2"]["R1_vmap"]))

    # big config
    for cfg2 in ("3,1024,8", "2,1024,4"):
        if cfg2 not in dt64:
            continue
        gv, gt = best(dt64, cfg2)
        gv3, gt3 = best(dt32, cfg2)
        cvb, ctb = best(dtcpu, cfg2)
        if gt and ctb:
            print("\n  at (%s): CPU f64 best %s=%.0f us; GPU f64 best %s=%.0f us "
                  "-> GPU %.2fx %s" % (cfg2, cvb, ctb, gv, gt, max(ctb / gt, gt / ctb),
                                       "FASTER" if gt < ctb else "SLOWER"))
            if gt3:
                print("  at (%s): GPU f32 best %s=%.0f us -> GPU %.2fx %s than CPU f64"
                      % (cfg2, gv3, gt3, max(ctb / gt3, gt3 / ctb),
                         "FASTER" if gt3 < ctb else "SLOWER"))

    # ------------------------------------------------------------ chains --
    hdr("VECTORIZED-CHAIN SCALING (production point, 250+250, compile included)")
    for legname, ch in (("float64", R.get("chains", {})),
                        ("float32", f32.get("chains", {}))):
        if not ch:
            continue
        sub("GPU %s" % legname)
        print("  %-16s %7s %9s %11s %13s %13s"
              % ("variant", "chains", "wall_s", "vs_1chain", "ms/chain-it",
                 "A6000 f64 ms"))
        for nm, row in ch.items():
            for C, d in sorted(row.items(), key=lambda kv: (len(kv[0]), kv[0])):
                if not isinstance(d, dict) or "wall_s" not in d:
                    continue
                aw = g(A6000_CHAINS, nm, C)
                ams = (aw / (int(C) * 500) *
                       1e3) if (aw and C.isdigit()) else None
                at = g(A6000_THROUGHPUT_VS_1, nm, C)
                print("  %-16s %7s %9.2f %10s %13.3f %13s %s"
                      % (nm, C, d["wall_s"],
                         "%.2fx" % d["throughput_vs_1chain"]
                         if d.get("throughput_vs_1chain") else "-",
                         d["ms_per_chain_iteration"],
                         "%.3f" % ams if ams else "-",
                         "(A6000 f64 %.2fx)" % at if at else ""))
    cch = (cpuch.get("chains") or cpu.get("chains") or {})
    if cch:
        sub("same-node CPU float64 (the baseline the GPU must beat)")
        for nm, row in cch.items():
            for C, d in row.items():
                if isinstance(d, dict) and "ms_per_chain_iteration" in d:
                    print("  %-16s %-14s %9.2f s  %8.3f ms/chain-iteration"
                          % (nm, C, d["wall_s"], d["ms_per_chain_iteration"]))
        print("  [workstation reference: 4-chain default, current model %.2f "
              "ms/chain-it; R1 %.2f]"
              % (WS_CPU_MS_PER_CHAIN_ITER["default4_current"],
                 WS_CPU_MS_PER_CHAIN_ITER["default4_R1"]))

    # crossover
    cpu_best_ms = None
    for nm, row in cch.items():
        d = row.get("4_default")
        if isinstance(d, dict) and d.get("ms_per_chain_iteration"):
            if cpu_best_ms is None or d["ms_per_chain_iteration"] < cpu_best_ms:
                cpu_best_ms = d["ms_per_chain_iteration"]
    if cpu_best_ms:
        sub("CROSSOVER: how many vectorized GPU chains to beat the best CPU "
            "4-chain run (%.2f ms/chain-iteration)" % cpu_best_ms)
        for legname, ch in (("float64", R.get("chains", {})),
                            ("float32", f32.get("chains", {}))):
            for nm in ("R1_vmap", "R1_unroll_sep"):
                row = (ch or {}).get(nm, {})
                win = [int(C) for C, d in row.items()
                       if C.isdigit() and isinstance(d, dict)
                       and d.get("ms_per_chain_iteration", 1e9) < cpu_best_ms]
                print("  GPU %-8s %-16s beats the CPU at chains >= %s"
                      % (legname, nm, min(win) if win else "(none tested)"))

    # --------------------------------------------------------- rhs sweep --
    hdr("cuBLAS triangular-solve RHS threshold")
    for legname, rs in (("float64", R.get("rhs_sweep", {})),
                        ("float32", f32.get("rhs_sweep", {}))):
        for n, row in (rs or {}).items():
            ms = row.get("_max_step", {})
            t8, t16 = row.get("8"), row.get("16")
            print("  %-8s n=%-5s largest step: k=%s -> k=%s  x%.2f      "
                  "t(16)/t(8) = %s"
                  % (legname, n, ms.get("from"), ms.get("to"),
                     ms.get("ratio", float("nan")),
                     "%.2f" % (t16 / t8) if (t8 and t16) else "-"))
            if n == "205" and legname == "float64":
                print("           A6000 f64 reference: %s"
                      % {k: round(v, 1) for k, v in A6000_RHS_205_F64.items()})

    # ------------------------------------------------------- f32 accuracy --
    acc = f32.get("f32acc", {}) or {}
    if acc:
        hdr("FLOAT32 ACCURACY / ROBUSTNESS (P12) -- expected to be "
            "silicon-independent")
        print("  %-20s %-18s %11s %11s %8s" % ("case (modes|a_scale_max)",
                                               "variant", "rel_err_U",
                                               "rel_err_grad", "finite"))
        nan_current, nan_r1 = [], []
        for case, row in acc.items():
            for var, d in row.items():
                if not isinstance(d, dict) or "rel_err_U" not in d:
                    continue
                print("  %-20s %-18s %11.2e %11.2e %8s"
                      % (case, var, d["rel_err_U"], d["rel_err_grad"],
                         d["all_finite"]))
                if not d["all_finite"]:
                    (nan_current if var == "current" else nan_r1).append(case)
        print()
        print("  cases where `current` lost float32 (NaN): %s" %
              (nan_current or "none"))
        print("  cases where an R1 form lost float32     : %s" %
              (nan_r1 or "none"))
        ok = bool(nan_current) and not set(nan_current) <= set(nan_r1)
        print("  " + BOLD + "P12 robustness asymmetry (current NaNs where R1 does "
              "not): %s" % ("REPRODUCED" if ok else "NOT reproduced") + OFF)
        print("  A6000 reference: potential ~5e-7 and gradient ~4e-4 at benign")
        print("  conditioning; `current` NaN at high cond where R1 stays finite.")
        print("  " + DIM + "If the gradient errors here are >10x worse than the "
              "A6000's, suspect TF32\n  lowering of the f32 dots on sm_90 -- re-run "
              "with jax.default_matmul_precision('float32')." + OFF)

    # ---------------------------------------------------------- compile ----
    hdr("COMPILE TIME")
    for legname, cp in (("float64", R.get("compile", {})),
                        ("float32", f32.get("compile", {}))):
        ig = g(cp or {}, "isolated_grad") or {}
        for cfgk, row in ig.items():
            print("  %-8s isolated jit(grad) (%s): %s" % (
                legname, cfgk,
                "  ".join("%s=%.2fs" % (k, v) for k, v in row.items())))
            if cfgk == "2,205,2":
                print("           [A6000 f64 reference: %.2f-%.2f s]"
                      % A6000_COMPILE["isolated_grad_prod"])
        for nm, d in (g(cp or {}, "mcmc") or {}).items():
            if isinstance(d, dict) and "compile_setup_s" in d:
                print("  %-8s MCMC %-10s compile+setup ~%.2f s, %.3f ms/iter "
                      "(A6000 f64: ~9.3 s)"
                      % (legname, nm, d["compile_setup_s"], d["ms_per_iter"]))

    # ==================================================================== #
    hdr("SCORECARD vs PREDICTIONS.md")
    if env.get("smoke") or "H100" not in kinds.upper():
        print(BOLD + "  !! NOT AN H100 RUN and/or a --smoke run: the scorecard "
              "below is a\n     self-test of the machinery, not a result." + OFF)
    sub("P1 / P2 / P2b -- absolute per-gradient float64")
    for pid, cfgk in (("P1", "2,205,2"), ("P2", "3,1024,8"), ("P2b", "2,1024,4")):
        for var in P[pid]["items"]:
            check(pid, var, ug(dt64, cfgk, var), note="  [%s]" % cfgk)
    sub("P3 -- the launch-overhead floor is not silicon")
    check("P3", "floor_no_likelihood", ug(
        dt64, "2,205,2", "floor_no_likelihood"))
    sub("P4 -- R1 ratios preserved")
    c0 = ug(dt64, "2,205,2", "current")
    check("P4", "R1_vmap/current", ratio(c0, ug(dt64, "2,205,2", "R1_vmap")))
    check("P4", "R1_unroll_sep/current",
          ratio(c0, ug(dt64, "2,205,2", "R1_unroll_sep")))
    check("P4", "R1_vmap/R1_unroll_sep",
          ratio(ug(dt64, "2,205,2", "R1_unroll_sep"), ug(dt64, "2,205,2", "R1_vmap")))
    sub("P5 / P6 -- CPU vs GPU, float64")
    check("P5", "gpu_over_cpu_best", r_f64)
    check("P5", "gpu_over_cpu_current",
          ratio(ug(dt64, "2,205,2", "current"), ug(dtcpu, "2,205,2", "current")))
    gvb, gtb = best(dt64, "3,1024,8")
    cvb, ctb = best(dtcpu, "3,1024,8")
    check("P6", "cpu_over_gpu_best", ratio(ctb, gtb))
    sub("P7 -- chain throughput, float64, R1_vmap")
    for C in ("4", "16", "64"):
        check("P7", C, g(R, "chains", "R1_vmap", C, "throughput_vs_1chain"))
    sub("P8 -- trsm RHS threshold, float64, n=205")
    check("P8", "max_step_ratio", g(R, "rhs_sweep", "205", "_max_step", "ratio"),
          note="  at k=%s->%s" % (g(R, "rhs_sweep", "205", "_max_step", "from"),
                                  g(R, "rhs_sweep", "205", "_max_step", "to")))
    sub("P9 -- compile")
    check("P9", "isolated_grad_R1_vmap_s",
          g(R, "compile", "isolated_grad", "2,205,2", "R1_vmap"))
    check("P9", "mcmc_R1_vmap_compile_s",
          g(R, "compile", "mcmc", "R1_vmap", "compile_setup_s"))
    sub("P10 / P10b -- float32")
    for key, (cfgk, var) in {"(2,205,2) current": ("2,205,2", "current"),
                             "(2,205,2) R1_unroll_sep": ("2,205,2", "R1_unroll_sep"),
                             "(2,205,2) R1_vmap": ("2,205,2", "R1_vmap"),
                             "(3,1024,8) R1_vmap": ("3,1024,8", "R1_vmap"),
                             "(3,1024,8) R1_unroll_sep": ("3,1024,8",
                                                          "R1_unroll_sep")}.items():
        check("P10", key, ratio(ug(dt64, cfgk, var), ug(dt32, cfgk, var)))
    for var in P["P10b"]["items"]:
        check("P10b", var, ug(dt32, "2,205,2", var))
    sub("P11 -- THE HEADLINE: CPU float64 vs GPU float32")
    check("P11", "gpuF32_over_cpuF64_best", r_f32)
    for C in ("4", "16", "64"):
        check("P11b", C, g(f32, "chains", "R1_vmap", C, "ms_per_chain_iteration"))

    # --------------------------------------------------------- falsifiers --
    hdr("FALSIFICATION TESTS for the 'latency-/launch-bound' diagnosis")
    v = ug(dt64, "2,205,2", "R1_vmap")
    fl = ug(dt64, "2,205,2", "floor_no_likelihood")
    f32r = ratio(ug(dt64, "2,205,2", "R1_vmap"),
                 ug(dt32, "2,205,2", "R1_vmap"))
    rs = g(R, "rhs_sweep", "205") or {}
    t8, t16 = rs.get("8"), rs.get("16")
    small = ratio(A6000_F64["2,205,2"]["R1_vmap"], v)
    big = ratio(A6000_F64["3,1024,8"]["R1_vmap"],
                ug(dt64, "3,1024,8", "R1_vmap"))
    tests = [
        ("F1", v is not None and v <= 305, "R1_vmap f64 = %s us (falsify at <=305)"
         % ("%.1f" % v if v else "n/a")),
        ("F2", fl is not None and fl < 33, "floor = %s us (falsify at <33)"
         % ("%.1f" % fl if fl else "n/a")),
        ("F3", f32r is not None and f32r >= 2.5,
         "f32/f64 for R1_vmap = %s (falsify at >=2.5; the A6000 measured 2.19x "
         "for this variant, and the H100's FP64 handicap is 16x smaller)"
         % ("%.2f" % f32r if f32r else "n/a")),
        ("F4", (t8 and t16) and (t16 / t8) >= 1.8, "t(k=16)/t(k=8) = %s "
         "(falsify at >=1.8)" % ("%.2f" % (t16 / t8) if (t8 and t16) else "n/a")),
        ("F5", (small and big) and abs(small - big) / max(small, big) < 0.20,
         "A6000/H100 speedup: %s at (2,205,2) vs %s at (3,1024,8) "
         "(falsify if equal to within 20%%)"
         % ("%.2f" % small if small else "n/a", "%.2f" % big if big else "n/a")),
    ]
    any_f = False
    n_eval = 0
    for fid, fired, msg in tests:
        evaluable = "n/a" not in msg
        n_eval += evaluable
        any_f |= bool(fired)
        print("  %-4s %-10s %s"
              % (fid, BOLD + ("FALSIFIED" if fired else
                              ("ok" if evaluable else "no data")) + OFF, msg))
    if any_f:
        verdict = ("*** THE LATENCY-BOUND DIAGNOSIS IS FALSIFIED -- at least one "
                   "test fired. ***")
    elif n_eval >= 3:
        verdict = ("Latency-bound diagnosis SURVIVES: %d/%d tests evaluable, none "
                   "fired." % (n_eval, len(tests)))
    else:
        verdict = ("INSUFFICIENT DATA: only %d/%d falsification tests could be "
                   "evaluated." % (n_eval, len(tests)))
    print("\n  " + BOLD + verdict + OFF)

    hdr("SCORE: %d HIT / %d MISS / %d n-a" % (SCORE["HIT"], SCORE["MISS"],
                                              SCORE["n/a"]))
    tot = SCORE["HIT"] + SCORE["MISS"]
    if tot:
        print("  %.0f%% of scoreable predictions landed inside their interval."
              % (100.0 * SCORE["HIT"] / tot))
    print("  (An 80% interval should contain ~80% of outcomes; far above that "
          "means\n   the intervals were too wide, far below means the model is "
          "wrong.)\n")


if __name__ == "__main__":
    main()
