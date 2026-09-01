# A-priori predictions for the ringdown model on a single NVIDIA H100

**Written before any H100 data exists.** Every number below is a commitment made from
the A6000 time attribution in `GPU_BENCHMARKS.md` and `CLAIMS_VERIFICATION.md`, so that
the H100 run can be *scored* rather than merely described. `analyze.py` checks each
numbered prediction against the JSON that `bench.py` emits and prints HIT / MISS / n/a.

Scope: **one** H100, one or many chains via `chain_method='vectorized'`. Multi-GPU is
out of scope. Both precisions are in scope: float32 is a production GPU mode here
(`ringdown/cli/ringdown_fit.py:94` maps a `float32` flag onto `jax_enable_x64=False`).

---

## 1. The reasoning, in one page

### 1.1 What the A6000 measurement actually says

The attribution probe in `GPU_BENCHMARKS.md` §5 decomposes R1-`vmap` at the production
point `(n_det, n_analyze, n_modes) = (2, 205, 2)`, float64, per gradient:

| component | µs | share |
|---|---|---|
| priors + unconstraining transforms, no likelihood (the fixed floor) | 66.8 | 7.5% |
| `rd_design_matrix` (all the `exp`/`sin`/`cos`) | +1.8 | 0.2% |
| likelihood with `L @ M` substituted for `L⁻¹M` (same shapes, same FLOPs) | +195 | 22% |
| the triangular structure itself (`L⁻¹M` minus `L @ M`) | +625 | **70%** |
| **total** | **888** | |

Three independent facts pin down *why* that 70% is expensive, and none of them is "not
enough FLOP/s":

1. **Achieved arithmetic rate: 2.4 GFLOP/s = 0.2% of the A6000's ~1.2 TFLOP/s FP64
   peak.** `nvidia-smi` sampled 0% utilization throughout.
2. **A single `n=205` triangular solve is almost flat in its right-hand-side count.**
   From the verifier's sweep: RHS 1 → 74.5 µs, RHS 8 → 81.7 µs, RHS 16 → 131.6 µs.
   Sixteen times the arithmetic for 1.8× the time. At least ~80% of a small `trsm` is
   fixed cost paid per solve, i.e. the `n/blocksize` chain of dependent block steps and
   the kernel launches that implement them — **not** arithmetic.
3. **The GEMM probe.** Identical shapes and identical FLOP count run 3.4–10.6× faster
   once the sequential dependency is removed.

Memory bandwidth is equally not the constraint: at `n=1024` a forward `trsm` touches
8.4 MB of `L`, which is 11 µs of A6000 bandwidth against a measured 770 µs — **1.4%**
utilization.

**So the model is launch-/latency-bound at every size tested, in the precise sense that
the arithmetic and the memory traffic per *dependent step* are both negligible against
the fixed cost of taking that step.**

### 1.2 What an H100 changes, and what it cannot

| H100 advantage | factor vs A6000 | does it touch the bottleneck? |
|---|---|---|
| FP64 vector throughput (PCIe ~26, SXM ~34 TFLOP/s vs 1.2) | 21–28× | **No.** We are at 0.2% of the *A6000's* peak. |
| FP64 : FP32 ratio (1:2 vs 1:32) | 16× | Only via §1.3 below — it changes what float32 buys. |
| HBM3/HBM2e bandwidth (2.0–3.35 TB/s vs 0.77) | 2.6–4.4× | **No.** 1.4% utilized. |
| SM count (114 PCIe / 132 SXM vs 84) | 1.4–1.6× | Only at large batch (many chains). |
| SM boost clock (1.76–1.98 GHz vs 1.80) | 0.98–1.10× | **This is the one that matters**, and it is ~1. |
| newer cuBLAS/cuSOLVER kernels for sm_90 | ? | Possibly the largest single unknown. |
| host-driven kernel launch latency | ~1× (same jaxlib, same CUDA runtime; host core may be *slower* than the workstation's 3.6 GHz Xeon) | **This is the bottleneck, and it does not improve.** |

The bundled CUDA runtime is identical on both machines (the pip `jax[cuda12]` wheels
ship their own cuBLAS 12.9.2.10 / cuSOLVER 11.7.5.82), so **library version is held
fixed**; only the architecture-conditioned kernel *selection* inside those libraries
differs. That matters for prediction 8.

### 1.3 Why float32 must gain *less* on H100 than on the A6000

On the A6000, float32 bought 2.2–2.9× at the production point. If the model were
FLOP-bound that number would be ~32× (the card's FP64:FP32 ratio). It is not, so the
measured 2.2–2.9× decomposes into a small arithmetic component plus a larger
traffic/occupancy/kernel-width component (half the bytes per element, half the register
pressure, wider tiles per wave).

On the H100 the arithmetic component can be **at most 2×** by construction (1:2 ratio),
and given §1.1 it should be ~1.0–1.1×. The traffic/occupancy component is unchanged.
**Therefore the H100's float32-over-float64 gain should be strictly smaller than the
A6000's.** This is the single sharpest falsification test in the whole exercise: it is
a *directional* prediction that a naive "more FLOPs → faster" model gets backwards.

### 1.4 What I cannot predict: the CPU side

The A6000 comparison used *this workstation's* CPU: 2× Xeon Gold 6244, **16 physical
cores at 3.6 GHz** — an unusually high-clock part. A rusty GPU node's host cores are a
different, unknown CPU (Slurm reports ~88 cores/node across the `gpu` partition, so
plausibly Xeon Ice Lake / Sapphire Rapids or EPYC Genoa, at 2.0–3.0 GHz base).

The model's CPU cost is dominated by small LAPACK calls at `n = 205`, which parallelise
poorly, so **single-thread performance is what matters** and the node is likely 0.7–1.1×
the workstation per core. I therefore predict a *range* for the node CPU and let the
same-node measurement settle it. `bench.py` runs the CPU leg on the same node, in a
subprocess with `JAX_PLATFORMS=cpu` exported before `import jax` (the verifier
documented that setting it after the import fails silently and yields GPU numbers
labeled "CPU"), and also runs an `OMP_NUM_THREADS=1` leg because the production CLI
sets exactly that (`ringdown/cli/ringdown_fit.py:80-82`).

**What I can predict is the ratio structure, not the CPU absolutes.**

---

## 2. Numbered predictions

Reference column = measured A6000 (device-time harness, `GPU_BENCHMARKS.md` B.6/B.9)
unless noted. Ranges are ~80% intervals; the point value is my central estimate.

### P1 — per-gradient device time, float64, production point (2, 205, 2)

| variant | A6000 | **H100 prediction** | point |
|---|---|---|---|
| current (shipped) | 3072 µs | **2000 – 2900 µs** | 2450 |
| R1 unrolled, separate solves | 1207 µs | **800 – 1150 µs** | 980 |
| R1 unrolled, `[M\|y]` concat | 1187 µs | **780 – 1130 µs** | 960 |
| R1 `vmap` | 913 µs | **500 – 900 µs** | 750 |

**H100/A6000 = 1.0 – 1.8×, point 1.25×** — a rounding error against 21× the FP64 FLOP/s.

### P2 — per-gradient device time, float64, large point (3, 1024, 8)

| variant | A6000 | **H100 prediction** | point |
|---|---|---|---|
| current | 25 093 µs | **10 000 – 17 000 µs** | 13 000 |
| R1 unrolled, separate | 10 466 µs | **4 200 – 7 500 µs** | 5 800 |
| R1 `vmap` | 5 931 µs | **2 500 – 4 500 µs** | 3 400 |

**H100/A6000 = 1.3 – 2.4×, point 1.75×.** Deliberately *larger* than P1: at `k = 32`,
`n = 1024` the achieved rate reaches ~8% of A6000 FP64 peak, so there is finally some
arithmetic for the H100 to remove. The mid point `(2, 1024, 4)` was not measured on the
A6000; I predict H100 `current` 7 000–12 000, `R1_unroll_sep` 3 000–5 500, `R1_vmap`
2 200–3 800 µs.

### P3 — the launch-overhead floor is *not* silicon

The no-likelihood model (identical priors, identical design matrix, trivial factor)
measured **66.4 µs/gradient** on the A6000 at (2,205,2) and ~74–116 µs at every larger
size — i.e. size-independent, therefore fixed cost.

**Prediction: 50 – 95 µs on the H100**, i.e. within ±40% and *not* a factor of several.
If the host core is slower than the workstation's, it may go *up*.

### P4 — the R1 speedup ratios are preserved

Both the fast and the slow variants are launch-bound, and their ratio is set by op count
(HLO `__cublas$triangularSolve` 42 → 8/10 → 7, `cusolver_potrf_ffi` `n_det` → 1), which
is architecture-independent. The floor is only 7.5% of R1-`vmap`, so even a large trsm
speedup compresses the ratio only slightly.

| ratio (float64, production) | A6000 | **H100 prediction** |
|---|---|---|
| R1 `vmap` / current | 3.37× | **3.0 – 4.2×** |
| R1 unrolled (sep) / current | 2.55× | **2.2 – 3.2×** |
| `vmap` / unrolled (the R2 flip) | 1.32× | **1.2 – 1.9×**, `vmap` still wins |

At `(3, 1024, 8)`: R1 `vmap`/current 4.23× → predict **3.5 – 5.5×**.

### P5 — best-to-best CPU vs GPU at the production point, **float64**: the verdict does **not** flip

* Node CPU, R1 unrolled-separate, all allocated cores: **250 – 450 µs** (workstation: 274).
* Node CPU, same, `OMP_NUM_THREADS=1`: **330 – 600 µs** (workstation: 356, verifier).
* H100, R1 `vmap`, float64: **500 – 900 µs** (P1).

**Prediction: the H100 remains 1.3 – 3.6× slower than the same node's CPU, point 2.3×.
Confidence 80%.** The A6000 margin was 3.3–4.0×, so the gap narrows by roughly half but
does not close. Current-model-to-current-model: GPU **2.0 – 3.5×** slower (A6000: 2.8–4.0×).

### P6 — at (3, 1024, 8) the GPU wins, and by more than on the A6000

A6000 best-to-best at that size: GPU 2.75× faster. **Prediction for the H100: GPU
3.5 – 8× faster than the same node's CPU, point 5×.**

### P7 — vectorized-chain scaling, float64, R1 `vmap`, 250+250, production point

Throughput relative to one chain (A6000: 3.39 / 8.68 / 22.68 at 4 / 16 / 64):

| chains | A6000 | **H100 prediction** |
|---|---|---|
| 4 | 3.39× | **3.0 – 3.9×** |
| 16 | 8.68× | **9 – 14×** |
| 64 | 22.68× | **22 – 36×** |

ms per chain-iteration at 64 vectorized chains (compile included): A6000 1.13 →
**0.45 – 0.85 ms**. Break-even against the same node's 4-chain CPU default is predicted
at **4 – 12 vectorized chains** in float64 (A6000: 4–16).

### P8 — the cuBLAS `trsm` RHS threshold: software, but architecture-conditioned

The A6000 shows a **kernel-selection threshold at RHS ≥ 17** (not an odd/even effect —
the verifier's correction), worth **3.5×** at `n = 205`, with an anomalously fast point
at exactly RHS 32. The bundled cuBLAS is *the same build* on both machines, so this is
not a library-version question. But cuBLAS carries separate kernel families and
heuristic tables per compute capability, and 16 is a natural FP64 column-tile width that
Hopper kernels frequently widen to 32 or 64.

**Prediction, float64, n = 205:**
* P(a discontinuity of ≥1.5× exists somewhere in RHS 8..40) = **0.7**
* P(it sits exactly at 16 → 17) = **0.45**; otherwise most likely at 32 → 33.
* If present, magnitude **1.5 – 3.0×** (down from 3.5×).

If **no** cliff appears, the `R2b` recommendation ("drop the `[M|y]` concat on GPU when
unrolling") becomes A6000-specific and should be re-scoped rather than shipped as a
general GPU rule.

**In float32** the tiles are naturally wider, so if a threshold exists at all I expect
it at a *higher* RHS than in float64; P(any ≥1.5× step in 8..40 in f32) = **0.5**.

*A6000 calibration taken with this kit:* the float64 threshold reproduces at
**3.57x** at `k=16 -> 17` (against the verifier's independently measured 3.51x), and
**in float32 the A6000 shows no cliff at all** -- `t(16)/t(8) = 1.05` and the largest
single step over 8..40 is only 1.19x. So the "drop the `[M|y]` concat" rule (R2b) is
already float64-specific on the A6000. If the H100 behaves the same way, R2b should be
scoped to *float64 GPU runs only*.

### P9 — compile time

| | A6000 | **H100 prediction** |
|---|---|---|
| isolated `jit(grad)`, (2,205,2), f64 | 0.46 – 1.07 s | **0.5 – 1.8 s** |
| full `MCMC.run` compile+setup, R1 `vmap`, f64 | ~9.3 s | **7 – 18 s** |

Compilation is host work plus sm_90 autotuning; it does not benefit from the device and
may cost more. **Predicted direction: equal or worse, never much better.**

### P10 — float32 speed: the sharpest falsification test

f32-over-f64 speedup, same card, per gradient:

| config / variant | A6000 measured | **H100 prediction** |
|---|---|---|
| (2,205,2) current | 2.73× | **1.4 – 2.4×** |
| (2,205,2) R1 unrolled | 2.94× | **1.4 – 2.5×** |
| (2,205,2) R1 `vmap` | 2.19× | **1.3 – 2.2×** |
| (3,1024,8) R1 `vmap` | 2.35× | **1.5 – 3.0×** |
| (3,1024,8) R1 unrolled | 8.18× | **2.0 – 5.0×** |

**Directional commitment: the H100's f32/f64 ratio at the production point will be
LOWER than the A6000's**, despite the H100 having 16× better *relative* FP64 hardware.
That is backwards under a FLOP-bound model and is exactly what a latency-bound model
predicts. Absolute H100 float32 predictions at (2,205,2): current **700 – 1200 µs**,
R1 unrolled **280 – 480 µs**, R1 `vmap` **300 – 500 µs**.

### P11 — the comparison the user actually runs: same-node CPU float64 vs GPU float32

* Node CPU, float64, best formulation: **250 – 450 µs** (P5).
* H100, float32, best formulation: **300 – 500 µs** (P10).

**Prediction: at the production point with ONE chain this is a TIE.** GPU/CPU ratio
**0.7 – 2.0×, point 1.2× (GPU marginally slower).** Float32 roughly halves the H100's
deficit and brings it to parity — it does **not** deliver a clear single-chain win.

With chains, float32 moves the crossover earlier than float64:

| configuration (production point, 250+250, compile included) | **H100 f32 prediction, ms per chain-iteration** |
|---|---|
| 1 chain, R1 `vmap` | 10 – 22 (compile-dominated) |
| 4 vectorized | 3.0 – 7.0 |
| 16 vectorized | 1.0 – 2.4 |
| 64 vectorized | **0.15 – 0.55** |
| *same node*, CPU f64, 4 chains, shipped default, current model | 5 – 9 |
| *same node*, CPU f64, 4 chains, R1 | 2.8 – 5.0 |

**Break-even against the node's best CPU configuration: 4 – 8 vectorized chains in
float32** (vs 4–12 in float64). At 16 chains the H100 in float32 should be **3 – 8×**
faster than the best CPU configuration; at 64 chains **8 – 25×**. For a production
1000+1000 run compile amortizes further and the crossover moves earlier still.

### P12 — float32 accuracy and robustness are silicon-independent

`GPU_BENCHMARKS.md` §3.2 and the verifier both found, on the A6000, that (a) the
float32 *gradient* is ~3 orders of magnitude less accurate than the float32 log density
at realistic conditioning, and (b) the **shipped sequential form returns NaN** at high
`cond(A)` where R1 still returns a finite value. These are IEEE-754 properties of the
algorithms, not of the card.

**Prediction: the H100 reproduces both, with relative errors within a factor of ~3 of
the A6000's and the identical NaN pattern (`current` NaN, `R1_*` finite).**

*Caveat that makes this worth measuring:* XLA may lower float32 dots onto TF32 tensor
cores, and the sm_90 default could differ from sm_86. **If the H100's float32 gradient
errors come back more than ~10× worse than the A6000's, suspect TF32 in the `WᵀW`
contraction and re-run with `jax.default_matmul_precision('float32')` before drawing
any conclusion about float32 in production.**

*(Local A6000 re-measurement with this kit, for calibration, at `cond(A⁻¹) ≈ 12`:
potential 5.1e-7 for every variant; gradient 3.7e-4 `current` / 7.6e-4 `R1_unroll_sep` /
7.6e-4 `R1_vmap`; `current` NaN from `cond ≈ 1.1e5` upward while both R1 forms stay
finite. Note this reproduces the NaN asymmetry and the potential-vs-gradient gap, but
**not** `GPU_BENCHMARKS.md`'s claim that R1's float32 gradient is 7× more accurate — in
this measurement R1 is ~2× worse on the gradient at benign conditioning while being far
more robust at bad conditioning. Treat "R1 is more accurate in f32" as unsupported;
"R1 is more *robust* in f32" is well supported.)*

---

## 2b. A6000 re-measurement with this kit (calibration, not an H100 result)

Before handing the kit over I ran it on this workstation's A6000 on a quiet card.
It reproduces `GPU_BENCHMARKS.md` closely and adds two numbers the original study
never reported, both of which sharpen the predictions above. **These are A6000
numbers; the H100 predictions above stand as written.**

| quantity, (2,205,2) | this kit, A6000/workstation |
|---|---|
| GPU f64 best (`R1_vmap`) | 897.5 us  (published 912.9) |
| GPU f32 best (`R1_unroll_sep`) | 352.2 us |
| CPU f64 best, all cores | 272.3 us  (published 273.5) |
| CPU f64 best, `OMP_NUM_THREADS=1` | 332.6 us  (verifier 356) |
| no-likelihood floor | 66.9 us  (published 66.4) |
| **GPU f64 / CPU f64** | **3.30x**  (published 3.34x) |
| **GPU f32 / CPU f64** | **1.29x**  (1.06x against OMP=1) |

The second bold row is new and it is the single most useful anchor for **P11**:
the *A6000* in float32 is already within 30% of this high-clock workstation CPU in
float64. Combining it with P1's predicted H100/A6000 factor of 1.0-1.8x gives
0.7-1.3x for the H100, i.e. the **lower half** of P11's stated 0.7-2.0 interval;
and a rusty node's host cores are likely *slower* than a 3.6 GHz Xeon 6244, which
pushes it further toward the GPU. **P11 stays as committed, but if I were writing
it now I would put the point estimate at ~1.0 rather than 1.2 -- a true tie, with a
real chance of a narrow single-chain GPU win.**

Also measured here and worth carrying into the comparison: at `(2, 1024, 4)` the
A6000 already beats the workstation CPU by **1.38x** in float64 and **3.61x** in
float32 (best-to-best), and the `[M|y]` concat penalty shows up in the *full model*
at that size -- `R1_unroll_concat` 14257 us vs `R1_unroll_sep` 6226 us, a 2.3x
penalty in float64, which is R2b reproduced at a size the original study did not test.

---

## 3. What would falsify the latency-bound diagnosis

The diagnosis is **"launch-/latency-bound: neither FLOP-bound nor bandwidth-bound at any
production-relevant size."** It is falsified by *any* of:

| # | Falsifier | Why it would refute |
|---|---|---|
| **F1** | H100 float64 R1 `vmap` at (2,205,2) comes in at **≤ 305 µs** (≥3× the A6000) | A 21× FLOP/s increase would have delivered a 3× wall-clock gain only if arithmetic were a large share |
| **F2** | The no-likelihood floor drops **below 33 µs** (>2× better) | The fixed cost would then be device-side, not host-driven dispatch |
| **F3** | The H100 f32/f64 ratio for R1 `vmap` at (2,205,2) is **≥ 2.5×** (the A6000 measured 2.19× for that variant) | The H100's FP64 handicap is 16× smaller; a *larger* f32 gain can only mean FP64 arithmetic was binding |
| **F4** | The trsm RHS sweep at n=205, f64, shows **t(k=16)/t(k=8) ≥ 1.8** | Near-linear growth in arithmetic would mean throughput-bound, not fixed-cost-bound |
| **F5** | H100/A6000 speedup is **uniform across sizes** (same factor at (2,205,2) and (3,1024,8) to within 20%) | Latency-binding predicts the gain grows with problem size as arithmetic share grows |

**Confirmation looks like:** H100/A6000 ≤ 1.8× at (2,205,2) *and* ≥ 1.3× at (3,1024,8);
floor within ±40%; f32/f64 lower on H100 than on A6000; t(16)/t(8) < 1.8 in the RHS sweep.

---

## 4. The one-line answer, committed in advance

> **On a single H100 the CPU-beats-GPU verdict at the production point survives in
> float64 (GPU ~2.3× slower, down from ~3.4×) and becomes a tie in float32. It flips
> decisively — in either precision — only when you run ≥4–16 vectorized chains, or when
> the problem grows to `n_analyze ≈ 1024` with `n_modes ≥ 4`. The H100's 21× FP64
> throughput and 3× bandwidth buy almost nothing here, because a 205×205 triangular
> solve is a dependency chain, and no amount of silicon shortens a dependency chain.**
