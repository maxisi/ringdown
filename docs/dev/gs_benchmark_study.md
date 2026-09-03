# Gohberg-Semencul benchmark study: applying C^-1 by FFT versus by Cholesky (PR #141)

> **Status (2026-09-02): measured, not merged.** This note records the benchmark kit under
> `benchmarks/gs/` (nothing in the `ringdown` package was changed), its runs (run 1: the local RTX
> A6000 / Xeon Gold 6244 workstation with the pre-verification kit, archived and superseded; run 2:
> the same workstation with the corrected kit; the H100 run: rusty H100 PCIe node, Slurm job
> 6974520, corrected kit), the adversarial verification pass that corrected the kit between runs 1
> and 2, a corrections log (section 0.1), and the recommendation. The plan's "H100: pending"
> section is superseded by the H100 run.
> The mathematics is in `docs/gohberg_semencul_likelihood.md` (the "math note"); the one-shot
> likelihood the variants plug into is `docs/marginalized_likelihood.md` (development history in
> `docs/dev/model_optimization_study.md`). The math note carries the algebra; this document
> carries protocols, numbers, caveats and the decision. Every number below is copied from a table
> in `benchmarks/gs/results/local.md` (run 2, tag `local-a6000-run2`) or
> `benchmarks/gs/results/results_6974520.md` (tag `rusty-h100 job 6974520`), from
> `benchmarks/gs/inputs/prep_log.json`, or from the verification report, and names its leg,
> configuration and ACF family. Where the two machines disagree the text says so.

---

## 0. Purpose and verdict

**Purpose.** PR #141 proposes replacing the per-gradient triangular solves against the Cholesky
factor of the noise covariance, $W_i = L_i^{-1} M_i$, by an application of $C_i^{-1}$ through the
Gohberg-Semencul (GS) formula (two Yule-Walker filters and FFT convolutions), exact except for its
log-determinant shortcut (math note, sections 4, 8, 9). The note shows the route is $O(N\log N)$ per
right-hand side against $O(N^2)$ and leaves speed and accuracy at ringdown's production size to
measurement (its section 7).

**Verdict.** At the production point $(n_{\rm det}, N, n_{\rm mode}) = (2,205,2)$ the PR's kernel as
coded is *slower* than `main` on CPU (`gs_pr_ascoded@pow2` 573.8 us versus `main` 397.1 us per
gradient on `cpu_f64_prod` locally, 0.69x; 485.0 versus 319.0 us, 0.66x, on the H100 node's CPU) and
modestly faster on an H100 in float64 (328.9 versus 471.2 us, 1.43x). The best GS form, the Gram
variant `gs_half`, is 1.27-1.33x faster than `main` per gradient in the production CPU thread
configuration on both machines (299.2 versus 397.1 us locally; 250.2 versus 319.0 us on the H100
node) and 2.66x on the H100 in float64 (177.4 versus 471.2 us). Because the gradient is only 20-34%
of a NUTS leapfrog step at this size (`nuts` section, `share` column), the achievable end-to-end CPU
gain at production size is 5-8% (with `main`'s share; section 5). The FFT route wins decisively only
at $N \geq 2048$: 6.6-7.3x on CPU (`cpu_f64_prod`, (2,2048,4)) and 14-29x on the H100 ((2,2048,4) to
(2,4096,4)). In float64 every route is within the $\epsilon\,{\rm cond}(C)$ envelope; the GS
variants lose 5-50x more than Cholesky on the real GW150914 ACF at $(2, N \geq 1024)$ (2.6x at
(2,410,2), no gap at (3,1024,8)), entirely because of the forward error of the float64 Yule-Walker
coefficients, and a longdouble-refined filter removes the gap (2.4e-8 to 1.8e-11 at (2,205,2)
aligo02). **Recommendation: do not merge PR #141 as written; do not change the production CPU path
for speed; if a GS path is implemented for large $N$ or GPU use, implement `gs_half` with the exact
log-determinant, refined coefficients and a stored noise-model representation** (section 6).

### 0.1 Corrections log

Superseded claims (run 1, kit docs, math note, earlier drafts), as in `model_optimization_study.md`
0.3.

| # | Superseded claim (source) | Correction (where) |
|---|---|---|
| 1 | Run 1 headline: `gs_half` 13.9x faster than `main` at (2,1024,2) (`cpu_f64_omp1`, 13.88x). | A single-thread number. In production threads (`cpu_f64_prod`) it is 2.12x locally and 3.07x on the H100 node (3.5). |
| 2 | Run 1 kit: `--omp N` also exported `NPROC=N`, so no leg matched production (1 BLAS thread, full XLA pool). | Independent `--omp` / `--xla-threads`; `cpu_f64_prod` leg added; both CPU pathologies labelled by leg (2.3, 3.5). |
| 3 | Run 1 inputs: the hoisted variants' "exact" $\log\lvert C\rvert$ was a float64 Levinson sum, 2.7e-8 / 1.4e-5 / 1.6e-4 nats off at `gw150914` $N$ = 1024 / 2048 / 4096. | Longdouble Levinson sum stored by `prep_inputs.py --refresh-precompute`; reference log-det by blocked longdouble Cholesky (2.5, 4.3). |
| 4 | Run 1 gate: per-point normalization gave 1e-11 gate failures in 25 / 21 / 24 cells (`cpu_f64_omp8` / `omp1` / `gpu_f64`) and a `[fail]` on `main` at (2,4096,4) `gw150914` (9.7e-3). | Cloud normalization; `gate all pass: True` on every leg; `main` there is 2.4e-7 and the real result is the 51x GS digits-lost gap (2.5, 4.1). |
| 5 | Run 1 kit: the `gpu_f32` accuracy tables read as float32 route accuracy. | They measure TF32 matmul rounding (4.1e-2 to 4.6e-2 on `white`); the `gpu_f32_hi` leg is the source of every GPU float32 accuracy statement (2.6, 4.4). |
| 6 | Math note remark F1: "no accuracy gap was observed" between GS and Cholesky. | Holds for the analytic aLIGO ACF on the N(0,1) points; on the GW150914 ACF GS loses 37x / 51x more digits at (2,1024,2) / (2,4096,4), caused by `solve_toeplitz`'s coefficient error and removed by the longdouble filter (4.1, 4.2). |
| 7 | Math note section 11 (earlier text) and the kit's original docstrings: the PR's `fft_as`, `fft_bs` "are constant-folded at compile time". | They stay in the compiled gradient (4 extra FFT ops, 7-9% of `gs_pr`'s time on GPU) and are hoisted out of `while` loops on CPU only; the math note now says so (2.4, caveat 9). |
| 8 | Run 1 NUTS: one cold run, compile included, no `main_hoisted`. | Cold/warm split, two seeds, `main_hoisted` included; ESS/s still ranks nothing at one chain (2.7, 5). |
| 9 | Kit README, results section (written before run 2): gradient share "25-40%", `gs_half` "13.9x / ~2.2x". | Run 2: 20-34%, 12.67x / 2.12x; the README paragraph describing `results/local.json` as pre-fix predates run 2 (caveat 12). |
| 10 | Earlier drafts of this note: GS loses "5-50x more than Cholesky on the GW150914 ACF at $N \geq 1024$"; `cpu_f64_omp16` at (3,1024,8) "slower than single-threaded for every variant"; TF32 "inflates every cell 50-600x"; `gemm_linv` "most accurate at cond >= 1e10"; repeatability "3-9% on GPU". | $(2, N \geq 1024)$ only, none at (3,1024,8), 2.6x at (2,410,2); GEMM and GS variants only (`main` 23843.1 versus 25021.2 us); 20x to 2600x; `aligo02` at $N \geq 1024$ and (2,2048,4) `gw150914` only; per-leg maxima 0.9% to 43.4% (4.1, 3.5, 4.4, 2.4). |
| 11 | Plan: an "H100: pending" section. | Superseded by Slurm job 6974520; both machines are reported side by side throughout. |

---

## 1. Question and scope

**What PR #141 proposes.** Replace `solve_triangular(L, M)` and `solve_triangular(L, y)` by
`apply_cinv_gs_fast`, $C^{-1}\psi = \sigma^{-2}[L(a)L(a)^\top - L(\tilde a)L(\tilde a)^\top]\psi$
with four zero-padded real FFT convolutions per vector (math note, section 5.4), `vmap`ped over the
$k$ columns of $M$; pass `ar_coeffs` and `sigma` to the model instead of `L`; replace $\log|C|$ by
$2N\log\sigma_{N-1}$; drop `cholesky_factor` from `get_arviz`. The math note establishes that
everything but the log-determinant (D1) is exact, that the stored `Result` loses its noise model
(D3), and that the one-shot code needs $k+1$ right-hand sides per detector, not $k+2$ (section 11).

**What the kit measures**, for nine variants of the same one-shot model on six shapes, six ACF
families and seven legs per machine: device time per gradient, compile time and an HLO census
including what XLA hoists out of a `while` loop; float64 correctness against `main` and an
extended-precision reference, per gradient component, cloud-normalized, per point kind, with a
digits-lost-beyond-conditioning metric; float32 degradation; a short single-chain NUTS run at
production size. **Not measured:** NUTS throughput with enough chains to rank variants (section 5);
GPU NUTS; memory ($O(N^2)$ for $L$ or $L^{-1}$, $O(N)$ for the filters); A6000 float64 as a
production number (FP64 throttled). No variant changes the sampled posterior.

---

## 2. Method

### 2.1 Variants (`benchmarks/gs/variants.py`)

All share the design-matrix head of `ringdown.model.make_model` and the $k \times k$ tail (Cholesky
of $A^{-1}$, $u = R^{-1}v$; math note equation 6.2); they differ only in how $M_i^\top C_i^{-1}M_i$
is formed and what is precomputed.

| variant | how $M^\top C^{-1} M$ is formed | precomputed | role |
|---|---|---|---|
| `main` | `ringdown.model.make_model(marginalized=True)` literally: $W = L^{-1}M$, $z = L^{-1}y$ by `solve_triangular` every call | $L$ | baseline, as production runs |
| `main_hoisted` | same, with $z$, $Q = z^\top z$, $\log\lvert C\rvert$ passed in | $L$, $z$, $Q$, $\log\lvert C\rvert$ | baseline without $\theta$-independent work |
| `gemm_linv` | $W = L^{-1}M$ by one GEMM against a dense $L^{-1}$; $W^\top W$ | $L^{-1}$, $w = C^{-1}y$, $Q$, $\log\lvert C\rvert$ | decisive control: "avoid trsm" without FFTs |
| `gemm_cinv` | $C^{-1}M$ by GEMM against dense $C^{-1}$, symmetrized Gram | $C^{-1}$, $w$, $Q$, $\log\lvert C\rvert$ | numerical control |
| `gs_pr` | PR #141 kernel as coded (rfft of $a$, $\tilde a$; `flip` transposes; `vmap` over columns), spectra passed in, exact log-det | spectra, $\sigma^2$, $w$, $Q$, $\log\lvert C\rvert$ | the PR's algorithm |
| `gs_pr_ascoded` | as `gs_pr`, spectra computed inside the model body as the PR does | $a$, $\tilde a$, $\sigma^2$, $w$, $Q$, $\log\lvert C\rvert$ | the PR's model body |
| `gs_full` | batched GS giving $(C^{-1}M)^\top$ with conjugate-spectrum correlations, 4 batched FFT passes | spectra, $\sigma^2$, $w$, $Q$, $\log\lvert C\rvert$ | best full-$C^{-1}M$ form |
| `gs_half` | Gram form: $P = L(a)^\top M$, $R = L(\tilde a)^\top M$ by one rfft and one batched irfft; $A^{-1} \mathrel{+}= (P^\top P - R^\top R)/\sigma^2$ | spectra, $\sigma^2$, $w$, $Q$, $\log\lvert C\rvert$ | GS analogue of `main`'s $W^\top W$ |
| `floor` | prior only | none | kernel-launch and head floor |

Every hoisted variant uses the exact $\log|C| = \sum_m\log\sigma_m^2$ evaluated in longdouble; the
PR's $N\log\sigma^2$ is recorded (`logdetC_pr`) and never used in a model. GS variants take $a$,
$\tilde a$, $\sigma^2$ from `scipy.linalg.solve_toeplitz` as the PR does (`--gs-coeffs pr`, default)
or from a longdouble Levinson recursion rounded to float64 (`refined`); `correctness` always
evaluates the other policy through the same compiled function (`alt_coeffs`). FFT length: `--nfft
pow2` (the PR) or `fast` (`scipy.fft.next_fast_len(2N-1, real=True)`); they coincide for $N \geq
1024$ and differ at $N = 205$ (512 versus 432) and $N = 410$ (1024 versus 864). Census per gradient
(both machines, $n_{\rm det} = 2$): `main` 10 trsm (6 $N$-sized; 13 and 9 at $n_{\rm det} = 3$), 1
potrf; `main_hoisted` 8 trsm (4 $N$-sized); every GEMM and GS variant 4 $k \times k$ trsm, 1 potrf;
FFT ops at $n_{\rm det} = 2$: `gs_pr` 30, `gs_pr_ascoded` 34, `gs_full` 16, `gs_half` 8 (45, 51, 24,
12 at $n_{\rm det} = 3$).

### 2.2 Inputs (`prep_inputs.py`, `inputs/prep_log.json`)

**Grid.** $(2,205,2)$ (production), $(2,410,2)$, $(2,1024,2)$, $(2,2048,4)$, $(3,1024,8)$,
$(2,4096,4)$; $k = 4n_{\rm mode}$ = 8, 16 at the two $n_{\rm mode} = 4$ shapes, 32 at $(3,1024,8)$.
**ACF families**, ${\rm cond}(C)$ (max over detectors) at $N$ = 205 / 1024 / 4096: `white` 1.0;
`aligo20` (analytic aLIGO ACF, wall floored at 20 Hz) 5.6e2 throughout; `expcos` (the
`benchmarks/h100/bench.py` control) 7.3e2 / 7.8e2 / 7.8e2; `aligo2` (floored at 2 Hz) 3.8e6 / 7.9e6
/ 7.9e6; `gw150914` (H1/L1 ACFs from `etc/ringdown_fit_example.ini` via `Fit.compute_acfs`, strain
scaled by its std as `Fit.strain_scale` does) 1.8e6 / 5.4e8 / 1.5e12; `aligo02` (floored at 0.2 Hz)
5.7e9 / 2.8e10 / 8.2e10.

**Data convention.** `tests/test_model.py::_make_data` (unit-normalized ACF, injected ringdown, seed
42), with the injection rescaled so the network optimal SNR is 20 before the noise is added
(`snr_achieved` 20.0 in every file) and $a_{\rm scale,max} = 5\,a_{\rm true}$; without this the
unit-normalized aLIGO ACF gave SNR ~1e4 (`snr_recipe` 15943 at aligo02 (2,205,2), 15959 at (2,410,2)
and (2,1024,2)).

**Fixed points.** Per (shape, family): 20 N(0,1) draws in unconstrained coordinates (`pts_kind` 0,
seed 7) and 10 typical-set points (`pts_kind` 1): post-warmup samples of a 150 + 10 NUTS run of
`make_model(marginalized=True)` on that file's data, mapped back with `unconstrain_fn`; the warmup
child has a 300 s timeout, which the five non-gw150914 $N = 4096$ files and `aligo2` (3,1024,8) hit
(normal points only). With `m_max = 150` and injected 150 Hz content the warmup posteriors sit at
the prior edge (`m_mean` 145-149.5 for most files).

### 2.3 Legs and thread configurations

Each leg is a separate process with `JAX_PLATFORMS`, `jax_enable_x64`, the BLAS thread count
(`OMP/MKL/OPENBLAS_NUM_THREADS`, `--omp`) and the XLA:CPU Eigen pool size (`NPROC`, `--xla-threads`)
set before `import jax`; both thread knobs are recorded.

| leg | BLAS threads | XLA:CPU pool | sections | note |
|---|---|---|---|---|
| `cpu_f64_omp8` (local) / `cpu_f64_omp16` (H100 node) | 8 / 16 | 8 / 16 | env, reference, correctness, devtime, compile, f32acc, nuts | only CPU leg with $N = 4096$ |
| `cpu_f64_omp1` | 1 | 1 | env, correctness, devtime, compile | single core |
| `cpu_f64_prod` | 1 | unset = one thread per logical CPU (32 local, 16 H100 node) | env, devtime, compile | **production**: `import ringdown` sets only `OMP_NUM_THREADS=1` |
| `cpu_f32_omp8` / `cpu_f32_omp16` | 8 / 16 | 8 / 16 | env, f32acc, devtime, compile | |
| `gpu_f64` | 1 | n/a | env, correctness, f32acc, devtime, compile | A6000: FP64 at 1/32-1/64 rate, smoke only |
| `gpu_f32` | 1 | n/a | env, f32acc, devtime, compile | JAX default matmul precision: **TF32** dots |
| `gpu_f32_hi` | 1 | n/a | env, f32acc | `--matmul-precision highest`; timing unaffected (<3%) |

Hardware: `ccalin016`, 2 x Xeon Gold 6244 @ 3.60 GHz (`lscpu`: 2 sockets x 8 cores x 2 hardware
threads, so 16 physical cores and 32 logical CPUs; the "unset" 32-thread XLA pool runs two threads
per physical core), RTX A6000; `workergpu163`, Xeon Platinum 8362 @ 2.80 GHz (a 32-core part; the
Slurm allocation was 16 CPUs, so the "unset" 16-thread pool matched the allocation), H100 PCIe;
ringdown `v1.1.0-42-g0246591`, DUCC / cuFFT.

### 2.4 Timing (`harness.py`, from `benchmarks/h100/bench.py`, extended)

Device time per gradient by the **slope method**: the compiled gradient runs inside a `fori_loop` of
two lengths ($R_2 = 3R_1$), each timed 5 times with `block_until_ready`; the slope removes dispatch
and anything XLA hoisted out of the loop; a 1e-30 feedback keeps the loop-carried point
data-dependent on the gradient (no CSE). Constants are jit *arguments*, so the plain-gradient census
counts every op and the looped census reports what moved (`moved`). `us_spread` prints as `+-x%`;
identical executables timed twice (`@fast` = `@pow2` at $N \geq 1024$; GS variants only, 24-32 pairs
per leg) give the repeatability as a maximum (median) spread. Local: `cpu_f64_omp8` 35.1% (2.4%),
`cpu_f32_omp8` 25.1% (3.6%), `cpu_f64_prod` 3.7% (0.8%), `cpu_f64_omp1` 1.4% (0.5%), `gpu_f64` 2.3%
(0.2%), `gpu_f32` 43.4% (6.0%). H100 node: `cpu_f64_omp16` 5.4% (1.0%), `cpu_f32_omp16` 5.4% (0.9%),
`cpu_f64_prod` 3.4% (0.6%), `cpu_f64_omp1` 1.5% (0.2%), `gpu_f64` 0.9% (0.2%), `gpu_f32` 1.4%
(0.2%). The H100 numbers are resolved to ~1%, the production CPU legs to ~4%; speedup differences
below ~1.35x on the local multi-thread CPU leg and ~1.4x on the A6000 `gpu_f32` leg are not resolved
(`main` is not in these pairs; the verification pass's run-1 `cpu_f64_omp8` spot checks put its
spread at up to 28%).

**Hoisting.** XLA:CPU moves $\theta$-independent work out of `while` loops (the timing loop, NUTS
tree building): `main`'s two $N \times 1$ solves $z = L^{-1}y$ (`trsm 10->8, trsm_big 6->4`) and
`gs_pr_ascoded`'s four spectrum rffts (`fft 34->30`); nothing else has anything to hoist, and
XLA:GPU hoists nothing. So on CPU `main` ~= `main_hoisted`, and "speedup vs main" excludes the
recomputation production pays on GPU, where `main_hoisted` is 1.08-1.11x faster than `main` in
float64 and 1.11-1.16x in float32 (H100); "vs `main_hoisted`" is the route comparison on every
platform. XLA does *not* constant-fold ops on closed-over arrays in jax 0.11.1 (an rfft of an
embedded constant stays in the compiled gradient on both backends).

### 2.5 Correctness and accuracy protocol (`ref_longdouble.py`; sections `reference`, `correctness`)

**Reference.** $C^{-1}M$, $C^{-1}y$ by mixed-precision iterative refinement (float64 `cho_solve`,
longdouble residual, one correction; residual ~1e-21); $A^{-1}$, $R$, $u$, $\log\mathcal L$ in
longdouble; gradient by a hybrid VJP: the closed form $\partial\log\mathcal L/\partial M = C^{-1}(y
- M\alpha)\alpha^\top - C^{-1}MA$ in longdouble pulled back through the float64 `jax.vjp` of the
design-matrix head (agrees with longdouble central differences to 1.4e-11). Reference $\log|C|$:
blocked longdouble dense Cholesky; its difference from the longdouble Levinson sum (2e-11 nats at
cond 5e8, 2e-8 at 3.5e11) is the floor of the potential comparison.

**Metrics.** $U_{\rm lik} = U_{\rm var} - U_{\rm floor}$ in nats. Per site, $|g - g_{\rm ref}|$ over
the maximum of $|g_{\rm ref}[{\rm site}]|$ across **all** points of the cell ("cloud" normalization;
the per-point normalization is kept as a diagnostic because near the mode $|g_\chi|$ drops to 0.02
against ~70 over the cloud and turned 1e-13 roundoff into apparent 1e-11 gate failures in run 1).
`err_over_eps_cond` $= {\rm err}/(\epsilon_{64}\,{\rm cond}\,C)$, ~1 meaning as good as the
conditioning allows. Gate: 1e-11 versus `main` on `white`/`expcos`. Flags versus the reference: >
1e-8 concerning, > 1e-6 fail, never silenced. All per point kind.

### 2.6 float32 protocol (`f32acc`) and TF32

Float32 legs recompute at the same points and compare to the reference and to their float64 twin
(per point, cloud, per kind). Spectra are the float64 rfft cast to complex64; the alternative (rfft
of the float32-cast filter, what `gs_pr_ascoded` does) is evaluated as `alt_spectra` and differs at
the float32-eps level (~1e-7). JAX's default matmul precision runs float32 dots in **TF32** on
Ampere/Hopper, as production does, so the `gpu_f32` accuracy tables measure TF32 rounding (`white`,
cond 1: 4.1e-2 to 4.6e-2 for every variant on the H100 at (2,205,2) where `cpu_f32_omp16` gives
2.8e-4 to 6.6e-4); GPU route accuracy is read from `gpu_f32_hi` (`--matmul-precision highest`).

### 2.7 NUTS protocol (`nuts`)

CPU float64 multi-thread leg, (2,205,2) `aligo02`, 300 + 300, one chain, variants `main`,
`main_hoisted`, `gs_half`, `gemm_linv`. Per variant one cold run (seed 0, compile included) then one
warm run per seed in {0, 1}; us/leapfrog = warm wall / `num_steps`; ESS($m$) per seed; posterior
mean and sd of $m$, $\chi$; device us/grad of the same cell as a share of the step.

---

## 3. Performance results

Unless stated otherwise: `aligo02` (the `expcos` twin of every table agrees within the quoted
spread, with one exception: H100 `gpu_f64` `gemm_linv` at (2,205,2) is 133.3 us on `expcos` against
143.0 on `aligo02`, 7%), device microseconds per gradient, speedups relative to `main` of the same
leg.

### 3.1 Production size (2,205,2) on every leg, both machines

| leg | machine | `main` | `main_hoisted` | `gemm_linv` | `gs_pr@pow2` | `gs_full@fast` | `gs_half@fast` | `floor` |
|---|---|---|---|---|---|---|---|---|
| `cpu_f64_prod` | local | 397.1 | 394.1 | 347.6 (1.14x) | 568.0 (0.70x) | 444.5 (0.89x) | **299.2 (1.33x)** | 60.7 |
| `cpu_f64_prod` | H100 node | 319.0 | 307.3 | 264.8 (1.20x) | 485.2 (0.66x) | 384.6 (0.83x) | **250.2 (1.27x)** | 49.7 |
| `cpu_f64_omp1` | local | 390.1 | 405.7 | 325.7 (1.20x) | 599.8 (0.65x) | 455.8 (0.86x) | 304.5 (1.28x) | 63.0 |
| `cpu_f64_omp1` | H100 node | 326.3 | 359.2 | 265.4 (1.23x) | 499.3 (0.65x) | 386.4 (0.84x) | 250.4 (1.30x) | 51.2 |
| `cpu_f64_omp8` | local | 452.0 | 461.4 | 365.9 (1.24x) | 607.6 (0.74x) | 462.1 (0.98x) | 310.6 (1.46x) | 61.3 |
| `cpu_f64_omp16` | H100 node | 249.9 | 248.4 | 265.6 (0.94x) | 498.1 (0.50x) | 385.6 (0.65x) | 255.8 (0.98x) | 51.6 |
| `cpu_f32_omp8` | local | 318.6 | 326.7 | 174.3 (1.83x) | 403.3 (0.79x) | 302.0 (1.06x) | 200.5 (1.59x) | 26.2 |
| `cpu_f32_omp16` | H100 node | 223.1 | 227.4 | 162.2 (1.38x) | 373.6 (0.60x) | 269.7 (0.83x) | 175.8 (1.27x) | 23.6 |
| `gpu_f64` | H100 | 471.2 | 424.4 | **143.0 (3.29x)** | 303.1 (1.55x) | 239.6 (1.97x) | 189.1 (2.49x); `@pow2` 177.4 (2.66x) | 23.8 |
| `gpu_f64` (smoke) | A6000 | 1205.5 | 1121.7 | 291.9 (4.13x) | 544.9 (2.21x) | 573.1 (2.10x) | `@pow2` 342.1 (3.52x) | 61.7 |
| `gpu_f32` | H100 | 340.1 | 293.5 | **114.8 (2.96x)** | 307.5 (1.11x) | 225.9 (1.51x) | `@pow2` 174.0 (1.96x) | 21.0 |
| `gpu_f32` | A6000 | 300.9 | 258.9 | 98.9 (3.04x) | 524.3 (0.57x) | 328.6 (0.92x) | 159.4 (1.89x) | 24.8 |

On CPU in the production configuration only `gs_half` (1.27-1.33x) and `gemm_linv` (1.14-1.20x) beat
`main`; `gs_pr`, the PR's algorithm, is 0.66-0.70x; with 16 BLAS threads and a 16-thread pool
(`cpu_f64_omp16`) nothing beats `main` (249.9 us). On the H100 the dense-GEMM control is the fastest
variant in both precisions (143.0 and 114.8 us) and the GS forms are 1.2-1.5x slower. The `floor` is
50-63 us on CPU in float64 (23.6-26.2 in float32) and 21-24 us on the H100 (`gs_half` 7.5x the floor
there, `main` 20x).

### 3.2 Scaling with N

**`cpu_f64_prod` (production threads), local, `aligo02`:**

| config | `main` | `gemm_linv` | `gs_pr@fast` | `gs_full@fast` | `gs_half@fast` | `floor` |
|---|---|---|---|---|---|---|
| 2,205,2 | 397.1 | 347.6 (1.14x) | 509.0 (0.78x) | 444.5 (0.89x) | 299.2 (1.33x) | 60.7 |
| 2,410,2 | 1111.1 | 2136.1 (0.52x) | 1090.0 (1.02x) | 986.3 (1.13x) | 641.8 (1.73x) | 129.9 |
| 2,1024,2 | 10540.5 | 6203.1 (1.70x) | 2824.4 (3.73x) | 7523.4 (1.40x) | 4976.8 (2.12x) | 263.9 |
| 3,1024,8 | 29049.2 | 17766.4 (1.64x) | 19515.4 (1.49x) | 17648.5 (1.65x) | 14982.9 (1.94x) | 6134.8 |
| 2,2048,4 | 88273.1 | 22477.6 (3.93x) | 15251.1 (5.79x) | 14368.5 (6.14x) | 12136.0 (7.27x) | 1200.0 |

**`cpu_f64_prod`, H100 node (16-thread pool), `aligo02`:**

| config | `main` | `gemm_linv` | `gs_pr@fast` | `gs_full@fast` | `gs_half@fast` | `floor` |
|---|---|---|---|---|---|---|
| 2,205,2 | 319.0 | 264.8 (1.20x) | 440.3 (0.72x) | 384.6 (0.83x) | 250.2 (1.27x) | 49.7 |
| 2,410,2 | 881.0 | 763.2 (1.15x) | 943.9 (0.93x) | 853.4 (1.03x) | 542.6 (1.62x) | 97.3 |
| 2,1024,2 | 4566.3 | 1507.5 (3.03x) | 2601.7 (1.76x) | 2164.5 (2.11x) | 1485.8 (3.07x) | 203.8 |
| 3,1024,8 | 8642.2 | 4791.4 (1.80x) | 5497.2 (1.57x) | 5247.2 (1.65x) | 4441.3 (1.95x) | 1812.1 |
| 2,2048,4 | 22239.5 | 8784.7 (2.53x) | 4200.0 (5.30x) | 4190.8 (5.31x) | 3394.9 (6.55x) | 1154.2 |

The machines disagree above $N = 410$: `main` at (2,1024,2) is 10.5 ms locally and 4.6 ms on the
H100 node, and the batched-FFT variants are 3.3x slower locally (`gs_half` 4976.8 versus 1485.8 us)
while `gs_pr` is nearly the same (2824.4 versus 2601.7). Single-threaded (`cpu_f64_omp1`) the FFT
variants agree to ~10% (`gs_half` (2,1024,2) 1564.0 versus 1430.3) and `main` still differs 2x
(19814.4 versus 10266.5): part is trsm speed on the two CPUs, part the 32- versus 16-thread pool
(section 3.5). `gemm_linv` at (2,410,2) is the one cell where the sign differs: 0.52x locally, 1.15x
on the H100 node (0.97-1.01x single-threaded on either).

**`gpu_f64`, H100, `aligo02`** (spreads +-0-1% for every variant, +-2% for the `floor` at
(2,205,2)):

| config | `main` | `main_hoisted` | `gemm_linv` | `gs_pr@pow2` | `gs_full@pow2` | `gs_half@pow2` | `floor` |
|---|---|---|---|---|---|---|---|
| 2,205,2 | 471.2 | 424.4 | 143.0 (3.29x) | 303.1 (1.55x) | 216.6 (2.18x) | 177.4 (2.66x) | 23.8 |
| 2,410,2 | 790.3 | 715.6 | 140.5 (5.62x) | 323.0 (2.45x) | 231.5 (3.41x) | 188.9 (4.18x) | 29.9 |
| 2,1024,2 | 1774.4 | 1621.0 | 183.6 (9.66x) | 341.4 (5.20x) | 235.7 (7.53x) | 199.6 (8.89x) | 32.4 |
| 3,1024,8 | 2589.2 | 2357.6 | 293.7 (8.81x) | 490.6 (5.28x) | 353.1 (7.33x) | 274.5 (9.43x) | 47.7 |
| 2,2048,4 | 3294.3 | 2999.6 | 322.9 (10.20x) | 419.6 (7.85x) | 300.1 (10.98x) | 229.7 (14.34x) | 48.4 |
| 2,4096,4 | 8088.3 | 7467.5 | 786.1 (10.29x) | 571.6 (14.15x) | 384.6 (21.03x) | 276.0 (29.30x) | 33.6 |

The A6000 `gpu_f64` smoke leg shows the same ordering (`main` 1205.5 to 18864.0 us across the grid,
`gs_half@pow2` 342.1 to 1059.8 us); its 3.5-18x speedups mostly measure a slow `dtrsm`.

**`gpu_f32`, `aligo02`, both GPUs** (TF32 dots; timing unaffected by matmul precision):

| config | H100 `main` | H100 `gemm_linv` | H100 `gs_pr@pow2` | H100 `gs_half@pow2` | A6000 `main` | A6000 `gemm_linv` | A6000 `gs_half@pow2` |
|---|---|---|---|---|---|---|---|
| 2,205,2 | 340.1 | 114.8 (2.96x) | 307.5 (1.11x) | 174.0 (1.96x) | 300.9 | 98.9 (3.04x) | 220.2 (1.37x) |
| 2,410,2 | 494.0 | 118.4 (4.17x) | 323.6 (1.53x) | 177.8 (2.78x) | 443.5 | 109.8 (4.04x) | 146.3 (3.03x) |
| 2,1024,2 | 1009.0 | 128.6 (7.85x) | 328.5 (3.07x) | 185.3 (5.44x) | 1040.4 | 152.0 (6.84x) | 163.0 (6.38x) |
| 3,1024,8 | 1848.9 | 183.9 (10.05x) | 469.6 (3.94x) | 243.2 (7.60x) | 1795.6 | 203.1 (8.84x) | 240.7 (7.46x) |
| 2,2048,4 | 2567.2 | 224.2 (11.45x) | 350.0 (7.34x) | 190.1 (13.51x) | 2421.5 | 311.6 (7.77x) | 188.3 (12.86x) |
| 2,4096,4 | 5696.0 | 445.7 (12.78x) | 400.3 (14.23x) | 209.1 (27.24x) | 5100.5 | 909.5 (5.61x) | 213.7 (23.87x) |

The A6000 `gpu_f32` cells carry spreads up to 21% (`gs_half@pow2` (2,410,2)) and duplicate
executables disagree by up to 43.4%, so its ordering at $N \leq 1024$ is not resolved; the H100
cells are +-0-1%. Scaling exponents (verification pass, `cpu_f64_omp1`, $N = 205 \to 1024$, $k =
8$): `main` $N^{2.46-2.52}$, `gemm_linv` $N^{1.85-1.90}$, GS variants $N^{0.95-0.99}$.

### 3.3 "Avoid trsm" versus "use FFT": `gemm_linv` against `gs_half`

| config | H100 `gpu_f64` gemm / gs_half | H100 `gpu_f32` | H100 node `cpu_f64_prod` | local `cpu_f64_prod` |
|---|---|---|---|---|
| 2,205,2 | 143.0 / 177.4 (GEMM 1.24x better) | 114.8 / 174.0 (GEMM 1.52x) | 264.8 / 250.2 (tie) | 347.6 / 299.2 (gs_half 1.16x) |
| 2,410,2 | 140.5 / 188.9 (GEMM 1.34x) | 118.4 / 177.8 (GEMM 1.50x) | 763.2 / 542.6 (gs_half 1.41x) | 2136.1 / 641.8 (gs_half 3.33x) |
| 2,1024,2 | 183.6 / 199.6 (GEMM 1.09x) | 128.6 / 185.3 (GEMM 1.44x) | 1507.5 / 1485.8 (tie) | 6203.1 / 4976.8 (gs_half 1.25x) |
| 2,2048,4 | 322.9 / 229.7 (gs_half 1.41x) | 224.2 / 190.1 (gs_half 1.18x) | 8784.7 / 3394.9 (gs_half 2.59x) | 22477.6 / 12136.0 (gs_half 1.85x) |
| 2,4096,4 | 786.1 / 276.0 (gs_half 2.85x) | 445.7 / 209.1 (gs_half 2.13x) | not run | not run |

On GPU at $N \leq 1024$ the whole gain is "avoid trsm": the GEMM control is at least as fast as the
best GS form. The FFT-specific gain appears at $N \geq 2048$ ($k = 16$) and grows: 1.4x over GEMM at
2048 and 2.9x at 4096 on the H100 in float64, 1.9-2.6x on CPU at 2048. On CPU with production
threads the two tie at $N$ = 205 and 1024 on the H100 node (`gs_half` 1.16-1.25x locally); (2,410,2)
is the one small-$N$ CPU cell where the FFT route beats GEMM outright on both machines, 1.41x on the
H100 node and 3.33x locally (where `gemm_linv` is slower than `main`, section 3.2).

### 3.4 The PR kernel as coded versus `main`

`gs_pr_ascoded@pow2` is the PR's model body (power-of-two padding, spectra computed in the body,
`vmap` over columns) with the exact log-determinant substituted.

| leg, (2,205,2) unless noted | `gs_pr_ascoded@pow2` | `main` | speedup | vs `main_hoisted` |
|---|---|---|---|---|
| local `cpu_f64_prod` | 573.8 | 397.1 | 0.69x | 0.69x |
| H100 node `cpu_f64_prod` | 485.0 | 319.0 | 0.66x | 0.63x |
| H100 `gpu_f64` | 328.9 | 471.2 | 1.43x | 1.29x |
| H100 `gpu_f32` | 329.3 | 340.1 | 1.03x | 0.89x |
| H100 `gpu_f64`, (2,4096,4) | 613.9 | 8088.3 | 13.18x | 12.16x |

The in-body spectra cost 8.5% on the H100 in float64 at (2,205,2) (328.9 versus `gs_pr@pow2` 303.1)
and 7% at (2,4096,4) (613.9 versus 571.6) because XLA:GPU does not hoist them; on CPU the two are
equal within noise. With 30 FFT ops per gradient against `gs_half`'s 8, the PR's form is the slowest
GS variant: `gs_pr_ascoded@pow2` is 1.80-2.22x slower than `gs_half@pow2` on the H100 in float64
across the six shapes (328.9 versus 177.4 us at (2,205,2), 613.9 versus 276.0 at (2,4096,4);
`gs_pr@pow2` 1.71-2.07x), and `gs_pr@fast` is 1.6-1.8x slower than `gs_half@fast` single-threaded on
CPU on both machines; the one exception is the local pool pathology at (2,1024,2), where `gs_pr` is
1.8x *faster* than the batched `gs_half` (2824.4 versus 4976.8 us, `cpu_f64_prod`).

### 3.5 Thread configuration on CPU

`gs_half@fast` at (2,1024,2) `aligo02` and its speedup over `main` in the same leg:

| leg | local `gs_half` | local `main` | speedup | H100 node `gs_half` | H100 node `main` | speedup |
|---|---|---|---|---|---|---|
| `cpu_f64_omp1` | 1564.0 | 19814.4 | 12.67x | 1430.3 | 10266.5 | 7.18x |
| `cpu_f64_omp8` / `omp16` | 4793.7 | 11376.5 | 2.37x | 1425.3 | 3997.0 | 2.80x |
| `cpu_f64_prod` | 4976.8 | 10540.5 | 2.12x | 1485.8 | 4566.3 | 3.07x |

The run-1 headline "13.9x at $N = 1024$" was a `cpu_f64_omp1` number; in production it is 2.1x
locally and 3.1x on the H100 node. Two pathologies, one per machine:

* **Local 32-thread XLA pool at $N \geq 1024$.** The batched-FFT variants and the prior-only
  `floor` run 2.8-3.3x slower with the 8- or 32-thread pool than with one thread (`floor`
  (3,1024,8): 1914.7 us `omp1`, 6351.7 `omp8`, 6134.8 `prod`; `gs_full` (2,1024,2): 2546.1 /
  7142.4 / 7523.4) while `main`'s trsm gets faster; the verification pass isolated it to `NPROC`
  (1.5 cores busy for `gs_full` at 8 threads). It does **not** reproduce on the H100 node
  (`floor` (3,1024,8): 1882.9 / 1847.8 / 1812.1; `gs_half` (2,1024,2) 1430.3 / 1425.3 / 1485.8).
  The obvious, unmeasured candidate for the machine difference is hyperthreading: the local
  "unset" pool is 32 threads on 16 physical cores (section 2.3), the node's 16 on 16. It cannot be
  the whole story, since the local 8-thread leg shows the same slowdown with half the cores idle;
  the two-socket layout of the local box is the other unmeasured difference.
* **16 BLAS threads at $k = 32$ on the H100 node.** `cpu_f64_omp16` at (3,1024,8) is slower than
  single-threaded for every GEMM and GS variant: `gemm_linv` 33004.6 us (`omp1` 19216.1, `prod`
  4791.4), `gs_pr` 40111.6 (16725.7, 5497.2), `gs_half` 19993.5 (10650.5, 4441.3); `main` is
  slightly faster with 16 threads (23843.1 versus 25021.2; `prod` 8642.2) and `floor` is unchanged
  (1847.8 versus 1882.9). One BLAS thread with a 16-thread pool (`prod`) is fastest for every
  variant there.

Any CPU speedup must be quoted with its thread setting; the production one is `cpu_f64_prod`.

### 3.6 FFT length, and compile time

At (2,205,2), `fast` (432) beats `pow2` (512) on DUCC by 1.08-1.12x (`gs_pr` 509.0 versus 568.0 and
`gs_half` 299.2 versus 329.9 us on local `cpu_f64_prod`; `gs_half` 250.2 versus 271.0 on the H100
node) and loses on cuFFT (H100 `gpu_f64`: `gs_pr` 344.0 versus 303.1, `gs_half` 189.1 versus 177.4;
A6000 `gpu_f64`: `gs_pr` 858.1 versus 544.9, `gs_full` 573.1 versus 405.3). A production
implementation would pick per backend or accept ~10% on one.

`jit(grad)` lower + compile is 0.18-0.75 s on CPU (the 0.75 is local `cpu_f32_omp8` `gs_full@pow2`
at (2,205,2); the next largest is 0.65) and 0.31-1.24 s on GPU for every variant and shape; GS adds
0.05-0.15 s (local `cpu_f64_prod` (2,205,2): `main` 0.37, `gs_half` 0.40, `gs_pr` 0.43-0.47; H100
`gpu_f64`: 0.44, 0.51-0.54, 0.56-0.58). The NUTS-level compile (cold minus warm, `nuts` section) is
6.2 s for `main` versus 1.8 s `gs_half` and 0.0 s `gemm_linv` locally (2.9 / 0.8 / 0.0 s on the H100
node).

---

## 4. Numerical results

### 4.1 float64 gradients against the longdouble reference

Cloud-normalized maximum relative gradient error versus the reference. The four GS variants and both
paddings agree to the printed digits, as do `gemm_linv` and `gemm_cinv`; the values are identical on
every float64 leg of both machines (`main_hoisted` differs from `main` by 2.2e-13 at (2,205,2)
`aligo02` on the H100 node's CPU; the maximum over all cells is at (2,4096,4) `gw150914`: 2.9e-12
`cpu_f64_omp16`, 3.2e-12 local `cpu_f64_omp8`, 6.1e-12 H100 `gpu_f64`; 1.3e-12 at (2,2048,4)).
Flags: `[concerning]` > 1e-8, `[fail]` > 1e-6.

| config | family | ${\rm cond}(C)$ | `main` | `gemm_linv` | GS variants | GS, refined filter (`alt_coeffs`) |
|---|---|---|---|---|---|---|
| 2,205,2 | aligo02 | 5.7e9 | 1.5e-8 | 1.4e-8 | 2.4e-8 | 1.8e-11 |
| 2,410,2 | aligo02 | 1.1e10 | 1.2e-8 | 1.6e-8 | 2.7e-8 | 2.2e-11 |
| 2,1024,2 | aligo02 | 2.8e10 | 2.3e-8 | 7.0e-9 | 1.6e-8 | 1.5e-11 |
| 3,1024,8 | aligo02 | 2.8e10 | 2.1e-8 | 6.5e-9 | 2.1e-8 | 2.0e-11 |
| 2,2048,4 | aligo02 | 5.6e10 | 6.6e-8 | 1.1e-8 | 2.6e-8 | 6.2e-11 |
| 2,4096,4 | aligo02 | 8.2e10 | 3.1e-8 | 8.1e-9 | 1.6e-8 | 1.1e-11 |
| 2,205,2 | gw150914 | 1.8e6 | 1.0e-11 | 1.2e-11 | 6.3e-12 | 7.0e-14 |
| 2,410,2 | gw150914 | 3.7e6 | 1.3e-11 | 1.1e-11 | 3.2e-11 | 8.8e-14 |
| 2,1024,2 | gw150914 | 5.4e8 | 4.9e-11 | 3.6e-11 | 1.8e-9 | 4.1e-13 |
| 3,1024,8 | gw150914 | 5.4e8 | 1.4e-11 | 1.2e-11 | 1.7e-11 | 1.2e-11 |
| 2,2048,4 | gw150914 | 3.5e11 | 9.8e-9 | 2.3e-9 | 4.9e-8 | 8.8e-11 |
| 2,4096,4 | gw150914 | 1.5e12 | 2.4e-7 | 6.4e-7 | 1.2e-5 `[fail]` | 1.4e-8 |
| 2,205,2 | aligo20 | 5.6e2 | 9.2e-14 | 6.4e-14 | 7.2e-14 to 8.5e-14 | 6.5e-14 to 7.4e-14 |
| 2,205,2 | expcos | 7.3e2 | 2.1e-12 | 2.1e-12 | 2.1e-12 | 2.1e-12 |
| 2,205,2 | white | 1.0 | 4.4e-14 | 3.6e-14 | 3.3e-14 to 4.4e-14 | unchanged |

The algebra gate passes for every variant, shape and leg (`gate all pass: True`); the worst
variant-versus-`main` cell on the gate families is 1.7e-13 on `cpu_f64_omp16` (`gs_full`, (3,1024,8)
`white`), 2.0e-13 on local `cpu_f64_omp8` (`gemm_linv`, (3,1024,8) `expcos`) and 2.3e-13 over all
legs (local `gpu_f64`, `gs_pr`, (2,2048,4) `white`). The shared residual of every route against the
reference on those families (6.9e-11 at (3,1024,8) `expcos`, 3.7e-11 on `white`) is $k = 32$ tail
roundoff common to `main`. On `aligo02` at ${\rm cond} \leq 8.2e10$ every route including `main`
sits at 1e-8 to 7e-8, the `[concerning]` flag fires for all, and `gemm_linv` is the most accurate on
`aligo02` at $N \geq 1024$ (7.0e-9 / 6.5e-9 / 1.1e-8 / 8.1e-9 against `main`'s 2.3e-8 / 2.1e-8 /
6.6e-8 / 3.1e-8) and at (2,2048,4) `gw150914` (2.3e-9 versus 9.8e-9), but not at (2,410,2) `aligo02`
(1.6e-8 versus 1.2e-8) or (2,4096,4) `gw150914` (6.4e-7 versus 2.4e-7).

**Digits lost beyond conditioning**, ${\rm err}/(\epsilon_{64}\,{\rm cond}\,C)$ (both machines):

| config | family | `main` | `gemm_linv` | GS |
|---|---|---|---|---|
| 2,205,2 | aligo02 | 0.012 | 0.011 | 0.019 |
| 2,1024,2 | aligo02 | 0.0037 | 0.0011 | 0.0026 |
| 2,4096,4 | aligo02 | 0.0017 | 0.00044 | 0.00086 |
| 2,205,2 | gw150914 | 0.025 | 0.029 | 0.016 |
| 2,410,2 | gw150914 | 0.015 | 0.013 | 0.039 |
| 2,1024,2 | gw150914 | 0.00041 | 0.0003 | 0.015 |
| 3,1024,8 | gw150914 | 0.00011 | 0.0001 | 0.00014 |
| 2,2048,4 | gw150914 | 0.00013 | 2.9e-5 | 0.00063 |
| 2,4096,4 | gw150914 | 0.00074 | 0.002 | 0.038 |
| 2,205,2 | white | 2e2 | 1.6e2 | 1.5e2 to 2e2 |

Every value on the conditioned families is far below 1: no route loses digits the problem does not
force (the `white` entry divides summation roundoff by a cond of 1 and carries no information;
`expcos` gives 13 for every route). On `aligo02` GS is within a factor 2.3 of `main` at $N \leq 410$
(0.011 versus 0.0048 at (2,410,2)) and better than it at $N \geq 1024$. The GS-versus-Cholesky gap
is a `gw150914` effect: 2.6x already at (2,410,2) (0.039 versus 0.015), then 37x at (2,1024,2)
(0.015 versus 0.00041), 4.8x at (2,2048,4), 51x at (2,4096,4) (0.038 versus 0.00074); it is absent
at (2,205,2) (0.016 versus 0.025) and at (3,1024,8) with the same ACFs (0.00014 versus 0.00011).

### 4.2 Attribution: the Yule-Walker coefficients (F1), not GS cancellation (F2) or FFT roundoff

The verification pass reproduced the (2,1024,2) `gw150914` GS error exactly (1.44e-8 per point for
dense GS, FFT `gs_pr`, FFT `gs_full` and the `gs_half` Gram, against 1.7e-10 for `cho_solve`) and
isolated its cause: `solve_toeplitz`'s $a$ has relative error 4.1e-9 (detector 1, cond 5.4e8) and
6.8e-11 (detector 0) against a longdouble Levinson solve, flat across lags, 3.4e-6 at $N = 4096$
(cond 1.5e12); the same FFT kernels with a longdouble-accurate $a$ and $\sigma^2$ cast to float64
give 1.4e-12, better than Cholesky (refining $a$ alone reaches only 1.0e-8: both must be refreshed);
dense GS in **longdouble arithmetic** with the float64 $a$ still gives 1.44e-8, so the difference of
two positive terms (F2) contributes nothing, the cancellation ratio
$\|L(a)L(a)^\top\psi\|/\|\sigma^2 C^{-1}\psi\|$ being 1.003 for the vector form and 1.000 for the
Gram form on the real design matrices (1.00-2.07 across all ACFs); and FFT roundoff is nil (dense GS
equals FFT GS to all printed digits, `pow2` equals `fast`).

The `alt_coeffs` column of the 4.1 table confirms this in situ on both machines: the refined filter
(`a_ld`, `atilde_ld`, `sigma2_ld` from `ref_longdouble.levinson_ld`) takes every GS cell to 1e-11 or
better except (2,4096,4) `gw150914`, where 1.2e-5 becomes 1.4e-8 (17x better than `main`'s 2.4e-7).
The math note's remark F1 ("no accuracy gap observed") holds for the analytic aLIGO ACF on the
N(0,1) points, where GS with the PR's coefficients beats `cho_solve` (by kind, `cpu_f64_omp16`
(2,205,2) `aligo02`: normal points GS 7.2e-9 versus `main` 1.5e-8; typical-set points 2.4e-8 versus
1.3e-8, which is what the 4.1 table's maximum, 2.4e-8 versus 1.5e-8, reports), and fails for the
measured GW150914 ACF. The longdouble recursion must run anyway for the exact log-determinant, so
the refined coefficients are free. `alt_coeffs` reports gradients only: whether the refined filter
also removes the GS route's potential error (8.5e-4 nats at (2,4096,4) `gw150914` against `main`'s
3.3e-6, section 4.3) is unmeasured. Symmetry (F3): `gs_full` symmetrizes explicitly; `gs_half`'s
difference of Grams is exactly symmetric like `main`'s $W^\top W$; the pre-symmetrization asymmetry
is not a kit output (section 7).

### 4.3 The likelihood value and the log-determinant

$|U_{\rm lik} - U_{\rm ref}|$ in nats (`cpu_f64_omp16`; same to printed digits locally): (2,205,2)
`aligo02` `main` 4.3e-7, `main_hoisted` 3.1e-7, `gemm_linv` 4.5e-8, GS 4.4e-7; (2,1024,2) `gw150914`
6.9e-9 / 1.5e-9 / 1.1e-9 / 5.6e-8; (2,4096,4) `gw150914` 3.3e-6 / 2.3e-5 / 2.7e-5 / 8.5e-4; `white`
(2,205,2) 4.5e-13 to 8.0e-13. At ${\rm cond} \geq 10^{11}$ the nats columns partly measure the
reference's own floor (2e-8 nats at 3.5e11). That floor does not explain the spread among variants
sharing `main`'s $L$: at (2,4096,4) `gw150914` `main_hoisted` 2.3e-5 and `gemm_linv` 2.7e-5 against
`main` 3.3e-6 (2.4e-7 / 2.3e-8 against 6.4e-8 at (2,2048,4)). This is unexplained; the likely site
is the precomputed $Q = y^\top C^{-1}y$ (one float64 `cho_solve` at cond 1e12, where `main` forms
$z^\top z$ itself), but the kit does not isolate it. Run 1 carried a $\theta$-independent bias in
the hoisted variants from a float64 Levinson log-determinant, the least accurate route (2.7e-8 nats
off at `gw150914` $N = 1024$, 1.4e-5 at 2048, 1.6e-4 at 4096, against 9e-10 / 2e-7 / 2e-6 for
ringdown's $2\sum\log L_{tt}$): the exact formula must be evaluated in extended precision, or
replaced by the Cholesky diagonal, at ${\rm cond} \gtrsim 10^8$.

**The PR's shortcut** $2N\log\sigma_{N-1}$ (D1), `logdet_exact_minus_pr`, nats per detector: `white`
0; `expcos` 2.964; `aligo20` 3.817 / 3.808; `aligo2` 12.856 / 12.735; `aligo02` ($N = 205$) 21.804 /
20.945, ($N = 4096$) 22.509 / 21.791; `gw150914` ($N = 205$) 67.404 / 101.745, ($N = 1024$) 142.376
/ 210.951, ($N = 2048$) 233.725 / 330.362, ($N = 4096$) 29684.351 / 24451.469. $\theta$-independent,
so harmless for sampling; wrong for any absolute log-likelihood, evidence or `loo` use, and off by
tens of thousands of nats on the real ACF at large $N$.

### 4.4 float32

Cloud-normalized maximum relative gradient error versus the reference, threshold mark 1e-3. CPU
(`cpu_f32_omp16`, H100 node; `cpu_f32_omp8` locally agrees within a factor 1.53 in every quoted
cell, the 1.53 being `gemm_linv` (2,1024,2) `aligo02`, 5.2e-5 versus 3.4e-5):

| config | family | ${\rm cond}(C)$ | `main` | `gemm_linv` | `gs_pr` | `gs_half` |
|---|---|---|---|---|---|---|
| 2,205,2 | white | 1.0 | 3.0e-5 | 1.6e-5 | 1.6e-5 | 1.4e-5 |
| 2,205,2 | aligo02 | 5.7e9 | 1.5e-4 | 6.5e-5 | 6.7e-5 | 7.4e-5 |
| 2,1024,2 | aligo02 | 2.8e10 | 2.4e-4 | 3.4e-5 | 4.0e-5 | 2.8e-5 |
| 2,1024,2 | gw150914 | 5.4e8 | 1.1e-4 | 4.6e-5 | 6.0e-5 | 5.7e-5 |
| 2,2048,4 | aligo02 | 5.6e10 | 2.8e-3 | 3.5e-4 | 2.6e-4 | 2.9e-4 |
| 2,2048,4 | gw150914 | 3.5e11 | 2.0e-3 | 9.5e-4 | 1.0e-3 | 1.0e-3 |
| 3,1024,8 | white | 1.0 | 2.5e-2 | 2.5e-2 | 2.5e-2 | 2.5e-2 |

Per point (what a single NUTS step sees) the same leg gives 1.1e-3 to 1.6e-3 on `white` (2,1024,2),
1.2e-1 on `white` (2,2048,4) (`gs_half` 9.6e-2) and 3.1e-1 on `white` (3,1024,8) for every variant;
the cloud-normalized by-kind table splits the last into normal points 4.5e-6 and typical-set points
2.5e-2 (the per-point split, 3.0e-5 / 3.1e-1, is the verification pass's figure quoted in the kit
README). The $k = 32$ typical-set cells are a $k \times k$-tail cancellation near the mode
($\epsilon_{32}\,{\rm cond}(A^{-1})\,S/|g| \approx 0.13$) that every route shares and that has
nothing to do with ${\rm cond}(C)$.

**GPU (H100, `aligo02` (2,205,2) / (2,1024,2), per point).** Default TF32 dots (`gpu_f32`): `main`
6.0e-2 / 2.4e-1, `gemm_linv` 7.9e-2 / 2.4e-1, `gs_pr` 2.7e-1 / 3.7e-1, `gs_half` 9.4e-2 / 2.2e-1;
`white` (2,205,2) 4.1e-2 to 4.6e-2 for every variant. With `--matmul-precision highest`
(`gpu_f32_hi`): `main` 1.4e-3 / 1.3e-2, `gemm_linv` 7.2e-4 / 1.0e-2, `gs_pr` 4.2e-4 / 1.4e-4,
`gs_half` 3.9e-4 / 2.1e-4; `white` 8.6e-4 to 1.2e-3. The A6000 reproduces this (`gpu_f32_hi`
(2,1024,2): `main` 1.5e-2, `gemm_linv` 5.2e-4, `gs_pr` 9.0e-5, `gs_half` 1.1e-4). TF32 inflates the
quoted cells by roughly 20x to 2600x (`main` 43x / 18x, `gemm_linv` 110x / 24x, `gs_pr` 640x /
2600x, `gs_half` 240x / 1050x at (2,205,2) / (2,1024,2)), one to three orders of magnitude, and
reorders the variants; only the `highest` tables speak to the $C^{-1}$ route, and they agree with
CPU (cloud, (2,1024,2) `aligo02`: `main` 3.1e-4 H100 / 3.5e-4 A6000 against 3.1e-5 to 3.5e-5 for the
GS variants).

**Conclusion.** In float32 no route is usable at realistic conditioning: at production size on
`aligo02` every variant is 4e-4 to 1.4e-3 per point at the typical-set points even with true float32
dots, and at $k \geq 16$ the typical-set points reach 1e-1 for every variant on white noise (the
picture of `model_optimization_study.md` section 5.3). Within that, the routes that precompute the
whitening (`gemm_linv`, GS) degrade 2-10x less with ${\rm cond}(C)$ than `main` (2.8e-3 versus
2.6e-4 to 3.5e-4 at (2,2048,4) `aligo02`, cloud), plausibly because `main`'s float32 triangular
solve carries ${\rm cond}(L) = \sqrt{{\rm cond}\,C}$ (an interpretation, not something the kit
isolates). Between `gemm_linv` and the GS forms float32 is a wash.

---

## 5. End-to-end NUTS check

`nuts` section, (2,205,2) `aligo02`, one chain, 300 + 300, warm seeds 0 and 1 (`steps` and `ESS` are
identical on the two machines: same seeds, same chains).

| variant | H100 node us/leapfrog (warm) | us/grad (share) | local us/leapfrog (warm) | us/grad (share) | steps (seed 0 / 1) | ESS($m$) | $m$, $\chi$ (seed 0) |
|---|---|---|---|---|---|---|---|
| `main` | 992 [951..1034] | 250 (25%) | 1324 [1269..1380] | 452 (34%) | 3258 / 2794 | 181 / 111 | 148.19 +- 1.59, 0.9241 |
| `main_hoisted` | 1172 [970..1373] | 248 (21%) | 1575 [1298..1851] | 461 (29%) | 1968 / 3112 | 186 / 87 | 148.35 +- 1.58, 0.9252 |
| `gemm_linv` | 1078 [980..1176] | 266 (25%) | 1239 [1135..1343] | 366 (30%) | 3054 / 3558 | 88 / 114 | 147.94 +- 2.14, 0.9239 |
| `gs_half` | 1342 [1263..1421] | 270 (20%) | 1652 [1577..1726] | 337 (20%) | 3192 / 2574 | 150 / 140 | 148.37 +- 1.56, 0.9261 |

What it shows: posterior means agree within Monte Carlo error (MCSE($m$) 0.12-0.23, all $z \leq
1.8$); the compiled gradient is 20-34% of a leapfrog step at this size; the seed-to-seed spread of
ESS (`main` 111-181) exceeds every variant-to-variant difference, so the ESS/s column (`main` 58.3 /
38.3, `gs_half` 37.2 / 38.2 on the H100 node) ranks nothing at one chain.

One open anomaly: `gs_half`'s warm per-leapfrog wall is 25-35% *higher* than `main`'s on both
machines (1342 versus 992 us; 1652 versus 1324) although its device us/grad is equal or lower (270
versus 250; 337 versus 452), and the ranges do not overlap. The kit does not resolve why
(candidates: the NUTS `while` body with closed-over rather than argument constants, as production
numpyro has it; the potential-only evaluations of tree building). Until this is understood no
end-to-end claim for the GS route can be made, at any $N$.

**Implication.** With the gradient at a share $s = 0.20$-$0.34$ of the step, a per-gradient speedup
$f$ changes the step by $1 - s(1 - 1/f)$: `gs_half`'s 1.27-1.33x in the production CPU configuration
is worth 5.3-8.4% with `main`'s share (25% on the H100 node, 34% locally; this is the "5-8%" quoted
throughout) or 4.3-5.0% with `gs_half`'s own measured share (20% on both), and an infinitely fast
gradient at most 20-34%. The rest is the sampler (tree building, mass-matrix products, control
flow), which no $C^{-1}$ route touches.

---

## 6. Decision

**(i) Do nothing.** Leaves 1.27-1.33x per gradient (5-8% end-to-end) on the table at production size
on CPU, and 2.4-3.3x per gradient on an H100 in float64, where the gradient share is unmeasured.
Keeps the LVK-reviewed path untouched; consistent with `model_optimization_study.md` section 4.4 (no
prewhitening for a 0% end-to-end gain).

**(ii) Cheapest win: precompute $L^{-1}$ and GEMM (`gemm_linv`).** Exact by construction (same $L$;
gate 1.3e-14 to 4.9e-14 at (2,205,2); the most accurate route on `aligo02` at $N \geq 1024$ and at
(2,2048,4) `gw150914`, though not at (2,410,2) `aligo02` or (2,4096,4) `gw150914`, section 4.1), no
new numerics, no FFT length, no filter. Per gradient 1.14-1.20x on CPU production threads at
(2,205,2), 3.29x on the H100 in float64 and 2.96x in float32 (the best variant there), 1.70-3.03x on
CPU at $N = 1024$. But it is not robust at intermediate $N$ on CPU (0.52x at (2,410,2) locally,
1.15x on the H100 node) and trails the FFT route by 1.9-2.6x at $N \geq 2048$ on CPU and 1.4-2.9x on
the H100. Cost: small (pass $L^{-1}$ or store both; `Result` keeps whitening from $L$), but it
changes the model signature, which the earlier study declined to do.

**(iii) Implement `gs_half` with the exact log-determinant and refined Yule-Walker coefficients,
storing a noise-model representation for `Result`.** The fastest route in every float64 CPU leg and
shape in production threads and in every leg at $N \geq 2048$. Exceptions: `cpu_f64_omp16` (2,205,2)
(0.98x); both GPUs at $N \leq 1024$ except (3,1024,8) in H100 float64, where `gemm_linv` leads by
1.1-1.5x (H100) and up to 1.6x (A6000 `gpu_f32`); and the float32 CPU legs at (2,205,2) on both
machines (174.3 versus 200.5 us locally, 162.2 versus 175.8 on the H100 node), (2,410,2) on the H100
node (326.7 versus 346.8) and (3,1024,8) on both (8457.7 versus 12454.6; 10945.5 versus 12105.1).
Per gradient 1.27-1.33x on CPU production threads at (2,205,2), 1.62-1.73x at (2,410,2), 2.1-3.1x at
(2,1024,2) (machine-dependent), 6.6-7.3x at (2,2048,4); 2.66x / 8.89x / 14.34x / 29.30x on the H100
in float64 at $N$ = 205 / 1024 / 2048 / 4096. Accuracy cost: none with the refined filter (1.8e-11
versus `main`'s 1.5e-8 at (2,205,2) `aligo02`; 1.4e-8 versus 2.4e-7 at (2,4096,4) `gw150914`); with
the PR's `solve_toeplitz` filter, 5-51x more digits lost than Cholesky on the GW150914 ACF at $(2, N
\geq 1024)$ (2.6x at (2,410,2), none at (3,1024,8)), still inside the $\epsilon\,{\rm cond}$
envelope. Implementation cost: moderate. A longdouble Levinson precompute ($O(N^2)$, needed anyway
for the exact $\sum_m\log\sigma_m^2$); the Gram kernel (`gs_kernels.py::gs_half_grams`); spectra,
$\sigma^2$, $w = C^{-1}y$, $Q$ and $\log|C|$ as model inputs; keeping `cholesky_factor` in
`get_arviz`, or storing $(a, \sigma)$ and whitening by the standardized innovations (math note
section 3.4); a per-backend FFT length; tests at the algebra-gate level and against an
extended-precision reference. None of this is in PR #141, which as written uses the wrong
log-determinant (D1), the slowest GS form (30 FFT ops per gradient against 8), un-hoisted in-body
spectra (7-9% on GPU), and breaks `Result` post-processing (D3).

**(iv) Size-dependent dispatch** (`gemm_linv` small $N$, `gs_half` large $N$). Buys at most 1.24x
over `gs_half` alone on the H100 at (2,205,2) in float64 (143.0 versus 177.4 us), 1.5x in float32,
and nothing on CPU, where `gs_half` already leads at every $N$ in production. Two code paths for a
sub-1.5x gain at one size on one platform is the trade `model_optimization_study.md` section 4.1
rejected for the detector loop; reject it here too.

**Recommendation.** Do not merge PR #141 as written. For the production CPU workflow at $N \approx
205$, do nothing on performance grounds: the best available per-gradient gain is 1.27-1.33x
(`cpu_f64_prod`, `gs_half`, both machines), bounded to 5-8% end-to-end by the gradient's 20-34%
share of a leapfrog step, and the PR's own kernel would make production slower (0.66-0.69x). If fits
at $N \geq 1024$ or on GPUs become routine, implement option (iii) as a single always-on path: the
only route never slower than `main` in production, with a real FFT-specific gain from $N = 2048$
(1.4-2.9x over the dense-GEMM control on the H100, 1.9-2.6x on CPU) and no numerical cost with the
refined coefficients. Option (ii) is a reasonable interim for a GPU-only deployment (3.3x per
gradient, minimal churn), with the caveat that it loses to `main` at (2,410,2) on one of the two
CPUs measured.

**What would have to be true for the answer to change.** (a) The gradient share of a leapfrog step
on GPU, or with `chain_method='vectorized'` at 16-64 chains, is much higher than the 20-34% measured
on CPU at one chain; then 2.7-3.3x per gradient on the H100 would be a large end-to-end gain and
(iii) or (ii) becomes worth doing now. (b) The `gs_half` per-leapfrog anomaly of section 5 is
explained and removed; until then any GS end-to-end number is untrustworthy. (c) Production $N$
grows: at $N = 1024$ the CPU gain is 2.1-3.1x per gradient, at 2048 6.6-7.3x. (d) float32 becomes a
production dtype: `main` then degrades 2-10x faster than the precomputed-whitening routes with ${\rm
cond}(C)$ (section 4.4), a reason to change the route independent of speed, though no route meets
1e-3 there.

---

## 7. Caveats and open items

1. **A6000 float64 is a smoke test.** GA102 runs FP64 at 1/32-1/64 rate; its `gpu_f64` speedups
   (3.5-18x) mostly measure a slow `dtrsm`. GPU float64 conclusions come from the H100 only.
2. **TF32.** The `gpu_f32` accuracy tables (JAX default matmul precision) measure TF32 rounding
   and are labeled as the production floor; GPU float32 route accuracy is read from
   `gpu_f32_hi`. Timing is unaffected (<3%).
3. **Single-chain NUTS, two seeds.** ESS/s ranks nothing; the `gs_half` per-leapfrog anomaly
   (section 5) is unexplained; no GPU NUTS was run.
4. **Typical-set points sit at the `m_max` edge** (`m_mean` 145-149.5, unconstrained $m$ ~4-9):
   they probe the near-mode cancellation but do not represent a well-contained posterior.
5. **$N = 4096$ warmups fell back to N(0,1) points.** The five non-gw150914 (2,4096,4) files and
   `aligo2` (3,1024,8) hit the 300 s warmup timeout and carry 20 normal points only.
6. **gw150914 antenna patterns and injection are synthetic.** Only the H1/L1 ACFs are real;
   design matrices, injection and SNR 20 are the kit's convention, not the GW150914 fit.
7. **Checklist items 3 and 4 are not implemented** ($U(p) - U(p_0)$ differences across points;
   dense-GS cancellation ratio, Gram-level cancellation and pre-symmetrization asymmetry as kit
   output). The verification pass measured the cancellation ratios once (1.003 / 1.000, a null
   result); the numbers live in its report, not in `analyze_gs.py` tables.
8. **The two machines disagree on CPU above $N = 410$** (sections 3.2, 3.5): the 32-thread XLA
   pool pathology is local only (hyperthreading, section 3.5, is the unmeasured candidate cause);
   `main`'s trsm is 2x faster single-threaded on the Platinum 8362 than on the Gold 6244. CPU
   speedups at $N \geq 1024$ are quoted as ranges.
9. **Math note section 11 (resolved).** An earlier text said the PR's `fft_as`, `fft_bs` "are
   constant-folded at compile time"; it now says they stay in the compiled gradient (measured: 4
   extra FFT ops on GPU, 7-9% of `gs_pr`'s time) and are hoisted out of `while` loops on CPU only.
10. **Fixed thresholds.** `[concerning]` at 1e-8 fires for every route including `main` at
    ${\rm cond} \geq 10^{10}$ (the digits-lost table is the meaningful statement there), and the
    reference's log-determinant floor (2e-8 nats at cond 3.5e11) is comparable to `main`'s
    potential error at (2,2048,4) and (2,4096,4) `gw150914`. Gradient tables are unaffected.
11. **Run 1 archive.** `benchmarks/gs/results/run1_pre_verify/` (tag `local-a6000`) predates the
    verification pass (float64 Levinson log-determinant, per-point gate normalization, `NPROC`
    coupled to `--omp`, no `cpu_f64_prod` or `gpu_f32_hi` leg, compile-included NUTS). Its
    `devtime`/`compile` numbers agree with run 2 within the quoted spreads; the rest is superseded.
12. **The kit README's results section predates run 2.** `benchmarks/gs/README.md` ("Results in
    `results/`") still describes `results/local.json` as the run made before the fixes, and its
    NUTS and thread paragraphs quote run-1 figures (share "25-40%", `gs_half` "13.9x / ~2.2x");
    run 2 gives 20-34% and 12.67x / 2.12x (sections 3.5, 5). To be updated; this note is current.

**Verification pass** (`/mnt/home/misi/.claude/jobs/79342a97/tmp/verify_result.json`): four lenses
(math fidelity, fairness, results audit, reproducibility), 21 confirmed findings each surviving
three independent refutation attempts, all applied to the kit before run 2 (fixes: longdouble
log-determinant, refined-coefficient diagnostic, TF32 leg, independent thread knobs and
`cpu_f64_prod`, hoisting footnotes, cloud normalization, per-kind f32 data, cold/warm NUTS, timing
spreads, sbatch budget, scale-invariance twin: `gw150914s1` `main` gradients agree to 3.1e-11 and
$U_{\rm lik}$ shifts by exactly $n_{\rm det}N\log s$ to 7.0e-10 nats). Not applied: caveat 7
(checklist items 3 and 4), a per-point ${\rm cond}(A^{-1})$ and $S/|g|$ record in the reference
section, and a thread-onset (`omp4`) timing leg; the math note correction (caveat 9) was applied
afterwards.

---

## 8. Reproduction

All commands from the repo root with the project venv; `benchmarks/` is excluded from ruff and
pre-commit; nothing under `ringdown/` is touched.

```bash
export PYTHONPATH=$PWD
PY=.venv/bin/python

# kernel self-tests (10 s)
JAX_PLATFORMS=cpu $PY benchmarks/gs/tests_kernels.py

# inputs (once; ~80 min on 8 cores with the NUTS warmups, ~5 min with --no-warmup; ~6.4 GB;
# needs the GW150914 data under data/ for the gw150914 family)
$PY benchmarks/gs/prep_inputs.py --out benchmarks/gs/inputs
# (--refresh-precompute upgrades pre-longdouble inputs in place; --configs "2,205,2"
#  --families gw150914s1 builds the scale-invariance twin; --smoke below = (2,205,2) only, ~5 min)

# full local matrix (7 legs, both paddings, NUTS in the multi-thread cpu_f64 leg; ~3.5 h)
mkdir -p benchmarks/gs/results
nohup $PY benchmarks/gs/bench_gs.py --platform gpu --inputs benchmarks/gs/inputs \
    --nfft both --sections env,reference,correctness,devtime,compile,f32acc,nuts \
    --out benchmarks/gs/results/local.json --tag local > benchmarks/gs/results/local.log 2>&1 &
$PY benchmarks/gs/analyze_gs.py benchmarks/gs/results/local.json --md > benchmarks/gs/results/local.md

# H100 (submit from the repo root; inputs prepared; the results dir must exist before sbatch)
mkdir -p benchmarks/gs/results
sbatch benchmarks/gs/submit_h100.sbatch
#   -> benchmarks/gs/results/results_${SLURM_JOB_ID}.json, then
$PY benchmarks/gs/analyze_gs.py benchmarks/gs/results/results_6974520.json --md > benchmarks/gs/results/results_6974520.md
```

`submit_h100.sbatch`: partition `gpu`, `--gres=gpu:h100_pcie:1`, 16 CPUs, 64 GB, 6 h, `--leg-timeout
10800`, `--cpu-omp $SLURM_CPUS_PER_TASK` (hence `cpu_f64_omp16`); `REF=<.ref.npz>` reuses a
reference and drops the `reference` section (~1.5 h). A single leg runs in-process with `--no-sub`,
e.g. `--platform gpu --x64 0 --no-sub --sections env,devtime,f32acc --ref <base>.ref.npz`.

**File map** (`benchmarks/gs/`): `gs_kernels.py` (precompute, log-determinant routes, spectra, JAX
appliers `gs_pr_cinv`, `gs_full_cinv`, `gs_half_grams` with numpy twins); `ref_longdouble.py`
(reference); `tests_kernels.py`; `variants.py` (nine model factories, `build_consts`); `harness.py`
(slope timing, HLO census, hoisting detection, leg spawning); `prep_inputs.py`; `bench_gs.py`;
`analyze_gs.py`; `submit_h100.sbatch`, `README.md`, `.gitignore`. Results: `results/local.{json,md,log}`
and `results/local.<leg>.json` (run 2); `results/results_6974520.{json,md}`, `.<leg>.json`,
`slurm-6974520.{out,err}` (H100); `results/run1_pre_verify/` (run 1, superseded).

**Not committed** (`benchmarks/gs/.gitignore`): `inputs/` (~6.4 GB, 37 npz files: 36 grid files plus
`gw150914s1`, 38 entries with `prep_log.json`; regenerate with `prep_inputs.py`), `results/*.npz`
(reference dumps, ~18 MB each), Slurm logs; `results/*.log` is re-included against the repository's
global `*.log` rule. The JSON and markdown results are committed. The venv's `jax[cuda12]` plugin
wheels are installed but not in `uv.lock` (`uv sync` removes them); do not `module load cuda` (see
`benchmarks/h100/README.md`).
