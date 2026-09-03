# Gohberg-Semencul vs Cholesky benchmark kit

Standalone benchmark of several ways to apply the inverse noise covariance
inside ringdown's one-shot marginalized likelihood, motivated by PR #141
(Gohberg-Semencul FFT route) and the math note
`docs/gohberg_semencul_likelihood.md`.  Nothing in the `ringdown` package is
modified; `benchmarks/` is excluded from ruff and pre-commit.

## Files

| file | role |
|---|---|
| `gs_kernels.py` | numpy precompute (Levinson, float64 log-det diagnostic, dense factors, spectra in a chosen dtype, FFT lengths) and JAX appliers `gs_pr_cinv`, `gs_full_cinv`, `gs_half_grams` with numpy twins |
| `ref_longdouble.py` | extended-precision (x87 longdouble) reference: refined `C^{-1}B`, longdouble Levinson (filter, innovation variances, log-det), longdouble dense Cholesky log-det, one-shot likelihood and closed-form gradient wrt the design matrices |
| `tests_kernels.py` | self-tests of the two modules above (numpy + jax on CPU) |
| `variants.py` | numpyro model factories, one per variant, all with the sites of `ringdown.model.make_model(marginalized=True)`; `build_consts` (with the `coeffs` and `spectra_from` policies), `head_design_matrices` |
| `harness.py` | timing (fori_loop slope method with a repeatability spread) and HLO census, copied from `benchmarks/h100/bench.py` and extended (gemm/fft counts, N x N vs k x k trsm, hoisting detection) |
| `prep_inputs.py` | builds the input npz files (ACF families, strains, constants, fixed points); `--refresh-precompute` upgrades existing files |
| `bench_gs.py` | the benchmark: sections `env, reference, correctness, devtime, compile, f32acc, nuts`; parent spawns one process per leg |
| `analyze_gs.py` | stdlib-only tables (text or `--md`) from the results JSON, with the caveats below printed next to the tables they apply to |
| `submit_h100.sbatch` | Slurm script for an H100 run (user submits manually) |
| `.gitignore` | `inputs/` (~6.4 GB), `results/*.npz`, Slurm logs are not committed; `results/*.log` is re-included (the repo ignores `*.log`) |

## Variants

`main` (ringdown as is), `main_hoisted` (z = L^{-1} y, Q, log|C| precomputed),
`gemm_linv` (dense L^{-1}, GEMM), `gemm_cinv` (dense C^{-1}, GEMM), `gs_pr`
(PR #141 kernel with spectra passed in), `gs_pr_ascoded` (spectra computed in
the model body, as the PR does), `gs_full` (batched GS, 4 FFT passes),
`gs_half` (Gram form, one rfft + one batched irfft), `floor` (prior only).

Every hoisted variant uses the exact log-determinant `sum_m log sigma_m^2`
**evaluated in longdouble** (`logdetC_exact` in the inputs).  The float64
evaluation of the same formula is kept as a diagnostic (`logdetC_f64lev`): it
is the least accurate of the three log-det routes at high conditioning
(gw150914: 2.7e-8 nats off at N=1024 / cond 5e8, 1.4e-5 at N=2048, 1.6e-4 at
N=4096 / cond 1.5e12, against 9e-10 / 2e-7 / 2e-6 for ringdown's
`2 sum log diag L` and 2e-11 / 2e-8 / 1e-7 for the longdouble recursion), so
using it as the model constant inflated the hoisted variants' potential
error by a theta-independent constant unrelated to the C^{-1} route.  The
PR's `N log sigma^2` shortcut is recorded as `logdetC_pr` (21.8 / 20.9 nats
off per detector on aligo02 at N=205: harmless for sampling, wrong for
evidence).

Yule-Walker coefficients (`--gs-coeffs`, default `pr`): `gs_pr`, `gs_full`,
`gs_half` take `a`, `atilde`, `sigma^2` from `scipy.linalg.solve_toeplitz`
exactly as PR #141 does (`pr`), or from a longdouble Levinson recursion
rounded to float64 (`refined`: `a_ld`, `atilde_ld`, `sigma2_ld`).  The
`correctness` section always evaluates the *other* policy too, through the
same compiled function (`alt_coeffs`), because the forward error of the
float64 Yule-Walker solve (4e-9 relative on gw150914 at cond 5e8) is what
limits the GS gradients there: at (2,205,2) aligo02 the GS error vs the
reference drops from 2.2e-8 to 8.9e-12 with the refined filter (main:
1.0e-8); the note's section 9.3 "F1: no gap observed" is ACF-dependent.
`gs_pr_ascoded` always takes the PR filter (it is the PR's model body).

Spectra in float32 legs (`--spectra-from`, default `f64`): the spectra of
`gs_pr`/`gs_full`/`gs_half` are the float64 rfft of the float64 filter cast
to complex64 (`f64`), or the rfft of the float32-cast filter in float32
(`leg`, what `gs_pr_ascoded`'s in-model `jnp.fft.rfft` and a float32
production path do).  `f32acc` evaluates the other policy as `alt_spectra`;
the two differ at the float32-eps level (~1e-7 relative) and do not change
any float32 conclusion.

## Running

All commands from the repo root with the project venv.

```bash
export PYTHONPATH=$PWD
PY=.venv/bin/python

# 1. kernel self-tests (10 s)
JAX_PLATFORMS=cpu $PY benchmarks/gs/tests_kernels.py

# 2. inputs (once; ~80 min on 8 cores WITH the default NUTS warmups -- ~75 min of it is the
#    warmup children, ~30 min the six (2,4096,4) files whose warmups mostly hit the 300 s
#    timeout; --no-warmup ~5 min; --smoke gives the (2,205,2) aligo02/white pair in ~1 min; ~6.4 GB)
$PY benchmarks/gs/prep_inputs.py --out benchmarks/gs/inputs
#    upgrade inputs written before the longdouble log-det / refined-filter keys existed (minutes, in place)
$PY benchmarks/gs/prep_inputs.py --refresh-precompute --out benchmarks/gs/inputs
#    optional scale-invariance twin of gw150914 (checklist item 8)
$PY benchmarks/gs/prep_inputs.py --out benchmarks/gs/inputs --configs "2,205,2" --families gw150914s1

# 3. smoke: every leg, (2,205,2) only, ~5 min on a workstation with a GPU
$PY benchmarks/gs/bench_gs.py --smoke --platform gpu --inputs benchmarks/gs/inputs \
    --out /tmp/gs_smoke.json
$PY benchmarks/gs/analyze_gs.py /tmp/gs_smoke.json --md

# 4. full local matrix (all legs, both FFT paddings, NUTS in the cpu_f64_omp8 leg; ~3.5 h)
mkdir -p benchmarks/gs/results
nohup $PY benchmarks/gs/bench_gs.py --platform gpu --inputs benchmarks/gs/inputs \
    --nfft both --sections env,reference,correctness,devtime,compile,f32acc,nuts \
    --out benchmarks/gs/results/local.json --tag local > benchmarks/gs/results/local.log 2>&1 &
$PY benchmarks/gs/analyze_gs.py benchmarks/gs/results/local.json --md

# 5. H100 (from the repo root, inputs already prepared; mkdir -p benchmarks/gs/results first)
sbatch benchmarks/gs/submit_h100.sbatch
```

A single leg can be run in-process with `--no-sub`, e.g.
`--platform gpu --x64 0 --no-sub --sections env,devtime,f32acc --ref <out base>.ref.npz`
(`--ref` points at the npz written by the cpu_f64 leg's `reference` section).
`--platform cpu` on the parent spawns only the CPU legs.  `--cpu-omp N`
(default 8) sets the thread count of the multi-thread CPU legs, which are then
named `cpu_f64_ompN` / `cpu_f32_ompN`.  Output files: `<out>` (parent, all
legs embedded), `<out base>.<leg>.json` per leg (`results/local.cpu_f64_omp8.json`),
`<out base>.ref.npz` (the reference dump, ~18 MB).

### GPU environment

The venv's `jax[cuda12]` plugin wheels (`jax-cuda12-plugin`, `jax-cuda12-pjrt`
and the `nvidia-*-cu12` 12.9 libraries) are installed in `.venv` but are not
in `uv.lock`, so `uv sync` removes them; do not `module load cuda` (the pip
wheels bundle their own CUDA), see `benchmarks/h100/README.md`.  JAX's
default matmul precision lets float32 dots run in **TF32** on Ampere+ GPUs
(the `matmul_prec None` column of the env table).  That is what production
does, but it means the `gpu_f32` accuracy tables measure TF32 rounding in the
k x k and N x k dots, not the C^{-1} route: on white noise (cond 1) every
variant shows 4e-2..1e-1 where `cpu_f32` gives 3e-4..1e-3, and with
`--matmul-precision highest` the errors drop 50-600x to the cpu_f32 level
and the variant ordering changes (at (2,1024,2) aligo02 `main` goes from best
to worst).  The `gpu_f32_hi` leg runs `f32acc` with `highest`; read float32
route accuracy from it and treat the `gpu_f32` accuracy table as the
production-TF32 floor (it is labelled as such by `analyze_gs.py`).

## Legs and sections

| leg | argv | sections |
|---|---|---|
| `cpu_f64_omp8` | `--omp 8 --xla-threads 8` | env, reference (writes `<out base>.ref.npz`), correctness, devtime, compile, f32acc, nuts; runs the N=4096 config |
| `cpu_f64_omp1` | `--omp 1 --xla-threads 1` | env, correctness, devtime, compile (1 BLAS thread, 1-thread XLA pool; no N=4096) |
| `cpu_f64_prod` | `--omp 1` | env, devtime, compile (production: 1 BLAS thread, XLA pool = one thread per core; no N=4096) |
| `cpu_f32_omp8` | `--omp 8 --xla-threads 8` | env, f32acc, devtime, compile (no N=4096) |
| `gpu_f64` | | env, correctness, f32acc, devtime, compile (A6000 FP64 is 1/64 rate: local numbers are a smoke test) |
| `gpu_f32` | | env, f32acc, devtime, compile (TF32 dots) |
| `gpu_f32_hi` | `--matmul-precision highest` | env, f32acc (true float32 dots; timing is unaffected, <3%) |

`devtime` and `compile` time only the families in `--timing-families`
(default `aligo02,expcos`), `compile` only the first of them; `correctness`,
`reference` and `f32acc` run every family.

* `correctness` (f64): likelihood part `U_var - U_floor` and every gradient
  component vs `main` (gate 1e-11 on `white`/`expcos`) and vs the longdouble
  reference (flag > 1e-8 concerning, > 1e-6 fail; never silenced).
  Gradient errors are **cloud-normalized**: `|g - g_ref|` per site divided by
  the maximum of `|g_ref[site]|` over *all* points of the cell, so a scalar
  site whose likelihood gradient is near zero at one point (the NUTS
  typical-set points sit near the mode: `|g_chi|` down to 0.02 against ~70
  over the cloud) does not turn a 1e-13 roundoff difference into a 1e-11
  gate failure.  The per-point normalization (`max|g_ref[site]|` at the same
  point) is kept as `*_perpoint` / plain `rel_grad_vs_*` keys.  Also
  reported: `err_over_eps_cond = err / (eps cond C)` ("digits lost beyond
  conditioning", ~1 means as good as the conditioning allows), per-kind
  maxima, the `alt_coeffs` diagnostic, and for a `gw150914s1` family the
  scale twin check against `gw150914` (main's gradients must agree, U_lik
  must differ by exactly `n_det N log scale`; measured 3e-11 and 7e-10 nats).
* `reference` (cpu_f64 leg only): longdouble one-shot likelihood and
  closed-form gradient at every point.  Its log|C| is a longdouble dense
  Cholesky by default (`--ref-logdet cholesky`; ~1 s per detector at N=1024,
  ~80 s at N=4096, so the section takes ~20 min longer than with
  `levinson`); the longdouble Levinson sum is recorded alongside as
  `logdet_ld_levinson` and their difference (`logdet_ld_chol_minus_lev`,
  2e-11 nats at cond 5e8, 2e-8 at 3.5e11) is the floor of the potential
  comparison.
* `devtime`: device us/gradient by the slope method with the constants as jit
  arguments, plus `us_spread` (range of the slope over the individual
  repetitions: a repeatability estimate, printed as `+-x%`), the resolved
  `nfft` and the thread configuration; census of the plain and of the looped
  gradient (hoisting).  For N >= 1024 the `fast` and `pow2` paddings coincide
  (2048/2048, 4096/4096, 8192/8192 vs 432/512 at N=205 and 864/1024 at
  N=410), so the `@fast`/`@pow2` columns there time the same executable
  twice; `analyze_gs.py` lists those pairs and their spread as the per-leg
  repeatability (local run: median 0.4-5%, max 5% on cpu_f64_omp1, 15% on
  cpu_f64_omp8, 40% on gpu_f32), together with the aligo02-vs-expcos spread
  of the same shape.  Speedup differences below that spread are not resolved.
* `f32acc`: errors vs the reference and vs the cpu_f64 twin at the same
  points, per point and cloud-normalized, overall and **per point kind**,
  tabulated against cond(C) by `analyze_gs.py` and labelled with the matmul
  precision.  Read trends from the per-kind table: the typical-set points
  carry a k x k-tail cancellation the N(0,1) points do not (cpu_f32 white
  (3,1024,8): normal 3.0e-5, warmup 3.1e-1), and several cells (aligo2
  (3,1024,8), the non-gw150914 (2,4096,4) files) have no warmup points at
  all, so the overall column mixes populations and its ordering is not a
  cond(C) trend.
* `nuts`: short NUTS runs at (2,205,2) aligo02 (`--nuts-variants`, default
  `main,main_hoisted,gs_half,gemm_linv`; `--nuts-n`), cpu_f64_omp8 only,
  never implied by `--sections all`.  Per variant one cold run (seed 0,
  compile included: ~10 s for `main` vs ~0.2 s for `gemm_linv`) and then one
  warm run per `--nuts-seeds` (default `0,1`) on the same MCMC object;
  us/leapfrog and ESS/s come from the warm runs and ESS is reported per
  seed (the seed-to-seed spread, main 111-181 at 300+300, exceeds the
  variant-to-variant differences: with < ~4 seeds the ESS/s column ranks
  nothing).  The table also shows the device us/grad of the same cell as a
  share of the leapfrog step (25-40% on CPU at this size).

## Thread configuration (CPU legs)

`--omp N` sets `OMP/MKL/OPENBLAS_NUM_THREADS`: the BLAS/LAPACK custom calls
(trsm, potrf, GEMM).  `--xla-threads N` sets `NPROC`, which is what sizes
XLA:CPU's Eigen intra-op pool (FFT, dot, fusions; 1/8/32 `tf_XLAEigen`
threads for NPROC = 1/8/unset on a 32-core box).  The two are independent
knobs and the kit records both (`thread_env`, `xla_cpu_pool_threads`).
Production ringdown (`import ringdown` sets only `OMP_NUM_THREADS=1`) runs
with 1 BLAS thread and a full-size XLA pool, which is the `cpu_f64_prod` leg;
neither `omp8` nor `omp1` matches it.  The pool size changes the *ranking*:
with an 8- or 32-thread pool the batched-FFT variants (`gs_full`, `gs_half`),
the Eigen dots (`gemm_*`) and the prior-only `floor` run 2-3x slower at
N >= 1024 than with a 1-thread pool (parallel overhead on (2,k,nfft) irffts)
while `main`'s trsm gets faster; `gs_pr`'s per-column vmapped FFTs are
insensitive (~2.8 ms at N=1024 in every setting).  So at (2,1024,2) aligo02
`gs_half` is 13.9x faster than `main` on cpu_f64_omp1 but ~2.2x in the
production configuration, and the omp8 net-of-floor exponents at k=16 are
meaningless.  Quote every CPU speedup with its thread setting; the
`analyze_gs.py` devtime tables print it.

Hoisting: XLA:CPU moves theta-independent work out of `while` loops (the
timing fori_loop, NUTS tree building) and XLA:GPU does not.  For `main` that
is the two N x 1 solves z = L^{-1} y (census `moved`: trsm 10->8), so on CPU
legs `main ~= main_hoisted` and "speedup vs main" excludes the per-gradient
recomputation production pays on GPU (where `main_hoisted` is 4-6% faster in
f64 and `gs_pr_ascoded` ~10% slower than `gs_pr`).  Use "vs main_hoisted"
as the route comparison on every platform.  Contrary to an earlier version
of this kit's docstrings, XLA does *not* constant-fold ops on closed-over
arrays (the rfft of an embedded constant stays in the compiled gradient on
both backends); the constants are passed as jit arguments so that the
plain-gradient census counts every op regardless.

## Inputs

`prep_inputs.py` writes `{family}_d{n_det}_n{N}_m{n_modes}.npz` for the grid
(2,205,2), (2,410,2), (2,1024,2), (2,2048,4), (3,1024,8), (2,4096,4) and the
families `aligo02`, `aligo2`, `aligo20` (analytic aLIGO ACF, wall floored at
f_low = 0.2, 2, 20 Hz), `expcos`, `white`, and `gw150914` (H1/L1 ACFs from
`etc/ringdown_fit_example.ini`, strain scaled by its std as
`Fit.strain_scale` does); the opt-in `gw150914s1` is the same data with
scale = 1.  cond(C) is recorded per (family, N) in `inputs/prep_log.json`
and grows with N: aligo02 5.7e9 (N=205) to 8.2e10 (N=4096); aligo2
3.8e6-7.9e6; aligo20 ~5.6e2; expcos ~7.5e2; white 1; gw150914 4.2e5 / 1.8e6
(N=205) to 4.0e11 / 1.5e12 (N=4096, the two detectors).  The data convention
is `tests/test_model.py::_make_data` (unit-normalized ACF, injected ringdown
with `f = linspace(150, 300)`, `g = linspace(30, 80)`, N(0,1) quadratures,
seed 42) with one change: the noiseless injection is rescaled so that the
network optimal SNR `sqrt(sum_i h_i^T C_i^{-1} h_i)` equals `--snr` (default
20) before the noise is added, and the prior bound is `a_scale_max = 5
a_true` (`_make_data`'s O(1) injection against the tiny in-band variance of
the unit-normalized aLIGO ACF had SNR ~1e4 and `Q = y^T C^{-1} y` ~1e8 at
N=205; now `Q ~ n_det N + snr^2`).  Each npz records `snr_target`,
`snr_achieved`, `snr_recipe` (the SNR before rescaling), `a_true`,
`a_scale_max`, `Q`, the PR-style and refined Yule-Walker filters and the
three log-det evaluations.  See the module docstring for every key.
`inputs/` (~6.4 GB) and `results/*.npz` are not committed (`.gitignore`);
regenerate with `prep_inputs.py`, or upgrade older files in place with
`--refresh-precompute`.

Fixed points come in two kinds (`pts_kind`): 20 N(0,1) draws in numpyro's
unconstrained coordinates (kind 0, seed 7) and 10 typical-set points (kind 1):
the post-warmup samples of a NUTS run (`num_warmup=150`, `num_samples=10`,
1 chain, `PRNGKey(0)`, CPU float64) of `ringdown.model.make_model(
marginalized=True)` on that file's data, mapped back to unconstrained
coordinates with `numpyro.infer.util.unconstrain_fn`.  The NUTS run happens in
a child process bounded by `--warmup-timeout` (300 s); a failure leaves the
20 kind-0 rows and is recorded in `inputs/prep_log.json`, which also holds
cond, SNR, Q and the warmup diagnostics per file (6 of the 36 warmups hit
the timeout: the five non-gw150914 N=4096 files and aligo2 (3,1024,8), so
those cells have normal points only).  With the Kerr 220/22n modes and
`m <= 150`, the injected 150 Hz content pushes the posterior to the `m_max`
prior edge, so the kind-1 points have unconstrained `m` ~ 4-9.

`bench_gs.py --npts K` uses the first K points *of each kind* (default: all
30; `--smoke`: 3 + 3).  Every accuracy section reports the maximum relative
gradient error per kind (`*_by_kind`) next to the overall maxima, and the
reference npz stores the points it used so a stale reference is detected.

## Checklist status (design-critique items enforced in the verify/fix pass)

1. f32 spectra policy: `--spectra-from {f64,leg}` plus the `alt_spectra`
   diagnostic in `f32acc`; immaterial (f32-eps level).  Done.
2. Typical-set points from a short NUTS warmup: `pts_kind` 1.  Done.
3. Potential differences `U(p) - U(p0)` across points: not implemented; the
   nats columns report `U_lik = U - U_floor` against the reference, whose
   theta-independent constants are now evaluated in longdouble.
4. GS cancellation diagnostics (dense-GS vs refined C^{-1}, cancellation
   ratio, Gram-level for gs_half): not in the kit's output.  The verify pass
   measured cancellation ratios of 1.003 (vector form) / 1.000 (Gram form)
   on the real design matrices and identical errors for dense GS in float64
   and longdouble arithmetic, i.e. a null result: the GS error is the
   precompute's forward error (item covered by `alt_coeffs`).
5. Census inside vs outside the while body for every variant: `census_grad`
   vs `census_looped.body`, `hoisted`.  Done.
6. `err / (eps cond C)`: `err_over_eps_cond` in `correctness`, rendered by
   `analyze_gs.py`.  Done.
7. NUTS with `main_hoisted` and us/leapfrog from `num_steps`: default
   variants include it; cold/warm split, `us_per_leapfrog_warm`.  Done.
8. gw150914 scale invariance: `gw150914s1` family and the `_scale_twin`
   check.  Done (3.1e-11 gradient, 7.0e-10 nats at (2,205,2)).

## Results in `results/`

`results/local.json` (+ per-leg files, `local.ref.npz`, `local.log`,
`local.md`) is the local A6000/Gold-6244 run made **before** the fixes
above: its inputs carried the float64 Levinson log|C| (so the nats columns of
the hoisted variants include that bias), its gate/flags use the per-point
normalization, its `f32acc` has no per-kind data, its NUTS section is the
single cold run, its `cpu_f64_omp8`/`omp1` legs had NPROC coupled to `--omp`
(recorded as such by `analyze_gs.py`), and it has no `cpu_f64_prod` or
`gpu_f32_hi` leg.  The timing numbers (`devtime`, `compile`) and the
gradient columns are unaffected by the fixes; the `reference`,
`correctness`, `f32acc` and `nuts` sections need a rerun for the new
fields.  `analyze_gs.py` renders the old file with the corresponding caveats.
