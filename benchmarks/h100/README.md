# H100 benchmark kit for the NumPyro ringdown model

Answers one question: **is "the CPU beats the GPU at the production problem size"
a property of the RTX A6000, or of this model?**  It re-runs the
`GPU_BENCHMARKS.md` measurements on a single NVIDIA H100 on `rusty`, in **both
float64 and float32**, together with a float64 CPU baseline taken **on the same
node** so the comparison is like-for-like.

`PREDICTIONS.md` states, in advance, what the H100 should and should not change,
with numeric intervals and explicit falsification criteria. (Development reports
cited by bare filename here — `GPU_BENCHMARKS.md`, `CLAIMS_VERIFICATION.md` — were
session documents that are not retained in the repository; their corrected substance
is consolidated in `docs/dev/model_optimization_study.md`.) `analyze.py` scores
the results against it.

---

## What you run

```bash
cd /path/to/ringdown
sbatch benchmarks/h100/submit.sbatch
```

When it finishes (`benchmarks/h100/slurm-<jobid>.out`):

```bash
.venv/bin/python benchmarks/h100/analyze.py benchmarks/h100/results_<jobid>.json
```

`analyze.py` is pure stdlib — no jax, no GPU — so it runs anywhere, including a
login node.

### The one placeholder you may need to change

`submit.sbatch` requests:

```
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100_pcie:1
```

The GRES type string `h100_pcie` was read from `scontrol show partition` on this
cluster, so it should be exact. **If Slurm rejects it** ("Invalid generic
resource specification" / "Requested node configuration is not available"),
edit those two lines to one of the alternatives commented immediately below them
in the file:

* `--partition=gpuxl` + `--gres=gpu:h100:1` — the H100 **SXM** parts (may need a
  specific allocation);
* `--partition=gpupreempt` + `--gres=gpu:h100_pcie:1` — preemptible, always
  available, but the job can be killed mid-run.

Everything else is filled in. Nothing needs to be installed or module-loaded:
the repo venv already has `jax 0.11.1` plus the `jax[cuda12]` pip plugin, and
those wheels **bundle their own CUDA 12.9 runtime**, so only the *node driver*
matters — CUDA 12.x minor-version compatibility needs driver **≥ 525.60.13**,
and every H100 node is far newer than that. Do not `module load cuda`; mixing
modules with the pip CUDA wheels is exactly the ABI mismatch the site guidance
warns about.

---

## What it costs

| | |
|---|---|
| walltime requested | 1 h |
| expected runtime | **~25-30 min** (`--deadline`, default 2000 s, is a *total* budget shared across the five legs; sections are run most-important-first and shed from the end rather than overrunning) |
| resources | 1 H100, 16 CPU cores, 64 GB |

**Why 16 cores.** The A6000 study's CPU baseline was this workstation's
2× Xeon Gold 6244 = **16 physical cores at 3.6 GHz**. Asking for 16 on the node
makes the same-node CPU-vs-GPU comparison a like-for-like core-count fight
rather than a handicap match, and it is close to the per-GPU share of a rusty
GPU node (~88 cores / 4 GPUs ≈ 22), so it schedules quickly. The benchmark also
runs an `OMP_NUM_THREADS=1` CPU leg, because that is what
`ringdown/cli/ringdown_fit.py:80-82` sets in production — so you get both the
best-case and the production-case CPU number and can bracket the answer.

---

## What comes back

One JSON (`results_<jobid>.json`) that embeds every leg, plus side files
(`.gpu_f32.json`, `.cpu_f64.json`, `.cpu_f64_chains.json`,
`.cpu_f64_omp1.json`, `.f64ref.npz`) that `analyze.py` does not need. The stdout in the `.out` file is a readable
transcript of the same thing.

Sections, in order:

1. **Environment** — `jax.devices()`, device kind, driver, compute capability,
   bundled cuBLAS/cuSOLVER versions, host CPU model, thread env, all `SLURM_*`,
   and a **GPU-contention snapshot** before and after (the A6000 numbers were
   taken on a verified-idle card; if the H100 is shared, absolute timings are
   contended and only ratios are usable — the analyzer says so loudly).
2. **Correctness** — every prototype vs the *unmodified*
   `ringdown.model.make_model`, on the potential **and every gradient
   component**, tolerance `1e-11`. Runs first; if it fails, nothing below means
   anything and the analyzer says so.
3. **Per-gradient device time**, float64 and float32, for
   `{current, whiten_seq, R1 unrolled ([M|y] concat), R1 unrolled (2 solves),
   R1 vmap, no-likelihood floor}` × `{(2,205,2), (2,1024,4), (3,1024,8)}`, plus
   the post-optimization HLO custom-call census (`__cublas$triangularSolve`,
   `cusolver_potrf_ffi`) which is the most robust evidence in the whole study.
4. **Triangular-solve RHS sweep** 8..40 at `n=205` (and a coarser one at
   `n=1024`), to see whether the cuBLAS **RHS ≥ 17 threshold** survives on
   Hopper. Same bundled cuBLAS build on both machines, so this isolates
   architecture-conditioned kernel selection from library version.
5. **Compile time**, isolated `jit(grad)` per variant and the full-NUTS
   compile+setup for `R1_vmap`.
6. **Vectorized-chain scaling** `{1,4,16,64}` × `{R1 unrolled, R1 vmap}` at the
   production point, 250 warmup + 250 samples, both precisions, plus the shipped
   `num_chains=4` default (which on one GPU silently degrades to *sequential*).
7. **float32 accuracy and robustness** — value and gradient against a float64
   reference computed at identical unconstrained points, over a sweep of
   `a_scale_max` and a degenerate-mode case, including the `cond(A)` at which the
   shipped sequential form goes NaN while R1 stays finite.
8. **CPU float64 baseline on the same node**, in three legs: per-gradient
   timings on **one** CPU device (as in the A6000 study), chain scaling with
   `numpyro.set_host_device_count(4)` (mirroring
   `ringdown/cli/ringdown_fit.py:104-122` -- without it the shipped
   `num_chains=4, chain_method='parallel'` degrades to *sequential* on CPU as
   well and the baseline would not be the production baseline), and a
   per-gradient leg at `OMP_NUM_THREADS=1`, which is what the production CLI
   sets.

The methodology is deliberately the one that survived `CLAIMS_VERIFICATION.md`:
the device-time harness (R gradients inside one jitted `fori_loop` with an inert
`1e-30` feedback so XLA cannot CSE the body, slope between two `R` values,
medians of ≥5), and platform/precision selected **before** `import jax` — the
verifier documented that setting `JAX_PLATFORMS` after the import fails silently
and yields GPU numbers labeled "CPU". Each leg is therefore a separate process.

---

## What `analyze.py` prints

* H100 vs A6000 vs same-node CPU, per gradient, per config, per precision.
* **The headline table**: best-config-to-best-config at the production point for
  GPU-f64 / GPU-f32 / CPU-f64(all cores) / CPU-f64(OMP=1) — including the
  comparison you actually care about, **same-node CPU float64 vs GPU float32**.
* Chain scaling with the crossover chain count against the node's own best CPU
  configuration, in both precisions.
* The RHS threshold, the float32 accuracy table, compile times.
* A **scorecard**: every numbered prediction from `PREDICTIONS.md` marked
  HIT / MISS / n-a with the predicted interval shown, and the five explicit
  **falsification tests** for the "latency-/launch-bound" diagnosis, ending in a
  single verdict line.

---

## Local validation

`bench.py` was run end-to-end on this workstation's RTX A6000 in `--smoke` mode
(reduced repetitions and configs) before the kit was handed over, and
`analyze.py` was run on the resulting JSON. Both precision paths, all five legs
and every section executed. Notes from that run:

* **The A6000 numbers reproduce `GPU_BENCHMARKS.md` closely** on a quiet card.
  Per-gradient device time at (2,205,2), float64, this kit vs the published
  table: `current` 2946 vs 3072 us, `whiten_seq` 1494 vs 1412,
  `R1_unroll_concat` 1348 vs 1187, `R1_unroll_sep` 1178 vs 1207, `R1_vmap`
  **898 vs 913**, no-likelihood floor **66.9 vs 66.4**. On the CPU side,
  `R1_unroll_concat` 272 vs 278 us and the floor 63.6 vs 60.1; at
  `OMP_NUM_THREADS=1` the best CPU formulation is 333 us against the verifier's
  356. The float32 gains land at 2.76x / 3.63x / 2.28x against the published
  2.73 / 2.94 / 2.19. Best-to-best, this kit puts the A6000 at **3.30x slower
  than the workstation CPU in float64** against the published 3.34x -- and, new,
  at only **1.29x slower in float32** (1.06x against the OMP=1 CPU).
* The RHS sweep gives 100/148/574/541/348 us at k=8/16/17/18/32 against the
  verifier's 82/132/462/440/279, a **3.88x** threshold at k=17 against their
  3.51x -- and **no cliff at all in float32** (largest step 1.22x). The `[M|y]`
  penalty also shows up in the *full model* at (2,1024,4), k=16:
  `R1_unroll_concat` 14257 us vs `R1_unroll_sep` 6226 us, i.e. R2b reproduced at
  a size the original study did not test.
* The GPU 4-chain default reproduces the documented failure mode verbatim:
  *"There are not enough devices to run parallel chains: expected 4 but got 1."*
* Correctness reproduces **exactly**: the `current` prototype agrees with the
  repo model to `0.0e+00`, and the HLO census reproduces the published counts
  bit-for-bit — `__cublas$triangularSolve` 42 → 18 (whiten_seq) → 8 (concat) →
  10 (sep) → 7 (vmap), `cusolver_potrf_ffi` 2 → 1.
* The float32 section reproduces `GPU_BENCHMARKS.md` §3.2 qualitatively: the
  potential is good to ~5e-7 while the **gradient** is only good to ~4e-4 at
  benign conditioning, and the shipped sequential form goes **NaN** at high
  `cond(A)` where both R1 forms stay finite. It does **not** reproduce the
  claim that R1's f32 gradient is 7× *more accurate* — here R1 is ~2× worse at
  benign conditioning while being far more robust at bad conditioning. Recorded
  in `PREDICTIONS.md` P12.
* **Contention matters and is now recorded.** An earlier smoke run, taken while
  another process held the workstation's A6000 at 100% utilization and 82 C
  (thermal throttling to 1800 of 2100 MHz), produced absolute timings 2-3x above
  the published table with the variant ordering scrambled. That is why the kit
  snapshots GPU compute processes and clocks before and after, and why the
  analyzer prints a loud warning when a foreign process is on the card. On a
  dedicated Slurm GPU there is no such confound.
* The **no-likelihood floor is the noisiest single number** in the kit (it is a
  handful of microseconds of kernel launches, measured as a slope), and it moved
  by several x between contended smoke runs. Prediction P3 is correspondingly
  wide; treat a floor measurement taken on a busy card as uninformative.
* Four defects were found and fixed by this validation, which is what a dry run
  is for: numpyro validates `chain_method` against a literal set (the "shipped
  default" row must pass `'parallel'`, not `None`); the CPU leg needs
  `set_host_device_count(4)` to reproduce the production 4-chain configuration,
  but only in the chains leg, since splitting XLA:CPU into four logical devices
  would perturb the per-gradient timings; and the float32 accuracy check must be
  gated to the float32 leg, or it compares the float64 reference against itself
  and reports a perfect 0.0.

---

## What was trimmed, and why

Adding the second precision roughly doubled the GPU work, so to stay inside the
30-minute budget:

* **`(2, 1024, 2)` was dropped from the size grid** (the grid is now
  `(2,205,2)`, `(2,1024,4)`, `(3,1024,8)`). `(2,1024,4)` was never measured on
  the A6000, so `analyze.py` shows a clearly-labeled bracketed estimate for
  that row; `(2,205,2)` and `(3,1024,8)` are exact A6000 comparisons.
* The **shipped `current` model runs only at 1 and 16 chains** (it is 3–4×
  slower and only needed as the status-quo anchor); the R1 variants get the full
  `{1,4,16,64}` sweep.
* The **full-NUTS compile decomposition** (the `N ∈ {250,1000}` affine fit) is
  taken for `R1_vmap` only, and only in float64. `CLAIMS_VERIFICATION.md` flags
  that decomposition as fragile anyway — chains decorrelate between variants —
  so raw wall times and leapfrog counts are recorded alongside it.
* The float32 leg uses a **coarse RHS sweep** (14 points) rather than the full
  8..40, and skips the `n=1024` sweep and the `(3,1024,8)` compile row.
* The CPU chain leg runs 1 chain plus the 4-chain default rather than the full
  chain sweep — vectorized chains are a GPU strategy, and the CPU number that
  matters is the shipped 4-chain configuration.

Nothing that carries a numbered prediction was trimmed.

## Files

| file | what it is |
|---|---|
| `bench.py` | the benchmark; self-contained, spawns its own precision/platform legs |
| `submit.sbatch` | the Slurm job (1 H100, 16 cores, 1 h) |
| `analyze.py` | ingests the JSON, prints comparisons and the prediction scorecard |
| `PREDICTIONS.md` | the a-priori analysis, numbered predictions, falsifiers |
| `README.md` | this file |
| `smoke.json`, `smoke.log` | the A6000 validation run described above; `analyze.py smoke.json` reproduces the transcript |

`bench.py --help` documents every flag; `--smoke` reproduces the reduced local
validation run in ~6 minutes.
