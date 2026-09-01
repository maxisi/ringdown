# Model optimization study — the one-shot marginalized likelihood (R1)

> **Status (2026-09-01): MERGED.** R1 shipped as **PR #164**, merged to `main` as **`b6e30b9`**
> on 2026-09-01 ("Rewrite the marginalized likelihood in closed form (2.5x faster per
> gradient)"). Its six commits were `7b5418a` tests, `ce5cf8a` the model rewrite, `b233c93`
> docs + changelog, and three review fixes `ee33bac`, `d682181`, `e022f70`. This branch has
> been fast-forwarded onto the merge, so **the working tree now holds the new one-shot code**
> and `tests/test_model.py` (15 tests, green). §7 is retained as the **plan of record** — the
> rationale behind each decision — with an executed-status block and the deviations at §7.0.
> §§1–6 are unchanged analysis.
>
> ### Reading the code references
>
> **The purpose of this document is to explain the equivalence between the old and the new way
> of writing the likelihood, so the old code is quoted deliberately and is not going to be
> excised.** It no longer exists in the tree, so every reference to it is pinned to an
> immutable anchor: **`7bc480a`**, the last commit before the rewrite (equivalently `ce5cf8a^`).
> Read it with `git show 7bc480a:ringdown/model.py`. Such references are written
> "`model.py:789-906` (at `7bc480a`, pre-R1)"; the pinning is stated once here and abbreviated
> to "at `7bc480a`" below.
>
> References to the **new** code carry no qualifier and point at the merged tree
> (`b6e30b9`) — the one-shot block is `ringdown/model.py:822-862`, the predictive draw is
> `:905`. Those line numbers will drift with future edits; the pinned ones will not.

This is the consolidated record of the investigation and the plan the implementation followed.

**Scope.** JIT/compile-time and per-leapfrog-step optimization of the NumPyro models built by
`ringdown.model.make_model`, with essentially all attention on the **marginalized generic**
model (`marginalized=True`, the `make_model` default) — the LVK-reviewed production path.
`aligned` and `single_polarization` are logged as not reviewed for LVK use and are
deprioritized throughout.

**Hard constraint respected throughout.** Every recommendation preserves the posterior density
exactly (up to the *same* parameter-independent additive constant the current code already
drops, $-\tfrac12\sum_i n_i\log2\pi$), the priors, and the sampled parameterization. Items that
would change geometry, dtype or parameterization are quarantined in §4.6.

---

## 0. How to read this document

### 0.1 Provenance and the archive

This note consolidates six development reports, produced by successive agents over one working
session. The originals were not retained in the repository; their corrected substance is
consolidated here (superseded claims are cataloged in §0.3), and they are cited below by their
original filenames: `MODEL_OPTIMIZATIONS.md`
(the original CPU analysis, the R1 derivation, the R1–R6 ranking), `GPU_BENCHMARKS.md` (the
RTX A6000 re-run), `CLAIMS_VERIFICATION.md` (an independent adversarial re-derivation and
re-measurement of both), `AMPLITUDE_SCALE_PLACEMENT.md` (where the amplitude scales should
live), `BACKEND_DISPATCH.md` (whether to dispatch on backend) and `R1_PR_PLAN.md` (§7 here).

> **The archived originals contain claims that were later superseded.** Where they and this
> document disagree, **this document is correct**; §0.3 lists every correction. The archive is
> kept because it holds the full methodology sections and raw-number appendices this note
> deliberately drops, and because it is the evidentiary record for a scientifically
> load-bearing change.

Three things are **not** superseded: **`docs/marginalized_likelihood.md`**, the standalone
methods note and single source of truth for the mathematics, which ships with the package;
**`benchmarks/h100/`**, the H100 benchmark kit (`PREDICTIONS.md`, `bench.py`, `analyze.py`, raw
`results_*.json`) — re-score with
`JAX_PLATFORMS=cpu .venv/bin/python benchmarks/h100/analyze.py benchmarks/h100/results_6969321.json`;
and **PR #164** (merged as `b6e30b9`), which is the implementation and the end-to-end
validation, and which supersedes this document wherever the two differ on *what the code does*
(§7.0). The merged code is now in this tree.

**Division of labor with the methods note.** `docs/marginalized_likelihood.md` is
**outward-facing**: it carries the mathematics and nothing else. Development history —
which internal report measured what, which earlier account was superseded, benchmark
harnesses and protocols, tolerance rationale, archive paths, PR mechanics — belongs **here**,
not there. The note keeps exactly one pointer back to this document (in its §0) and one
implementation reference (`b6e30b9`); everything else that used to be narrated there has been
moved into this document, principally §0.3, §5.1 and §5.3. When editing either file, keep that
split: if a passage would only make sense to someone who worked on the rewrite, it goes here.

### 0.2 Notation

**This note follows `docs/marginalized_likelihood.md`, which follows the code and
arXiv:2005.14199:** $A$ is the marginal posterior **covariance** of the quadrature amplitudes
and $A^{-1}$ its **precision**; $\Lambda$ is the amplitude prior **covariance**. So the object
actually built and factorized is $A^{-1}$ (the code's `A_inv`), and $A$ is never formed. The
archived `MODEL_OPTIMIZATIONS.md` used the opposite convention — it calls the precision $A$ —
and where the two disagree the convention here wins. Stated once; not repeated below.

Other symbols: $n_{\rm det}$ detectors, $n_t$ = `n_analyze` samples, $k = n_{\rm quad}\,
n_{\rm mode}$ quadratures; $M_i$ the design matrix, $C_i = L_iL_i^\top$ the noise covariance,
$y_i$ the strain; $W_i = L_i^{-1}M_i$, $z_i = L_i^{-1}y_i$ whitened;
$R = \operatorname{chol}(A^{-1})$ lower. **The production point** is
$(n_{\rm det}, n_t, n_{\rm mode}) = (2, 205, 2)$, so $k = 8$ ($n_t$ ranges 102–410), with NUTS,
`dense_mass=True`, 4 chains × (1000 + 1000).

### 0.3 Corrections applied relative to the archived originals

| # | Superseded claim | Correction |
|---|---|---|
| 1 | `MODEL_OPTIMIZATIONS.md` §R1.5: "XLA *could* fold" the constant `solve_triangular(L, y)`. | **Refuted.** On XLA:CPU these lower to `lapack_dtrsm_ffi` custom calls, opaque to the constant folder; $z_i$ is genuinely recomputed every call. The *conclusion* — do not hoist, do not change the model signature — survives for a different reason (§4.4). |
| 2 | Unqualified equivalence figure "1.2e-15" including gradients. | Holds only for a benign test covariance ($\operatorname{cond}(C)\simeq7.6\times10^2$). Under a realistic aLIGO-like PSD ($\operatorname{cond}(C)\simeq10^{10}$) the gradients agree to $\sim10^{-12}$. Shared roundoff, not bias; test tolerances follow the realistic number (§5.1). |
| 3 | `GPU_BENCHMARKS.md` §2.3: the `[M\|y]` concat penalty is an **odd**-RHS cuBLAS `DTRSM` cliff, 5.2×. | **Misdiagnosed.** It is a **threshold at RHS $\ge 17$** with exact powers of two as fast outliers, not a parity effect; ~3.5× on the A6000, 1.5–1.9× on the H100; and **float64-only** (§4.3). |
| 4 | `GPU_BENCHMARKS.md` §4.2: `chain_method='vectorized'` is "worth 14–23×". | Those are throughputs at **64 chains relative to one chain** — parallel scaling, not the value of the switch, which is 3–4× at 16 chains on the A6000 (§3.5). |
| 5 | `GPU_BENCHMARKS.md` §3.2: float32 NaNs because each `cho_solve` is "an opportunity to lose bits". | **The mechanism is dynamic range, not conditioning or accumulated rounding**: the sequential form builds $C_i^{-1}M_i$ and $C_i^{-1}r_i$ explicitly, and those overflow float32 even at $\operatorname{cond}(A^{-1})\approx14$ (§5.3). |
| 6 | `GPU_BENCHMARKS.md` §2 / §7: R2 "flips on GPU" — `vmap` the detectors on GPU. | **Superseded, unconditionally.** The A6000 f64 result is real but does not generalize: it vanishes in float32 (unrolled wins 1.15–1.97×) and does not reproduce on the H100 (`R1_vmap / R1_unroll_sep = 0.94`). **Ship one always-unrolled path** (§4.1). |
| 7 | Notation clash: `MODEL_OPTIMIZATIONS.md` calls the precision $A$. | Resolved in favor of the methods note / code convention; see §0.2. |
| 8 | `R1_PR_PLAN.md` follow-up 1: "possibly defaulting `chain_method='vectorized'` on GPU". | **Done.** Landed on `main` as `37baf98` (#163) in `get_sampling_kwargs`: defaults to `'vectorized'` on an accelerator whenever `local_device_count() < num_chains`; explicit settings still win; CPU untouched (§7.6). *Now present in this tree via the fast-forward.* |
| 9 | `CLAIMS_VERIFICATION.md`: docstring defect in the `rd_design_matrix` matrix (`model.py:172-177` at `7bc480a`; row 2 disagreed with the code in sign and trig function). | **Fixed on `main`** by `c6399ae` (#162), and now present in this tree. Not an item for this PR. |
| 10 | `GPU_BENCHMARKS.md` §5: the device floor (67 µs) "is not silicon". | Partly wrong. The H100's dispatch-free floor is **26.5 µs** against the A6000's 66.4 µs — pre-registered falsification test F2 fired (§3.3). The broader latency-bound picture survives. |
| 11 | `MODEL_OPTIMIZATIONS.md` §B.2: variant speedups quoted as `ms/iteration`. | **Invalid as a cross-variant metric** — the forms differ at $\sim10^{-12}$, so NUTS chains decorrelate and take different numbers of leapfrog steps (6562 vs 8164). Normalize by `num_steps` or use device time. The 2.39× headline survives, corroborated per gradient (§3.6). |
| 12 | `R1_PR_PLAN.md` §6.2: move the notes to `dev/notes/` and `dev/h100_bench/`. | Superseded by the present layout: compiled note in `docs/dev/`, kit in `benchmarks/h100/`; the original reports were consolidated here and not retained. |

One pre-existing repo defect was folded into the PR and is now **fixed there**: the comment
`# (note that |A| = -|A_inv|)` at `ringdown/model.py:887` (at `7bc480a`) now reads
`log|A| = -log|A_inv|` (merged tree, `model.py:803`). The code itself was always correct.

---

## 1. Overview and verdict

The marginalized branch **used to** integrate out the quadrature amplitudes **one detector at a
time** (`ringdown/model.py:789-906` at `7bc480a`), using the posterior after detector $i$ as the
prior for detector $i+1$ and emitting one `numpyro.factor(f"logl_{i}", …)` per detector. That
sum telescopes *exactly* onto a single closed-form expression built from two accumulators and
one Cholesky, which is what the code does now (`ringdown/model.py:822-862`).

> **Verdict: implement R1 — the one-shot whitened marginal likelihood — as a single,
> always-unrolled, two-separate-solves code path. Do not branch on backend or dtype.**

In one paragraph: it is an algebraic identity, independently re-derived and verified to roundoff
on the potential energy, every gradient component, the predictive draw and all twelve derived
physical quantities; it cuts LAPACK triangular solves in the compiled gradient from 42 to 8–10
($n_{\rm det}=2$) and Choleskys from $n_{\rm det}$ to 1; it is worth ~2.4× per gradient on CPU
float64 and ~2.8× on an H100; it reduces compile time (19% CPU, 3–4× GPU); and it is strictly
more robust in float32, where the current formulation returns NaN and the one-shot form does
not.

Ranked outcome of the whole investigation (everything below is measured):

| # | Item | Verdict |
|---|---|---|
| **R1** | One-shot whitened marginal likelihood | **Implement.** 2.0–2.9× per gradient CPU f64, 2.81× H100 f64; 42→8 `dtrsm`, $n_{\rm det}$→1 `potrf`; compile −19% CPU, −70% GPU |
| **R2** | Detector loop: unrolled vs `vmap` vs `scan` | **Always unrolled, one code path.** `vmap` wins only GPU+f64 on the A6000; `scan` never wins |
| **R2b** | `[M\|y]` concatenated solve | **Do not use it.** Two separate solves; avoids an f64 cuBLAS threshold at RHS ≥ 17 |
| **Λ placement** | `a_scale` into the prior covariance ($\Lambda = S^2$) | **Reject** — same value, strictly worse conditioning. Applying $S$ to the $k\times k$ Gram instead is free and bit-identical; fold in if convenient, do not commit alone |
| **Prewhitening** | Pass $z_i$, $Q$, $\sum\log[L_i]_{tt}$ as model arguments | **Reject** — 0% end-to-end, and it risks `get_arviz`'s positional `sampler._args` |
| **R3/R4/R5/R6** | Reshape the `nmodes` scatter loop; 2-block `aligned` design matrix; batch the non-marginalized MVN sites; hoist `KerrMode` coefficients | Cleanup/hygiene only, all negligible; separate PRs |
| **float32** | Precision change | Out of scope. R1 is a *prerequisite* for it, not an alternative |

---

## 2. The R1 reformulation

The full derivation is in `docs/marginalized_likelihood.md` §§4–6. This is what a reviewer
needs.

**What the old code did** (`model.py:789-906` at `7bc480a` — this is the formulation the whole
equivalence argument is *against*, so it is quoted rather than summarized away). Per detector:
`A_inv = Lambda_inv + M.T @ cho_solve((L,True), M)` (`:818`); `cholesky(A_inv)` (`:821`);
`a = cho_solve(...)` (`:828`); `b = M @ mu` (`:841`); `r = y - b` (`:862`);
`Cinv_r = cho_solve((L,True), r)` (`:863`); a Woodbury correction (`:865`, `:872`); a three-term
`log_sqrt_det_B` (`:888`); and `numpyro.factor(f"logl_{i}", logl)` (`:901`), carrying `mu`,
`Lambda_inv`, `Lambda_inv_chol` forward (`:904-906`). That is four separate `cho_solve` calls
against $L$ — eight $[n_t,\cdot]$ triangular solves per detector — plus a serial dependency
chain through the running state, so detectors cannot overlap.

**The one-shot form** (now `ringdown/model.py:822-862`). Let $W_i = L_i^{-1}M_i$,
$z_i = L_i^{-1}y_i$ (single lower-triangular
solves). Because $C_i = L_iL_i^\top$, the identities $M_i^\top C_i^{-1}M_i = W_i^\top W_i$,
$M_i^\top C_i^{-1}y_i = W_i^\top z_i$ and $y_i^\top C_i^{-1}y_i = z_i^\top z_i$ are exact. With

$$A^{-1} = \mathbb{1} + \sum_i W_i^\top W_i,\quad v = \sum_i W_i^\top z_i,\quad
Q = \sum_i \lVert z_i\rVert^2,\quad R = \operatorname{chol}(A^{-1}),\quad u = R^{-1}v,$$

the total log-likelihood is

$$\sum_i \ell_i \;=\; -\tfrac12 Q + \tfrac12\lVert u\rVert^2
- \sum_i\sum_t \log [L_i]_{tt} - \sum_j \log R_{jj}.$$

Two independent proofs exist and both were checked line by line: a probabilistic one (each
$\ell_i$ is exactly $\log p(y_i\mid y_{<i})$ with $-\tfrac12 n_i\log2\pi$ dropped, so the chain
rule gives the joint) and an inductive one (the recursion's running state satisfies
$[\Lambda^{(i)}]^{-1} = \mathbb{1} + \sum_{j\le i}M_j^\top C_j^{-1}M_j$ and
$[\Lambda^{(i)}]^{-1}\mu^{(i)} = v_i$, so the final $[A^{(n_{\rm det})}]^{-1}$ *is* the one-shot
$A^{-1}$ as a matrix, not merely in determinant). The determinants telescope because
$R^{(0)} = \mathbb{1}$. **The one-shot form drops exactly the same constant.** It also
eliminates `mu`, `b = M @ mu` and `r = y - b`: the residual chain is an artifact of the
sequential formulation. And $A^{-1}$ is *bit-exactly* symmetric — $A^{-1}_{ij}$ and
$A^{-1}_{ji}$ are the same reduction computed by the same `dot_general` — while `potrf` reads
one triangle, so no symmetrization is needed.

**The predictive draw.** The old predictive block consumed `mu` and `Lambda_inv_chol` from the
loop's final state (`model.py:947` at `7bc480a`). In the one-shot form $\mu = A v = R^{-\top}u$
and $\Lambda^{-1/2}_{\rm chol} = R$, so `quads = mu + solve(Lambda_inv_chol.T, ξ)` becomes
`quads = solve(R.T, u + ξ)` — one back-substitution instead of two; as merged this is
`ringdown/model.py:905`. This was checked to be
**pointwise** identical given the same $\xi$ (to $1.8\times10^{-16}$), not merely equal in
distribution: mean $= Av$ to $8.6\times10^{-17}$, covariance $= A$ to $1.9\times10^{-16}$, and
all twelve derived quantities (`a, acx, acy, apx, apy, ellip, h_det, h_det_mode, phi, phi_l,
phi_r, theta`) agree to $\le 2.3\times10^{-12}$ in the real model at a fixed PRNG key.
**This must change in the same commit as the loop:** if the loop changes and this does not, the
log-likelihood stays correct while posterior-predictive draws are silently wrong.

**Structural evidence** — the most robust in the study, because it is load-independent and was
reproduced exactly by an independent harness. Post-optimization HLO of
`jit(grad(potential_energy))`:

| | `dtrsm` $n_{\rm det}=2$ | $n_{\rm det}=3$ | `potrf` | HLO lines (2,205,2) |
|---|---|---|---|---|
| current | **42** | 66 | $n_{\rm det}$ | 1891–1924 |
| R1, unrolled, two solves | **10** | 13 | **1** | 1239 |
| R1, unrolled, `[M\|y]` | **8** | 10 | **1** | 1337 |
| R1, `vmap` | 7 | 7 | **1** | 1331 |

Identical counts appear on GPU with `__cublas$triangularSolve` / `cusolver_potrf_ffi`, so the
op-count argument transfers verbatim across backends. FLOP accounting per detector: current
$2n_t^2k + 6n_t^2 + n_tk^2$, R1 $n_t^2k + n_tk^2$ — a ratio of **2.68×** at the production
point, matching the measured 2.39–2.8×. The $6n_t^2$ term (the three vector `cho_solve`s) is
27% of the current cost at $k=8$; R1 removes it entirely.

---

## 3. The performance case

All three platforms agree on direction and roughly on magnitude. Ratios are load-bearing;
absolute numbers are machine- and session-dependent.

### 3.1 CPU (32-core shared Flatiron workstation, float64)

Isolated `jit(grad(potential_energy))` at (2,205,2) / (3,205,3) / (2,410,2), µs/call, median of
interleaved blocks: current **1974 / 2215 / 2779**; whitened-sequential (keeps `logl_i`) 1038 /
876 / 1662; **R1 one-shot unrolled 442 (4.46×) / 794 (2.79×) / 950 (2.92×)**; R1 `vmap` 958 /
1545 / 2029; no-likelihood floor 180 / 270 / 267.

End-to-end `MCMC.run`, 1 chain, `dense_mass=True`, (2,205,2): the current model runs 9.89 s at
1000+1000 (~1.72 s compile+setup, 4.085 ms/iteration); R1 with no API change runs **4.82 s**
(~1.40 s, **1.712 ms/iteration**). **Headline: 2.39× reduction in NUTS sampling time, 2.05× in
total wall clock, 19% less compile time**, at the production point; the consolidated honest
range across all end-to-end measurements is **2.0–2.9×**. Independently reproduced: raw
wall-clock ratio 2.38× (42.17 s → 17.71 s on an informative posterior), per leapfrog gradient
2.54× (two solves) / 2.96× (concat), kernel-level isolated ratio 4.48× vs the reported 4.46×.

**Where the remaining time goes.** Against a no-likelihood floor model (same priors, same
`rd_design_matrix`, trivial factor), normalized by the true NUTS gradient count, the likelihood
is still **66% / 71% / 72%** of the per-gradient cost after R1 at those three sizes — 1169,
2024, 1973 µs/grad against floors of 392, 581, 546. So R1 is the last large single win
available on the likelihood; the residue is the irreducible $n_t^2k$ whitening solve plus its
adjoint, and the ~30% that is `rd_design_matrix`, prior log-probs and NumPyro's per-site trace
machinery. Measure anything further against this floor first.

**Threading.** Production sets `OMP_NUM_THREADS=1` (`ringdown/cli/ringdown_fit.py:80-82`), which
costs the CPU 20–85% and compresses R1's CPU speedup from ~4.3× to ~2.8× at kernel level — but
changes no conclusion.

### 3.2 RTX A6000 (float64 device time)

Dispatch-free `fori_loop` harness; per-gradient device µs, GPU rows (at the production point the
CPU comparison is 769 current / **274** R1 unrolled / 748 R1 `vmap`):

| config | current | R1 unrolled | R1 `vmap` | best R1 |
|---|---|---|---|---|
| **(2, 205, 2)** *production* | 3072 | 1187 (2.59×) | **913 (3.37×)** | 3.37× |
| (2, 205, 8) $k{=}32$ | 3634 | 1482 (2.45×) | **1100 (3.30×)** | 3.30× |
| (2, 1024, 2) | 15847 | 6696 (2.37×) | **4457 (3.56×)** | 3.56× |
| (3, 1024, 8) $k{=}32$ | 25093 | 10466 (2.40×) | **5931 (4.23×)** | 4.23× |

R1 wins everywhere on the A6000 and never degenerates to "no win", unlike CPU at $k=32$.
Correctness was re-verified on GPU before any timing: worst deviation over 9 configurations
$5.9\times10^{-16}$. Two A6000-specific findings, both float64-only and both since qualified:
the `vmap` advantage (§4.1) and the concat penalty (§4.3).

**The A6000 loses the production configuration to the workstation CPU under every
formulation** — 4.0× slower per gradient for the current model, 3.3× best-to-best (the
independent check measured 2.77× and 4.00×, transposed but substantively identical). It starts
winning as $k$ grows and as $n_{\rm det}$ grows, not simply as $n_t$ grows.

### 3.3 H100 PCIe (`benchmarks/h100/`, Slurm job 6969321, same-node CPU baseline)

This run answers one question: is "the CPU beats the GPU at production size" a property of the
A6000 or of the model? Predictions were registered in advance in
`benchmarks/h100/PREDICTIONS.md` with numeric intervals and falsification criteria.
Per-gradient device µs at the production point, float64:

| variant | H100 | A6000 | A6000/H100 | same-node CPU | GPU/CPU |
|---|---|---|---|---|---|
| current | 1340.5 | 3072.0 | 2.29× | 603.6 | 2.22× slower |
| whitened-sequential | 592.0 | 1412.0 | 2.39× | 228.0 | 2.60× |
| **R1 unrolled, concat** | **463.8** | 1186.8 | 2.56× | **197.2** | 2.35× |
| **R1 unrolled, two solves** | **477.3** | 1206.6 | 2.53× | 228.2 | 2.09× |
| R1 `vmap` | 506.2 | 912.9 | 1.80× | 255.7 | 1.98× |
| floor (no likelihood) | 26.5 | 66.4 | 2.51× | 51.4 | — |

Best-config-to-best-config, same node: GPU f64 / CPU f64 = **2.35×**, GPU f32 / CPU f64 =
**1.73× — the GPU is slower either way** (1.16× against the production `OMP=1` CPU). At
(2,1024,4) the GPU is **2.28× faster**; at (3,1024,8), **4.41× faster**. R1's ratio is
preserved (`R1_unroll_sep / current` = **2.81×**), and `R1_vmap / R1_unroll_sep` = **0.943** —
`vmap` does **not** win here (correction 6). float32 buys only 1.09–1.40× at the production
point and ~1.00× at (3,1024,8), against 2.2–3.4× on the A6000, as expected from the H100's 1:2
rather than 1:32 FP64:FP32 ratio. Correctness: all variants within $2.6\times10^{-16}$.

**Scorecard: 12 HIT / 23 MISS against the pre-registered 80% intervals (34%).** Almost every
miss is in the same direction — the H100 is *faster than predicted*, typically by 1.1–1.9× —
i.e. the intervals were anchored too closely to the A6000. One falsification test fired: **F2**,
the claim that the dispatch-free launch floor is not silicon (correction 10). The substantive
predictions all held: P5 (CPU wins at production size), P6 (GPU wins for large analyses), P11 (GPU f32
vs CPU f64) and P11b (chain crossover) are all HITs. Net: a better card narrows the
production-size gap from 3.3–4.0× to 1.7–2.4× but does not close it; it confirms the GPU as the
right device for large analyses; and it settles the `vmap` question against branching.

### 3.4 Compile time

Isolated `jit(grad)` at (2,205,2) is 0.36–0.54 s on CPU, 0.46–1.07 s on the A6000 and
0.43–0.54 s on the H100, essentially independent of variant. Full `MCMC.run` compile+setup goes
from ~1.72 s to ~1.40 s on CPU (**−19%**) and from ~30.6 s to ~8.2–9.3 s on the A6000
(**3.3–3.7×**, a much larger relative win the CPU report could not see; the H100 needs
~3.05 s). Compiler work scales with op count, and R1 removes 34 of 42 `triangularSolve` custom
calls and $n_{\rm det}-1$ Choleskys.

There is **no unrolling problem to solve**: at $n_{\rm det}\in\{2,3\}$ the unrolled loop is 2–3
copies of a small block and compile times are within noise of every alternative. The only
construct whose compiled size grows with a model dimension is the `nmodes` scatter loop in
`get_quad_derived_quantities` (HLO 114/147/213 lines for `nmodes` 2/3/5), which R3 makes
constant at 83.

### 3.5 Chains

On the A6000, vectorizing is worth **3–4× wall clock at 16 chains** (not the 14–23× that
appeared as a headline — correction 4). R1 compounds with it: R1 + vectorized is 10.4× faster
than the current model run the way the shipped defaults would have run it. Chain throughput
relative to one chain, production point, 250+250, R1 `vmap`: **3.39× / 8.68× / 22.68×** on the
A6000 and **2.66× / 7.13× / 20.98×** on the H100 f64, at 4 / 16 / 64 chains. Crossover against
the same node's best 4-chain CPU run (2.66 ms/chain-iteration): the H100 needs **≥ 16
vectorized chains**, the A6000 16–64. Inner detector batching and outer chain batching
**compose** — the `vmap`/unrolled ratio is flat to within noise across a 64× change in chain
count — so the inner decision needs no chain-count axis.

### 3.6 Benchmarking pitfalls, recorded because they bit

**`ms/iteration` is not a valid cross-variant metric.** The formulations differ at
$\sim10^{-12}$, so NUTS trajectories decorrelate and the variants take materially different
numbers of leapfrog steps for the same seed (6562 vs 8164), and warmup adapts to different step
sizes. Normalize by `extra_fields=('num_steps',)`, use raw wall clock on an informative
posterior, or — best — device time per gradient; the affine compile/slope decomposition is
exposed to the same hazard and returned a *negative* compile intercept in independent hands.
**Per-call Python dispatch dominates the fast variants** in isolated microbenchmarks
(~150–500 µs/call): a full-`potential_energy` harness showed 1.78× where the kernel ratio was
4.48×, so use the dispatch-free `fori_loop` harness in `benchmarks/h100/bench.py`. Two smaller
traps: setting `JAX_PLATFORMS` *after* `import jax` silently does nothing (one "CPU" run in the
review was in fact a GPU run), and `compiled.cost_analysis()['flops']` under-reports by ~10×
because the solves are FFI custom calls invisible to it.

### 3.7 End-to-end confirmation on a real fit (post-implementation)

Everything above is a microbenchmark or a device-time measurement. After R1 landed in PR #164 it was A/B'd on the real `tests/test_fit_config.sh` fit — GW150914, 4 chains × (1000 + 1000),
CPU float64 — which is the closest thing to a production run the repo has. Details in PR #164's
*End-to-end validation* section.

| metric | before | after | ratio |
|---|---|---|---|
| **µs per leapfrog** (the fair metric) | 478 | 229 | **2.09×** |
| wall clock | 57.8 s | 37.1 s | 1.56× |
| total CPU time | — | — | 1.95× |
| ms per effective sample | — | — | 1.77× |
| warmup + compile / sampling / predictive+arviz | — | — | 2.0× / 2.2× / 1.27× |

Posteriors agree within about one run-to-run standard error. The wall-clock figure (1.56×) is
the smallest because ~8.3 s of the run is fixed CLI overhead — imports, data fetch, ACF
estimation — that R1 does not touch, and because the predictive/arviz phase only gains 1.27×.
**Per leapfrog the fit gets 2.09×, squarely inside the 2.0–2.9× predicted in §3.1**, which is
the number to quote when comparing against the microbenchmarks. Note also that the wall-clock
comparison is legitimate here only because it is a fixed-workload A/B on one configuration;
§3.6 still applies to cross-variant `ms/iteration` claims.

---

## 4. What was rejected, and why

### 4.1 `vmap` and `lax.scan` over detectors — always unroll

The one-shot form is `vmap`-able (the sequential recursion is not: it carries $(\mu_i,
\Lambda_i)$ across detectors). Whether to use that is the most-investigated question in the
study, and the answer is **no**, unconditionally.

| configuration | winner | margin |
|---|---|---|
| **CPU + float64** (production) | **unrolled** | **2.73×** |
| **GPU + float32** | **unrolled** | **1.15–1.97×** |
| **H100 + float64** | **unrolled** (tie) | 1.06× (`vmap`/unrolled = 0.94) |
| A6000 + float64, $n_t < 2048$ | `vmap` | 1.3–1.9× |
| A6000 + float64, $n_t \ge 2048$ | **unrolled** | 1.9–3.6× |

`vmap` wins one cell of the matrix, on one card, in a precision that card is bad at, at sizes
below a vendor kernel threshold. Meanwhile **a wrong static default is not benign**: on CPU at
the production point `vmap` measures 747.5 µs/gradient against the *current unoptimized* code's
768.7 — throwing away essentially the entire R1 benefit (2.81× → 1.03×) — and end-to-end
6.921 ms/iteration against the current model's 5.760, i.e. a net regression. The asymmetry is
the whole argument: wrong-on-CPU forfeits 2.7× on the platform that runs production;
wrong-on-GPU-f64 forfeits 1.3–1.9× on a platform that is losing anyway.

An auto-dispatch rule would need four conjuncts —
`backend == "gpu" and dtype == float64 and n_det > 1 and n_analyze < 2048` — one of which
hard-codes a cuBLAS kernel threshold, and it would silently change generated HLO from
process-global state. Every candidate mechanism fails somewhere: reading
`jax.default_backend()` in the model body works but is invisible in the trace; reading it in
`make_model` is ordering-fragile; the input arrays' device is unavailable (they are tracers
under `MCMC`); NumPyro exposes no hook. (That $n_t = 2048$ term is not a crossing but a
**cliff** — from 1920 to 2048 the unrolled path gets 2.3× *faster on a 6.7% larger problem* —
and the float32 sweep is smooth, so it is a pure `DTRSM` kernel-selection artifact.)
**`lax.scan` is dead** too, now measured rather than argued: 2.8–9.4× slower than unrolled on
CPU and worsening with size (XLA:CPU cannot fuse across the loop-carried dependence),
1.03–1.18× slower on GPU, with *no* compile-time benefit.

**Residual risk of always-unrolled:** if the project later runs float64 GPU production at
$n_t < 2048$ on Ampere-class hardware it leaves ~1.5× on the table — recoverable at any time,
since the two forms differ by ~5 lines and both are verified. Do not add the flag
speculatively.

### 4.2 The amplitude-scale placement

Write $M_i = M_i^0 S$ with $S = \operatorname{diag}(s)$, $s$ = `a_scale` tiled over quadrature
blocks, and $G = \sum_i W_{0i}^\top W_{0i}$. Three variants, all exactly the same function of
$\theta$: **(a)** status quo, $\Lambda = \mathbb{1}$ with $S$ applied to the $(n_t,k)$ design
matrix, $A^{-1} = \mathbb{1} + SGS$ — fine; **(b)** $\Lambda = \mathbb{1}$ with $S$ applied to
the $k\times k$ Gram instead — a free tidy-up; **(c)** $\Lambda = S^2$ with an unscaled design
matrix, $A^{-1} = S^{-2} + G$ — **reject**.

(b) is a pure reassociation — $S$ is diagonal and acts on columns while $L_i^{-1}$ acts on rows
— moving $O(n_tk)$ multiplies to $O(k^2)$ (1640 → 64 per detector at the production point). It
is bit-identical (0 ulp at all 18 sweep points), worth 1.05–1.16× in the isolated gradient and
**not resolvable end-to-end**. Two lines, no downstream consequences; not worth its own commit.

(c) is a genuine change of integration variable, and the two precisions are related by a
diagonal congruence $A^{-1}_{(c)} = S^{-1}A^{-1}_{(a)}S^{-1}$, so
$\operatorname{cond}(A^{-1}_{(c)}) \approx \operatorname{cond}(S)^2\operatorname{cond}
(A^{-1}_{(a)})$. Since `a_scale` $\sim\mathrm{Uniform}(0,\,$`a_scale_max`$)$ and NUTS can drive
it arbitrarily close to zero, that factor is unbounded in a region the sampler genuinely
visits: over four seeds of a 500+500 run the sampling-phase minimum of `a_scale/a_scale_max`
was $1.2$–$8.8\times10^{-4}$, at which $\operatorname{cond}(A^{-1}_{(c)})\approx7\times10^{7}$
— 8 of float64's 16 digits gone, in the sampling phase alone, before warmup. Gradients become
non-finite where (a) is exact, and (c) also loses **eleven digits** of gradient accuracy if
`a_scale_max` is merely mis-specified by $10^6$, which (a)/(b) absorb without noticing because
$A^{-1}\to\mathbb{1}$ as $s\to0$. No upside, real downside.

The **AD hypothesis** motivating (b)/(c) — that they lighten the backward pass through the big
triangular solve — is **not confirmed**: the reverse pass must compute the $W$ cotangent anyway
for the $m$/$\chi$ derivatives. The HLO is decisive — the number of instructions referencing
`f64[205,*]` arrays is identical (140 / 209) across all three variants, as is the `dtrsm` count.

### 4.3 The `[M | y]` concatenation

Solving once against the concatenated $[M_i \mid y_i]$, an $n_t\times(k+1)$ right-hand side, is
algebraically identical and was the form originally recommended. **It has been retired.** On CPU
it saves exactly one LAPACK dispatch and no arithmetic. On GPU in float64, cuBLAS switches to a
markedly slower `DTRSM` kernel once the right-hand side has $\gtrsim17$ columns — **a threshold,
not an odd/even effect** (correction 3): the A6000 sweep at $n_t=205$ gives RHS 16 → 131.6 µs,
17 → 462.1, **18 (even) → 439.7**, with 32 an anomalously fast outlier at 278.7 and 33 back to
461.4; the whole range 17–40 sits in the slow regime, and $k=64$ is fast too. On the H100 the
threshold is milder (largest step 16→17, ×1.49 at $n_t=205$, ×1.90 at 1024). In the full model
this costs 1.4–2.0× at $n_{\rm mode}\ge4$ ($k+1 = 17, 33$) and nothing at the production $k=8$.
**And the effect is float64-only** — in float32 there is no threshold and the concat is
marginally *faster* (372 vs 413 µs at (2,205,4)). Net rule: **two separate solves** — downside
bounded at ~10%, upside reaching 2×.

### 4.4 Prewhitening / hoisting the constants

$z_i$, $Q$ and $\sum_i\sum_t\log[L_i]_{tt}$ are constants: $L_i$ and $y_i$ are model arguments,
and with `jit_model_args=False` (never overridden anywhere in the repo) they are baked into the
executable as XLA literals — `f64[205,205] … constant` is visible in the optimized module.
**XLA does not fold them** (correction 1): on XLA:CPU `solve_triangular` lowers to a
`lapack_dtrsm_ffi` custom call, opaque to the constant folder, so $z_i$ is genuinely recomputed
on every call (the separate-solve form keeps 5 forward `dtrsm` against 3 when $z$ is hoisted).

The recommendation to **not** hoist nevertheless stands, on measurement: full `MCMC.run` with
the hoisted variant gives **1.716 vs 1.712 ms/iteration — 0% gain**; at kernel level hoisting is
worth 1.08–1.24×, a fraction of a percent end-to-end; the isolated microbenchmark's apparent
1.35× was per-call Python dispatch; and across contended repeats the hoisted variant lands
anywhere between 1.8× and 3.9×, i.e. it is *not reliably better*. Not hoisting also means
`ringdown/fit.py:335` (at `7bc480a`; now `:351`) `run_input` is untouched and `get_arviz`, which reads `sampler._args`
**positionally** (`model.py:1093`, `1104`, `1126`; `:1135`, `:1146`, `:1168` at `7bc480a`),
cannot break — worth more than 0%.

### 4.5 Everything else that was investigated and discarded

| Hypothesis | Verdict |
|---|---|
| `numpyro.deterministic` sites evaluated every leapfrog step; move to `postprocess_fn` | **Wrong.** `potential_energy` discards the model trace, so they are dead code and XLA DCEs them: `dynamic-update-slice` count = 0 with `store_h_det_mode=True`, HLO size unchanged (2258 vs 2277 lines). `Fit.run` also leaves `predictive=False` during MCMC, so the block is not reached. |
| Exploit symmetry of $A^{-1}$; explicitly symmetrize it | Rejected twice over: it is already bit-exactly symmetric and `potrf` reads one triangle; and a perfect `syrk` would save 6.6 kFLOP against $n_t^2k = 336$ kFLOP — under 2%. |
| `MultivariateNormal.log_prob` recomputes $\log\lvert L\rvert$ each step (non-marginalized path) | **XLA already removes it:** zero `log` ops on any `f64[205]` array, because `scale_tril` is a compile-time constant. The mahalanobis term is already one triangular solve (4 `dtrsm` for $n_{\rm det}=2$, the minimum). Nothing to gain beyond R5's site batching. |
| Fold $2\pi$ into $f$; single complex `exp` instead of `exp`+`cos`+`sin` | Already optimal / no gain — and moot: the entire design matrix costs **< 2 µs/gradient** (68.6 vs 66.8 µs with and without, on GPU), fully fused. |
| `concatenate` / layout / column-ordering copies | XLA fuses the concatenate into the producer and the downstream `dtrsm` needs a contiguous array regardless. Not measurable at $k\le12$. |
| Merge the `apx/apy/acx/acy_unit` sites | With `marginalized=True, predictive=False` those sites **do not exist** — only `m`, `chi`, `a_scale` are sampled. Merging would break `MODEL_VARIABLES_BY_MODE` and every downstream name for no gain. |
| `KerrMode.coefficients` expensive per trace | Memoized: 899 ms once per process, **6.1 µs** warm; the body is traced 5× per run. Hoisting into the `make_model` closure is tidier (0.2% of compile+setup), not a performance fix. Same for `chi_factors` (11 scalars, CSE'd) and `map(jnp.array, …)` (~6 ms total). |
| `ComposeTransform.log_abs_det_jacobian` recomputes the forward pass | **Real** but negligible: $n_{\rm mode}\le3$ scalars, almost certainly CSE'd, and only on the non-production free-damped-sinusoid path. One-line fix via `call_with_intermediates`. |
| dtype promotion / spurious `convert` ops | Non-issue. With x64 on the compiled grad HLO contains **zero** f32 operations; the only `convert`s are 5 on `f64[8,8]` from `jnp.eye(k)`. |

### 4.6 Out of scope (changes geometry, precision or parameterization)

**float32** (`etc/ringdown_pipe_example_imr.ini` already sets it, and the CLI wires it up) is a
numerics tradeoff, not a free win (§5.3), and would need a posterior-level validation, not a
log-density comparison. **Reparameterizing the amplitude scale or the quadratures** was out of
scope by instruction. **`TransformedDistribution` instead of `ImproperUniform` + explicit
`numpyro.factor`** on the ordered-frequency path is plausibly measure-preserving but changes
which space NUTS explores and how the mass matrix adapts — not without a dedicated study.

---

## 5. Numerical robustness and precision

### 5.1 float64 equivalence, and the right tolerance

Two independent prototypes — one transcribed from the `[M|y]` concat sketch, one written from
scratch from an independent re-derivation using separate solves — were compared against the
unmodified `ringdown.model.make_model` via `initialize_model` + `potential_energy`, at 6–8
random unconstrained points per configuration, over 12 configurations including exactly
degenerate modes. They agreed with each other to the last digit in every case.

| test covariance | $\operatorname{cond}(C)$ | worst rel. $\Delta U$ | worst rel. $\Delta\nabla U$ |
|---|---|---|---|
| identity | 1.0 | 2.6e-16 | 1.7e-15 |
| smooth exponential ACF (the original harness) | 7.6e2 | ~1e-15 | **1.2e-15** |
| aLIGO-design-like PSD (realistic) | 9.8e9 | ~1e-15 | **~6e-12** (norm-relative 3.4e-12) |
| aLIGO PSD + exactly degenerate modes | — | ~1e-15 | 1.6e-9 (norm-relative) |

**The headline 1.2e-15 is optimistic by ~3 orders of magnitude for production-like data**
(correction 2). This does not weaken the correctness case — $10^{-12}$ relative on a gradient is
far below anything HMC can resolve, and a central finite-difference check cannot distinguish
which formulation is more accurate, so it is shared roundoff amplified by
$\operatorname{cond}(C)$, not bias. But it sets the test tolerance: **use 1e-11, not 1e-15**
(§7.4); the H100 harness uses the same bar. Note also that the R1 variants agree with *each
other* far more tightly than either agrees with the repo model (2.8e-15 vs 3.9e-15 in float64;
3.0e-6 vs 1.4e-3 in float32), since they are the same algorithm with a different loop structure.

**As landed** (`tests/test_model.py`, merged in `b6e30b9`), against a frozen transcription of
the sequential scheme, the numbers the suite actually pins are:

| case | agreement |
|---|---|
| $(n_{\rm det}, n_t, n_{\rm mode}) \in \{(1,205,1),(2,205,2),(3,205,3),(2,410,2)\}$, aLIGO-like $\operatorname{cond}(C)$ of a few $\times10^{9}$ | $\le 4\times10^{-13}$ relative on the potential energy and every gradient component |
| near-degenerate modes, **white** covariance | $\sim3\times10^{-15}$ |
| near-degenerate modes, aLIGO-like covariance | $\sim10^{-8}$, i.e. $\operatorname{cond}(A^{-1})\cdot\epsilon$ — shared roundoff, bounded by `eps * cond(C)` (~1e-5) rather than by the flat 1e-11 |
| predictive draw and all twelve derived quantities, fixed $\xi$ | pointwise equal to the same tolerance |

The white-covariance row is the useful control: it isolates the algebra from the conditioning
of $C$, and there the two forms agree essentially to machine precision even on the degenerate
configurations, which is what establishes that the $10^{-8}$ row is $C$'s conditioning and not a
defect in either formulation.

Separately, the $(\mathcal{A},\epsilon,\vartheta,\varphi)$ inverse relation of the methods note
§6.1 — the map from quadratures to the physical amplitude/ellipticity/phase parameterization —
was checked both symbolically and numerically against `Ringdown.complex_mode`
(`ringdown/waveforms/ringdown.py`), the implementation of Eq. (8) of arXiv:2107.05609:
agreement to $5\times10^{-15}$. That is a statement about the post-processing, not the
likelihood, and it is unaffected by R1.

### 5.2 Conditioning in float64: the two forms are equally good

$A^{-1} = \mathbb{1} + \sum_i W_i^\top W_i$ is the identity plus a positive semidefinite matrix,
so its eigenvalues are bounded below by 1: it is never singular however degenerate the design
matrices become. And by Lemma 1 the recursion's final precision *is* the one-shot $A^{-1}$, so
both schemes factorize the same matrix and inherit the same conditioning. A stress test with
deliberately duplicated modes, driving $\operatorname{cond}(A^{-1})$ from $3\times10^1$ to
$4.3\times10^{13}$, gives relative errors (sequential / one-shot) of 1.4e-16 / 9.5e-16 at
$3.0\mathrm{e}1$, 5.5e-14 / 4.9e-14 at $9.4\mathrm{e}4$, **5.6e-12 / 5.6e-12** at
$2.9\mathrm{e}7$ and **6.5e-06 / 6.5e-06** at $2.9\mathrm{e}13$. Identical in the
ill-conditioned regime: the error is the shared ill-conditioning of $A^{-1}$, not the
reformulation. **The sequential form is not better conditioned.**

### 5.3 float32: R1 is strictly better, and the mechanism is overflow

In single precision the two forms do **not** fail together. With realistic aLIGO-scale noise
($L\sim10^{-22}$, so $C^{-1}\sim10^{44}$) at $\operatorname{cond}(A^{-1}) = 13.9$, the
sequential form is finite only at the smallest amplitude scale and returns **NaN** from
`a_scale` $\ge 10^{-18}$ upward (and even in *float64* at $10^{-15}$), where the one-shot form
stays finite throughout; its float32 gradient relative error is 2.9e-1 against the one-shot's
**1.3e-3**. The H100 run reproduced the asymmetry independently: `current` returns NaN in three
of nine stress cases; **no R1 form ever lost float32**.

**The controlling variable is dynamic range, not conditioning** (correction 5). The sequential
form builds $C_i^{-1}M_i$ and $C_i^{-1}r_i$ explicitly; with $L_i\sim10^{-22}$ those
intermediates reach $10^{26}$–$10^{44}$ and overflow float32's $3.4\times10^{38}$ ceiling even
where $\operatorname{cond}(A^{-1})$ is of order ten and the final contractions are $O(1)$.
Whitening once keeps $W_i$ and $z_i$ of order unity, so no such intermediate is ever formed. The
earlier account — a threshold at $\operatorname{cond}(A)\gtrsim5\times10^6$ ascribed to
accumulated rounding across the `cho_solve` chain — is superseded by this more direct
measurement.

Two caveats keep it in proportion. Remaining finite is not remaining accurate: at high
conditioning the one-shot form's float32 gradient is already $O(1)$ in relative error, and at
$\operatorname{cond}(A^{-1})\sim10^{13}$ it too returns NaN. And **the float32 gradient is ~3
orders of magnitude less accurate than the float32 log density** — at realistic conditioning the
potential is good to 5e-7 while the gradient is only good to 5e-4; NUTS integrates the gradient,
so any float32 validation that compares log densities measures the wrong quantity.
**Conclusion: R1 is a prerequisite for any future float32 work, not an orthogonal change** —
a reason to do R1, not a reason to do float32.

---

## 6. Platform guidance

The user-facing version now lives in **`docs/configuration.rst`** (the `Performance` section),
which is where to send anyone asking "CPU or GPU?". In brief, only as an index into that page:
**a single fit at typical size** (a few hundred samples, 2 detectors, ≲3 modes) runs fastest on
a multi-core CPU in float64 — best-config-to-best-config on the same node, one GPU was ~2.4×
slower on an H100 and ~3.3× on an A6000, and float32 narrows the H100 gap to 1.7× without
reversing it. **Many chains:** CPU across host devices up to a few tens of chains, beyond that a
GPU with vectorized chains (crossover 4–16 chains on an H100, 16–64 on an A6000). **Large
analyses** ($n_t \gtrsim 1000$ *and* $\gtrsim4$ modes): GPU, 2.3–4.4× faster than the same-node
CPU. **Precision:** float32 is worth 2–3× on workstation Ampere (1:32 FP64) and only ~1.1× on
datacenter cards (1:2 FP64), so prefer float64 on an A100/H100, and never float32 on CPU; see
§5.3 for the NaN caveat, which `docs/configuration.rst` carries as a warning.
**`chain_method`:** as of `37baf98` on `main`, `get_sampling_kwargs` defaults to `'vectorized'`
on an accelerator whenever `local_device_count() < num_chains`, with explicit settings still
winning; before that fix NumPyro's `'parallel'` default silently drew four chains *sequentially*
on an idle card.

The physical reason: the cost is dominated by triangular solves on $n_t\times n_t$ matrices,
which are **block-sequential by construction**, so at $n_t = 205$ the card waits on a dependency
chain a few hundred elements long. An identically shaped dense-GEMM probe — same FLOPs, no
sequential dependency — runs 3.4–10.6× faster, and R1 `vmap` at the production point achieves
2.4 GFLOP/s, **0.2% of the A6000's FP64 peak**, with `nvidia-smi` reporting 0% utilization. The
algebra has been rewritten as far as it usefully goes; the fix for a GPU is more work per launch
— bigger problems, or more chains — not more rewriting.

---

## 7. The PR plan

### 7.0 Status: MERGED as PR #164 — read this before the rest of §7

This section is the **plan of record**: it is why each choice was made, and it is what the
implementation followed. It was **carried out and merged** — PR #164, squashed onto `main` as
**`b6e30b9`** on **2026-09-01**, from branch `r1-oneshot-marginalization` (six commits:
`7b5418a` tests, `ce5cf8a` model, `b233c93` docs+changelog, then `ee33bac`, `d682181`,
`e022f70` from Copilot review). It touched `ringdown/model.py`, `tests/test_model.py` (new),
`docs/marginalized_likelihood.md`, `docs/models.rst`, `CHANGES.md` and
`docs/examples/GW150914.ipynb`, and did **not** touch `docs/dev/` or `benchmarks/`.

This branch has been fast-forwarded onto `b6e30b9`; `JAX_PLATFORMS=cpu pytest tests/test_model.py`
is green here (15 passed).

Where the implementation deliberately departs from the text below, the **implementation wins**:

| Plan said | What shipped | Why |
|---|---|---|
| Flat **1e-11** relative for every equivalence case (§7.4a) | 1e-11 for the main configurations; the two deliberately worst-case cases — near-degenerate modes on an ill-conditioned covariance, and the single-detector case whose plus/cross columns are nearly parallel — assert an **`eps · cond(C)`** bound instead (measured ~2e-8 against a ~1e-5 bound) | A flat 1e-11 is right where it holds and would have forced loosening the *main* tolerance to accommodate two outliers. Bounding the outliers by the roundoff theory instead keeps the main gate tight. Same reasoning as §5.1, taken one step further. |
| NUTS smoke on the production Kerr model, gate **3 s.e.** (§7.4d) | Generic **damped-sinusoid** model (`f`, `g`, `a_scale`), gate **5 s.e.** (`d682181`) | The damped-sinusoid model is what the test's `_make_reference_model` builds, and it exercises the same likelihood block without the `qnms` table load. The gate was widened on review because the test is noisy — see the follow-up below. |
| `model.py:887` (at `7bc480a`) comment fix as a possible separate commit (§7.1 row 5, §7.7 item 3) | **Fixed inside `ce5cf8a`**; no separate commit | The surrounding block was rewritten wholesale, so the corrected statement was carried into the replacement comment. |
| Changelog as its own commit (§7.7 item 5) | **Folded into `b233c93`** with the methods note | One docs commit rather than two. |
| Wire the methods note into Sphinx: `myst_parser`, `dollarmath`, `index.rst` toctree (§7.6) | **Superseded — the user declined the wiring.** `docs/models.rst` instead gained a short standalone **"Amplitude marginalization"** section ending in a plain-text pointer to ``docs/marginalized_likelihood.md`` "in the source repository". `docs/conf.py` and `docs/index.rst` are untouched. | See §7.6. |

Two review-driven clarifications worth keeping: `ee33bac` corrected a test comment that had
said the *posterior* becomes degenerate — it does not; $A^{-1} = \mathbb{1} + \sum_i W_i^\top
W_i$ keeps the posterior regular (§5.2), and the degeneracy is in the **likelihood**.
`e022f70` cleared stale output from `docs/examples/GW150914.ipynb`.

**Known issues opened by the implementation** — none blocking, all candidates for follow-up
PRs. **Re-checked against the merged tree on 2026-09-01: the merge resolved none of them**, and
all four reproduce exactly as recorded (`--remove-on-error` is still at
`tests/download_example_data.sh:21`; the note's macro counts are unchanged at
`\operatorname` ×12, `\mathbb` ×56, `\tag` ×16, `\boxed` ×11).

1. **The seeded NUTS smoke test is nearly vacuous.** At 300 + 300 with one chain, ESS comes out
   around 3 of 300 draws, so the 5-s.e. gate can barely fail. It is worth keeping as a
   "does the sampler still run" canary, but **the deterministic potential-energy and gradient
   comparison is the real equivalence check** — do not read the smoke test as posterior
   validation. Either lengthen it substantially or relabel it.
2. **`tests/download_example_data.sh:21` uses `curl --remove-on-error`**, which needs curl ≥ 7.83
   and is unavailable on these workstations, so the shell/integration test (§7.4f) fails on
   setup rather than on anything ringdown does.
3. **The methods note does not render cleanly on GitHub** — 581 math spans, including
   `\operatorname` ×12, `\mathbb` ×56, `\tag` ×16 and `\boxed` ×11, which GitHub's math
   support handles poorly. With the Sphinx wiring declined, **GitHub is now the note's only
   rendered surface**, so this is worth a pass over the macros.
4. **The `ringdown_fit` entry point always imports the venv's editable install** — i.e. this
   primary checkout — regardless of the working directory. Anyone A/B-testing a worktree must
   account for that or they will benchmark the wrong code.

**The single source of truth for the mathematics is `docs/marginalized_likelihood.md`** — §4
(telescoping proof), §5 (whitened final form, eq. 5.1), §6 (predictive draw), §7.4 (the
before/after code sketch, whose "After" block is what shipped).

### 7.1 Scope

| # | In scope | Location |
|---|---|---|
| 1 | Replace the sequential detector recursion with the one-shot whitened form: **unrolled** `for i in range(n_det)`, **two separate** `solve_triangular` calls per detector, loop-accumulated `A_inv`/`v`/`Q`/`logdetL`, one Cholesky, one $k$-vector solve. | was `model.py:789-906` at `7bc480a` → now `model.py:822-862` |
| 2 | Emit a single `numpyro.factor("logl_total", …)` in place of the per-detector `logl_{i}` factors. | was `model.py:901` at `7bc480a` → now `model.py:856-862` |
| 3 | **Retain** $Q$ and $\sum_i\sum_t\log[L_i]_{tt}$, so the numerical value of the log-likelihood is unchanged, not merely the posterior shape. | new block |
| 4 | Update the predictive draw in lockstep: `quads = solve(A_inv_chol.T, u + unit_quads)`. | was `model.py:908-956` at `7bc480a` → now `model.py:865-905` |
| 5 | Fix the comment `# (note that \|A\| = -\|A_inv\|)` → `log\|A\| = -log\|A_inv\|`. | was `model.py:887` at `7bc480a` → now `model.py:803`; done inside `ce5cf8a` |
| 6 | First tests for this code path (§7.4). | `tests/test_model.py` (new; 15 tests, green on this tree) |
| 7 | Docs + changelog (§7.6). | `docs/models.rst`, `docs/marginalized_likelihood.md`, `CHANGES.md` |

**Explicitly out of scope** — each investigated and rejected above; do not re-litigate:
`vmap`/`scan` over detectors and backend or dtype dispatch (§4.1); the `[M|y]` concatenated
solve (§4.3); $\Lambda = S^2$ and the Gram-side scale placement (§4.2); float32 (§4.6, §5.3);
prewhitening / passing $z_i$, $Q$, `logdetL` as model arguments (§4.4); the non-marginalized
branch (`model.py:957+` at `7bc480a`, already near-optimal at 4 `dtrsm`); and the `rd_design_matrix`
aligned-path cleanup and `h_det_mode` reshape (R3/R4 — real but tiny, separate PRs).

### 7.2 The diff, sketched

```python
k = dms.shape[2]                       # n_quad * n_modes, unchanged
A_inv   = jnp.eye(k)                   # 1 + sum_i W_i^T W_i   (accumulated)
v       = jnp.zeros(k)                 # sum_i W_i^T z_i
Q       = 0.0                          # sum_i ||z_i||^2       (theta-independent)
logdetL = 0.0                          # sum_i sum_t log L_tt  (theta-independent)

if not prior:
    for i in range(n_det):             # UNROLLED -- see section 4.1
        L = ls[i]
        W = jsp.linalg.solve_triangular(L, dms[i], lower=True)      # W_i = L_i^-1 M_i
        z = jsp.linalg.solve_triangular(L, strains[i], lower=True)  # z_i = L_i^-1 y_i
        A_inv   = A_inv   + W.T @ W
        v       = v       + W.T @ z
        Q       = Q       + jnp.dot(z, z)
        logdetL = logdetL + jnp.sum(jnp.log(jnp.diagonal(L)))

A_inv_chol = jsp.linalg.cholesky(A_inv, lower=True)                 # R
u = jsp.linalg.solve_triangular(A_inv_chol, v, lower=True)          # u = R^-1 v

if not prior:
    numpyro.factor(
        "logl_total",
        -0.5 * Q + 0.5 * jnp.dot(u, u)
        - logdetL - jnp.sum(jnp.log(jnp.diag(A_inv_chol))),
    )
```

**Resolved choices — do not re-litigate.** *Loop-accumulate, not build-then-sum:*
`A_inv = A_inv + W.T @ W` inside the loop is the form that was prototyped, verified and
benchmarked; stacking per-detector Grams into an $(n_{\rm det},k,k)$ array and summing is what
the `vmap` variant does, materializing an extra array for no gain. *Two separate solves*, not
the concatenated `[M|y]`. *Whitening lives inside `if not prior:`* — `dms` is still computed
above the branch (the predictive block needs it); only $W_i$, $z_i$ and the accumulators are
gated. *`A_inv_chol` and `u` are computed unconditionally*, outside the branch, so the
predictive block has a single code path; at $k=8$ the Cholesky of the identity is free. *Keep
the local name `A_inv_chol`* — it is the note's $R$ — so the diff reads against the existing
code and the note's symbol table stays valid.

**The `prior=True` limit, stated as an invariant.** Currently, with `prior=True` the loop is
skipped and the predictive block consumes the initial state `mu = 0`,
`Lambda_inv_chol = eye(k)`, giving `quads = ξ`, a draw from the $N(0,\mathbb{1})$ prior. Under
the one-shot form with the loop skipped the accumulators keep their initial values
$A^{-1} = \mathbb{1}$, $v = 0$, hence $R = \mathbb{1}$ and $u = 0$, so
`quads = R^{-T}(u + ξ) = ξ`. **Reproduced exactly, with no special-casing**, and no
`numpyro.factor` is emitted in either form.

### 7.3 Output / API compatibility

`numpyro.factor` is implemented as an **observed** sample site of the `Unit` distribution, and
`arviz_base.io_numpyro` builds the `log_likelihood` group from
`numpyro.infer.util.log_likelihood`, which filters on `site["type"]=="sample" and
site["is_observed"]` with **no `is_auxiliary` filter** — so factor sites *do* appear. Confirmed
against the committed `ringdown_fit.nc`, whose `log_likelihood` group is `['logl_0','logl_1']`,
`observed_data` is `['logl_0','logl_1','strain']` and `sample_stats` is `['diverging']` — note
**no `lp`**. **Repo-wide there is exactly one consumer**, `ringdown/result.py:487-492` in `Result.draw_sample(map=True)`, which falls back to `sum(v for k, v in log_likelihood.items()
if k.startswith("logl_"))` — and since `sampler.run` is called without `extra_fields`
(`fit.py:945`, `948`) there is no `lp`, so that fallback is the **live path**.

**Decision: name the merged site `logl_total`.** It still matches `startswith("logl_")`, so
`draw_sample(map=True)` keeps working with **zero changes to `result.py`**. Do not name it
`logl` (breaks the prefix match) and do not name it `strain` (collides with `get_arviz`'s
observed-data insertion at `model.py:1161` via the `k not in obsd` check at `fit.py:1010`).

A `Result` user sees `log_likelihood` and `observed_data` carrying **one** variable instead of
$n_{\rm det}$, each scalar-per-draw as before; `Result.draw_sample(map=True)` is unchanged;
`Result.loo` and `log_likelihood_timeseries` are unaffected, going through
`WHITENED_LOGLIKE_KEY`, which `_generate_whitened_residuals` computes independently from the
residuals. `MODEL_VARIABLES_BY_MODE` / `MODEL_DIMENSIONS` contain no `logl` entries; `tests/`
references `logl` nowhere; `docs/` only in stored notebook output cells.

**Should per-detector pointwise log-likelihood be preserved? No.** Nothing consumes it, and the
per-detector values were never a clean decomposition: $\ell_i = \log p(y_i\mid y_{<i},\theta)$
is a *conditional*, dependent on the arbitrary iteration order, so only the sum is
order-invariant. If a genuine need appears later, the right object is the *independent*
single-detector marginal likelihood $\log p(y_i)$, which is order-free and can be added cheaply
as `numpyro.deterministic` sites from the per-detector quantities already computed. Do not add
it speculatively. (*Pre-existing and unrelated, worth its own issue:* the `logl_`-prefix sum in
`draw_sample` already silently omits the `flat_a_prior`/`cutoff_lp` factors, so its "MAP" is
really a max-marginal-likelihood point.)

### 7.4 Test plan

**There is currently no test anywhere in `tests/` that touches `model.py` or `make_model`** —
the suite covers `data`, `imr`, `indexing`, `qnms`, `result`, `result_io`, `target`. This PR
adds the first coverage of the likelihood; CI runs `pytest -n=auto --cov=ringdown`. New file:
**`tests/test_model.py`**, with **the reference implementation living in the test, not in the
package**: transcribe the current sequential block into a module-level helper
`_sequential_reference(dms, ls, strains)`. Do *not* keep the old path in `ringdown/` behind a
flag — dead code there would need maintaining, would double the compile surface, and would
invite the `vmap`/dispatch variants back in.

| | Test | Method | Tolerance |
|---|---|---|---|
| **(a)** | Equivalence: potential energy **and every gradient component** | `initialize_model` → `potential_energy`; new model vs `_sequential_reference` at ≥8 random unconstrained points, for $(n_{\rm det}, n_t, n_{\rm mode})\in\{(1,205,1),(2,205,2),(3,205,3),(2,410,2)\}$. Use a **realistically conditioned** noise covariance (aLIGO-like ACF, $\operatorname{cond}(C)\sim10^{10}$), not a benign one. | **1e-11 relative.** Not 1e-15 — see §5.1; a 1e-15 threshold would be flaky. *As shipped: 1e-11 for the main configurations, an `eps · cond(C)` bound for the two conditioning-limited outliers — see §7.0.* |
| **(b)** | Predictive draw pointwise identity | Fix the PRNG key so the `*_unit` sites give the same $\xi$; compare `quads` and all derived sites (`a`, `phi`, `ellip`, `theta`, `phi_r`, `phi_l`, `apx/apy/acx/acy`, `h_det`, `h_det_mode`). Must be **pointwise** equal, not merely equal in distribution. | 1e-11 relative |
| **(c)** | `prior=True` equivalence | Build with `prior=True, predictive=True`; assert (i) no `logl*` site in the trace, (ii) `quads == ξ` for fixed $\xi$, per §7.2. | exact for (i); 1e-12 for (ii) |
| **(d)** | Seeded NUTS smoke | 1 chain, 300+300, fixed seed, (2,205,2); posterior means agree between old and new within MC error. ~5 s. *As shipped: the generic damped-sinusoid model (`f`, `g`, `a_scale`), not the Kerr model.* | 3 s.e. → **5 s.e.** as shipped. Near-vacuous at this length (ESS ≈ 3/300); see §7.0 issue 1. |
| **(e)** | Existing suite stays green | Checked: the only committed fixture, `tests/data/legacy_result_arviz0.nc`, has **no `log_likelihood` group at all**, so nothing encodes the `logl_i` names and nothing in `test_result*.py` can break on the rename. | — |
| **(f)** | Shell/integration | `tests/test_fit_config.sh` runs a real `ringdown_fit`; run manually once pre-merge (not in CI — it downloads data). This is the fit the end-to-end A/B of §3.7 used. Its data fetch currently fails on these workstations — §7.0 issue 2. | smoke |

Fixtures: synthesize data with an exponentially correlated ACF at $1/2048$ s and take its
Cholesky. Scale invariance means `a_scale_max = 1.0` with $O(1)$ data is representative of the
production $10^{-21}/10^{-21}$ pairing (the model is invariant under $y\to\alpha y$,
$L\to\alpha L$, $s_{\max}\to\alpha s_{\max}$ up to an additive constant).

### 7.5 Acceptance benchmarks

**Do not use wall-clock MCMC time to compare the two formulations** — see §3.6. Measure device
time per gradient with the `fori_loop` harness:

```bash
# correctness + per-gradient device time at the production config, CPU float64
.venv/bin/python benchmarks/h100/bench.py --platform cpu --x64 1 --no-sub \
    --sections env,correctness,devtime --configs '2,205,2' \
    --out /tmp/r1_accept.json --tag r1-acceptance
.venv/bin/python benchmarks/h100/analyze.py /tmp/r1_accept.json

# wider sweep including the large config, if time allows (~15 min)
.venv/bin/python benchmarks/h100/bench.py --platform cpu --x64 1 --no-sub \
    --configs '2,205,2;3,1024,8' --out /tmp/r1_accept_full.json
```

(`--no-sub` suppresses the GPU/float32 child legs; drop it on a GPU box for the full matrix.)
**Acceptance criteria:** the correctness block reports *all variants within 1e-11*; per-gradient
device time for the new path is ≥ **2.0×** faster than `current` at (2,205,2) on CPU f64
(measured 2.4×, and 2.81× on the H100 run's same-node CPU); no regression at (3,1024,8).

### 7.6 Docs, changelog and housekeeping

**Ship `docs/marginalized_likelihood.md` — but the Sphinx wiring this section originally
prescribed is SUPERSEDED.** The observation that prompted it still stands: `docs/conf.py` has no
`myst_parser` in `extensions` and no `source_suffix` override, so Sphinx does not recognize
`.md` and the note is not built. The plan was to add `myst_parser` + `myst-parser` to
`docs/requirements_docs.txt`, enable `dollarmath`, and put `marginalized_likelihood` in the
`docs/index.rst` toctree. **The user explicitly declined that wiring**, so PR #164 did none of
it: `docs/conf.py` and `docs/index.rst` are untouched, and the note reaches readers instead
through a short standalone **"Amplitude marginalization"** section added to `docs/models.rst`,
which explains what `marginalized` does and closes with a plain-text pointer to
``docs/marginalized_likelihood.md`` "in the source repository". Do not re-propose the toctree
wiring. The consequence to be aware of is §7.0 issue 3: **GitHub is now the note's only rendered
surface**, and its math does not render cleanly there.

**Development artifacts stay out of the package.** The present layout supersedes the earlier
"move everything to `dev/`" recommendation (correction 12): compiled note at
`docs/dev/model_optimization_study.md`, the H100 kit at `benchmarks/h100/`. Add whatever packaging excludes keep `benchmarks/` and `docs/dev/`
out of the wheel, and keep `docs/dev/` out of the Sphinx toctree.
`docs/marginalized_likelihood.md` is the one file that graduates into the built docs.

**`CHANGES.md`**, under `## Unreleased` — shipped essentially as drafted (folded into the docs
commit `b233c93`, with the speed claim generalized to "substantially faster per gradient" and
the readthedocs URL replaced by the repo path, since the note is not in the built docs):

```markdown
- Rewrote the marginalized likelihood as a single closed-form Gaussian
  marginalization over all detectors, replacing the sequential per-detector
  recursion (#NNN). This is an exact algebraic identity - the log-likelihood
  value, the priors and the sampled parameterization are unchanged - and makes
  the model ~2.4x faster per gradient on CPU and ~2.8x on an H100. The
  mathematics is documented in the new
  [marginalized likelihood note](https://ringdown.readthedocs.io/en/latest/marginalized_likelihood.html).
  - The per-detector `logl_0`, `logl_1`, ... factor sites are replaced by a
    single `logl_total` site, so `Result.log_likelihood` and
    `Result.observed_data` now carry one variable instead of one per detector.
    `Result.draw_sample(map=True)`, `Result.loo` and the whitened pointwise
    log-likelihood are unaffected. The old per-detector values were
    order-dependent conditionals log p(y_i | y_<i), never an independent
    per-detector decomposition.
```

**Config-side follow-ups.** Both originally proposed have now been discharged:
(1) ~~`chain_method='vectorized'` guidance/defaults~~ — **done**, `37baf98` (#163) on `main`,
plus the `docs/configuration.rst` `Performance` section; (2) ~~clean stale claims in
`MODEL_OPTIMIZATIONS.md`~~ — **done**, by this document (§0.3).

### 7.7 Commit and PR structure — as shipped

1. `7b5418a` `test: add reference sequential marginalized likelihood and equivalence tests` —
   adds `tests/test_model.py` with `_sequential_reference` and the equivalence tests **passing
   against the current implementation**, so the harness is demonstrably valid before anything
   changes.
2. `ce5cf8a` `perf: one-shot whitened marginalized likelihood` — changes 1–5 of §7.1 together
   (loop and predictive draw in one commit; they must not be separable), absorbing the
   `model.py:887` (at `7bc480a`) comment fix.
3. `b233c93` `docs: add the marginalized-likelihood methods note and changelog entry` — the
   note, the `models.rst` "Amplitude marginalization" section, and `CHANGES.md`. No Sphinx
   wiring (§7.6).
4. `ee33bac`, `d682181`, `e022f70` — Copilot-review fixes: the likelihood-vs-posterior
   degeneracy comment, the 3→5 s.e. smoke gate, and a stale notebook-output clear.

(The plan had the comment fix and the changelog as separate commits 3 and 5; both were folded
in, as recorded in §7.0.)

**PR skeleton** (drafted here; PR #164 as filed followed it, and added an *End-to-end
validation* section carrying the §3.7 numbers)

> **Title:** Rewrite the marginalized likelihood in closed form (≈2.4× faster per gradient)
>
> **Summary.** The marginalized branch integrates out the quadrature amplitudes one detector at
> a time, using each detector's posterior as the next one's prior. That recursion telescopes
> exactly onto a single closed-form expression: with $A^{-1} = \mathbb{1} + \sum_i M_i^\top
> C_i^{-1}M_i$ and $v = \sum_i M_i^\top C_i^{-1}y_i$, the total log-likelihood needs one
> Cholesky and one whitening solve per detector instead of $n_{\rm det}$ Choleskys and four
> `cho_solve`s each. **This is an identity, not an approximation:** the posterior, the priors,
> the sampled parameterization and the *numerical value* of the log-likelihood are unchanged
> (the same $-\tfrac12\sum_i n_i\log2\pi$ is dropped as before). Proof:
> `docs/marginalized_likelihood.md` §§4–5; independently re-derived and re-verified.
>
> **Numbers.** LAPACK `dtrsm` in the compiled gradient 42 → 8–10 ($n_{\rm det}=2$), Choleskys
> $n_{\rm det}$ → 1; per-gradient device time **2.4× on CPU f64**, **2.8× on H100**; compile
> −19% CPU, −70% GPU; and **2.09× per leapfrog on the real GW150914 fit** (§3.7). Agreement
> with the old path ~1e-15 on benign covariances, ~1e-12 with an aLIGO-like
> $\operatorname{cond}(C)\approx10^{10}$ (symmetric roundoff, not bias).
>
> **User-visible change.** One `logl_total` site replaces `logl_0…logl_{n-1}`; see CHANGES.md.
> **Not in this PR:** vmap/scan variants, backend/dtype dispatch, `[M|y]` concatenation,
> $\Lambda=S^2$, float32 — each investigated and rejected; rationale in
> `docs/dev/model_optimization_study.md`.

**Reviewer's checklist — the three things to be convinced of.** (1) *The telescoping identity
holds:* methods note §4, Lemma 1 (the running state is $(\mathbb{1}+J_i)^{-1}v_i$, so
intermediate means carry no independent information), Lemma 2 (determinants telescope because
$R^{(0)}=\mathbb{1}$), Lemma 3 (all $\mu$-dependence collapses to $\mu^\top(N+P-S)\mu = 0$);
test (a) is the empirical counterpart. (2) *The constants are retained:* $Q$ and
$\sum_i\sum_t\log[L_i]_{tt}$ are $\theta$-independent and could legally be dropped, but are kept
so reported log-likelihood values match the old code exactly rather than up to an offset —
check the signs. (3) *The predictive draw changed in lockstep:* `quads = solve(R.T, u + ξ)`
replaces `mu + solve(Lambda_inv_chol.T, ξ)`; if the loop changed and this did not, the
log-likelihood would still be right and the posterior-predictive draws silently wrong. Test (b)
pins it pointwise, test (c) pins the `prior=True` limit.

---

## Appendix. Index of raw evidence

Nothing below needs to be re-read to implement the PR, but a reviewer wanting a specific number
will find it here.

**Live references.** **PR #164**, merged as `b6e30b9` — the implementation, the test suite
(`tests/test_model.py`), and the *End-to-end validation* section behind §3.7. The pre-rewrite
code this document analyzes is pinned at `7bc480a`. `docs/marginalized_likelihood.md` — the methods note: §1 notation, §2 the
Woodbury/determinant lemmas, §3 the sequential recursion, §4 the telescoping proof (Lemmas 1–3),
§5 the whitened one-shot form and its conditioning, §6 the predictive draw, §7 the symbol table
and normative code sketch. `benchmarks/h100/` — `PREDICTIONS.md` (pre-registered intervals
P1–P12 and falsification tests F1–F5), `bench.py` (the harness: dispatch-free `fori_loop` device
timing, correctness block, chain scaling, RHS sweep, float32 robustness — reusable for §7.5),
`analyze.py` (pure-stdlib scorer; run it on `results_6969321.json` to reproduce every H100
number in §3.3), plus the raw results and Slurm logs for job 6969321.

**Original reports (not retained)** — the six source documents were development artifacts,
deleted after consolidation into this study; superseded claims are cataloged in §0.3. Citations
to them by name below and throughout refer to the documents as they existed during the session.
For orientation, their contents were: `MODEL_OPTIMIZATIONS.md`: CPU methodology and appendix B.1–B.9 (isolated
timings with op counts, end-to-end, per-configuration equivalence, the conditioning stress
table, the DCE evidence, `h_det_mode` reshape timings, non-marginalized HLO, trace-time costs,
per-leapfrog timing under contention). `GPU_BENCHMARKS.md`: A6000 methodology and the
device-time harness, appendix B.1–B.10 (GPU correctness, 11-configuration timings, GPU HLO op
counts, batched-vs-unrolled primitives and the RHS-width sweep, concat vs separate, end-to-end,
the chain table, float32, the GEMM attribution probe), plus the record of the additive
`jax[cuda12]` install and its rollback. `CLAIMS_VERIFICATION.md`: the full verdict table (claims
0.1–2.9), the independent re-derivation, and the sweeps behind corrections 2, 3 and 5, plus the
`OMP_NUM_THREADS` robustness check. `BACKEND_DISPATCH.md`: the float64/float32 decision matrices
over 12 configurations, the dense $n_t$ crossover scan, the chains-interaction grids at
$C\in\{1,4,16,64\}$, the `lax.scan` measurement, the dispatch-mechanism failure-mode table, the
ragged-input shape analysis, and the recommended tolerances. `AMPLITUDE_SCALE_PLACEMENT.md`: the
(a)/(b)/(c) algebra with the diagonal-congruence proof, the conditioning sweep to
$s/s_{\max} = 10^{-139}$, the observed NUTS `a_scale` range over four seeds, the HLO fusion
counts, and the erratum that fixed eq. (2.5) of the methods note. `R1_PR_PLAN.md`: the source of
§7.

**Environment.** All measurements used `/mnt/home/misi/src/ringdown/.venv`: Python 3.13,
jax/jaxlib 0.11.1, numpyro 0.21.0, numpy 2.5.2, arviz-base 1.3.0, `jax_enable_x64=True` unless
stated. CPU work on a shared 32-core Flatiron workstation (absolute times drift ±30% between
sessions; ratios stable to a few percent); A6000 work on that workstation's idle RTX A6000
(driver 595.91.07), with `jax[cuda12]==0.11.1` installed purely additively; H100 work on `rusty`
(`workergpu162`, H100 PCIe, driver 580.178.04) against a same-node 16-core Xeon 8362 CPU
baseline. Every comparison interleaved variants within each repeat; medians over 5–11 repeats.
