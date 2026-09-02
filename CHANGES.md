# Changelog

Notable user-facing changes to `ringdown`. For a full history, see the
[git log](https://github.com/maxisi/ringdown/commits/main).

## Unreleased

- Added `ringdown.setup()`, a one-call replacement for the jax/numpyro
  configuration boilerplate: it sets the platform, CPU host device count
  (honoring `RINGDOWN_DEVICE_COUNT`, defaulting to 4 and clamped to the CPU
  count; on GPU/TPU the device count is not controlled here, and an explicit
  `num_devices` is ignored with a warning), and precision (`x64` defaults
  to true on CPU, false on GPU), and raises if called after jax has already
  initialized its backends, where jax and numpyro would silently ignore the
  settings. Both CLIs and the example notebooks now use it.
- Importing `ringdown` now sets `OMP_NUM_THREADS=1` by default (a pre-set
  value is respected), so BLAS/OpenMP threading is capped before numpy/jax
  load it. Previously only the CLIs set this, and they did so too late for
  numpy. Override with the environment variable or `setup(num_threads=...)`.
- Importing `ringdown` now suppresses the `Wswiglal-redir-stdio` warning
  that `lal` emits when first imported under IPython/Jupyter, so notebooks
  no longer need `warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")`
  before `import ringdown`. The filter is scoped to that one import and
  does not alter the interpreter's warning configuration.
- Fixed `--platform gpu` in the CLIs, which crashed with an `AssertionError`
  under NumPyro 0.21 (`numpyro.set_platform` accepts `'cuda'` but no longer
  `'gpu'`); the platform name is now translated automatically.
- Rewrote the marginalized likelihood as a single closed-form Gaussian
  marginalization over all detectors, replacing the sequential per-detector
  recursion. This is an exact algebraic identity - the log-likelihood value,
  the priors and the sampled parameterization are unchanged - and makes the
  model substantially faster per gradient. The mathematics is documented in
  the new methods note at `docs/marginalized_likelihood.md`.
  - The per-detector `logl_0`, `logl_1`, ... factor sites are replaced by a
    single `logl_total` site, so `Result.log_likelihood` and
    `Result.observed_data` now carry one variable instead of one per detector.
    `Result.draw_sample(map=True)`, `Result.loo` and the whitened pointwise
    log-likelihood are unaffected. The old per-detector values were
    order-dependent conditionals log p(y_i | y_<i), never an independent
    per-detector decomposition.
- On accelerators (GPU/TPU), `chain_method` now defaults to `'vectorized'`
  whenever there are fewer devices than chains, rather than letting NumPyro
  fall back to drawing chains sequentially. Pass `chain_method` explicitly
  (directly, via `sampler`, or the `[run]` config section) to override.
- Fixed the CLI `--device-count` clamp: requesting more CPU devices than
  available cores logged a warning but then skipped
  `numpyro.set_host_device_count` entirely, leaving JAX on a single device.
- Migrated to arviz 1.x (#154): `Result` now subclasses `xarray.DataTree`
  instead of wrapping `arviz.InferenceData`. Most idioms carry over; notable
  changes:
  - `Result.waic` is removed (dropped upstream); use `loo`.
  - JSON result I/O is removed; results are netCDF only. Files written with
    pre-1.x arviz remain readable.
  - `Result.plot_trace` is now backed by `arviz_plots.plot_trace_dist`.
  - Dependency pin changed to `arviz[matplotlib,h5netcdf]~=1.3`.
- Upgraded to NumPyro 0.21 and JAX 0.11; Python >= 3.12 is now required, and
  Python 3.13/3.14 are supported (#151).
- Dependency bumps: numpy 2.5 (#152), lalsuite 7.26.15, h5py 3.16.

## v1.1.0 (2026-08-28)

- Expanded support for initializing and comparing fits with
  inspiral-merger-ringdown (IMR) posteriors (`IMRResult`,
  `Fit.from_imr_result`, `[imr]` config section), including whitened
  residuals and LOO computations, with workflow fixes (#121).
- Added the `ringdown_pp_pipe` executable for probability-probability (PP)
  tests, along with PP plotting utilities.
- `ResultCollection` improvements (reference times, multi-indexing).
- GWOSC downloads point at gwosc.org.
- Bumped NumPyro to 0.19.
- Package versions are now derived only from `v*` git tags.

## v1.0.0 (2024-10-02)

- See the git history for changes up to this release.
