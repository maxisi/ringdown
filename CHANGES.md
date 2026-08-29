# Changelog

Notable user-facing changes to `ringdown`. For a full history, see the
[git log](https://github.com/maxisi/ringdown/commits/main).

## Unreleased

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
