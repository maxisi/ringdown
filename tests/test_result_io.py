"""I/O and API tests for Result, including backward compatibility with
result files written by arviz 0.x (pre-DataTree)."""

import os

import numpy as np
import pytest

from ringdown.result import Result

LEGACY_PATH = os.path.join(
    os.path.dirname(__file__), "data", "legacy_result_arviz0.nc"
)


@pytest.fixture
def legacy_result():
    return Result.from_netcdf(LEGACY_PATH)


def test_legacy_netcdf_read(legacy_result):
    r = legacy_result
    for group in ["posterior", "sample_stats", "observed_data", "constant_data"]:
        assert group in r
    assert "h_det_mode" in r.posterior
    assert r.strain_scale == 1e-21
    # config JSON stored in root attrs must decode
    assert r.config["injection"]["a"] == [1e-21, 5e-22]


def test_legacy_netcdf_skip_h_det_mode():
    r = Result.from_netcdf(LEGACY_PATH, load_h_det_mode=False)
    assert "h_det_mode" not in r.posterior
    assert "a" in r.posterior


def test_netcdf_roundtrip(legacy_result, tmp_path):
    path = str(tmp_path / "roundtrip.nc")
    legacy_result.to_netcdf(path)
    r = Result.from_netcdf(path)
    np.testing.assert_allclose(
        r.posterior["a"].values, legacy_result.posterior["a"].values
    )
    assert r.config == legacy_result.config


def test_copy_preserves_config(legacy_result):
    assert legacy_result.config  # populate lazy cache
    copied = legacy_result.copy()
    assert isinstance(copied, Result)
    assert copied.config == legacy_result.config


def test_stacked_samples(legacy_result):
    stacked = legacy_result.stacked_samples
    nchain = legacy_result.posterior.sizes["chain"]
    ndraw = legacy_result.posterior.sizes["draw"]
    assert stacked.sizes["sample"] == nchain * ndraw


def test_ess(legacy_result):
    ess = legacy_result.ess
    assert np.isfinite(ess) and ess > 0
