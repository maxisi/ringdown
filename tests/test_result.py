import json
from html import escape

import numpy as np
import pytest
import xarray as xr
from arviz_base import dict_to_dataset

from ringdown.model import Aellip_from_quadratures
from ringdown.result import Result
from ringdown.utils.swsh import construct_sYlm


def test_strain_scale_length_one_constant_data():
    # dict_to_dataset promotes a scalar to shape (1,) with a dummy dim.
    ds = dict_to_dataset({"scale": 2.0}, sample_dims=[])
    result = Result(xr.DataTree.from_dict({"constant_data": ds}))
    assert result.strain_scale == 2.0


def test_repr_html_config_collapsible():
    # config attribute renders as nested collapsible <details> elements
    ds = dict_to_dataset({"scale": 2.0}, sample_dims=[])
    result = Result(xr.DataTree.from_dict({"constant_data": ds}))
    config = {"model": {"modes": 2, "prior": {"a_max": 1e-20}}, "run": "x"}
    result.attrs["config"] = json.dumps(config)
    result.attrs["other"] = "plain value"
    html = result._repr_html_()
    # raw escaped JSON blob replaced by collapsible view
    assert f"<dd>{escape(result.attrs['config'])}</dd>" not in html
    assert "<summary style='cursor:pointer'>model</summary>" in html
    assert "<summary style='cursor:pointer'>prior</summary>" in html
    # every level collapsed by default
    assert "<details open" not in html
    # leaves rendered as key: value lines
    assert "a_max: 1e-20" in html
    # other attributes untouched
    assert "<dd>plain value</dd>" in html


def _aligned_posterior(modes, nchain=2, ndraw=10, cosi=True, seed=1234):
    # synthetic aligned-model posterior with amplitudes for the given modes
    rng = np.random.default_rng(seed)
    data = {
        "a": (
            ("chain", "draw", "mode"),
            rng.uniform(0.1, 5.0, (nchain, ndraw, len(modes))),
        ),
    }
    if cosi:
        data["cosi"] = (
            ("chain", "draw"),
            rng.uniform(-1, 1, (nchain, ndraw)),
        )
    posterior = xr.Dataset(
        data,
        coords={
            "chain": np.arange(nchain),
            "draw": np.arange(ndraw),
            "mode": np.array(modes),
        },
    )
    return Result(xr.DataTree.from_dict({"posterior": posterior}))


def test_generic_amplitude_matches_quadrature_definition():
    # the aligned-to-generic amplitude conversion must reproduce the
    # generic model's own amplitude definition when applied to the
    # aligned-model quadratures
    rng = np.random.default_rng(42)
    for ell, m in [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)]:
        cosi = rng.uniform(-1, 1)
        c = rng.uniform(0.1, 5.0)
        phi = rng.uniform(0, 2 * np.pi)
        swsh = construct_sYlm(-2, ell, m)
        ylm_p = np.asarray(swsh(cosi)).ravel()[0]
        ylm_m = np.asarray(swsh(-cosi)).ravel()[0]
        yp, yc = ylm_p + ylm_m, ylm_p - ylm_m
        apx, apy = c * yp * np.cos(phi), c * yp * np.sin(phi)
        acx, acy = -c * yc * np.sin(phi), c * yc * np.cos(phi)
        a_generic, _ = Aellip_from_quadratures(apx, apy, acx, acy)
        a_converted = c * (np.abs(ylm_p) + np.abs(ylm_m))
        assert np.isclose(float(a_generic), a_converted)


def test_get_generic_amplitude():
    # includes a retrograde mode, which must be indexed correctly
    modes = [b"1,-2,2,2,0", b"1,-2,3,3,0", b"-1,-2,2,2,0"]
    result = _aligned_posterior(modes)
    a = result.get_generic_amplitude()
    assert a.dims == ("chain", "draw", "mode")
    assert list(a.mode.values) == modes
    cosi = result.posterior.cosi.values
    for mode in modes:
        _, _, ell, m, _ = map(int, mode.decode().split(","))
        swsh = construct_sYlm(-2, ell, m)
        expected = result.posterior.a.sel(mode=mode).values * (
            np.abs(np.asarray(swsh(cosi)))
            + np.abs(np.asarray(swsh(-cosi)))
        )
        assert np.allclose(a.sel(mode=mode).values, expected)


def test_get_generic_amplitude_fixed_cosi():
    # a fit with fixed inclination stores no cosi samples; the value must
    # be picked up from the configuration instead
    modes = [b"1,-2,2,2,0"]
    result = _aligned_posterior(modes, cosi=False)
    result.attrs["config"] = json.dumps({"model": {"cosi": "0.3"}})
    a = result.get_generic_amplitude()
    swsh = construct_sYlm(-2, 2, 2)
    expected = result.posterior.a.sel(mode=modes[0]).values * (
        np.abs(np.asarray(swsh(0.3))) + np.abs(np.asarray(swsh(-0.3)))
    )
    assert np.allclose(a.sel(mode=modes[0]).values, expected)


def test_get_generic_amplitude_requires_aligned_fit():
    modes = [b"1,-2,2,2,0"]
    result = _aligned_posterior(modes, cosi=False)
    with pytest.raises(KeyError, match="aligned-model"):
        result.get_generic_amplitude()
    result.posterior["apx"] = result.posterior.a
    with pytest.raises(ValueError, match="generic"):
        result.get_generic_amplitude()


def test_repr_html_without_config():
    # repr must not raise or inject details when config is absent or invalid
    ds = dict_to_dataset({"scale": 2.0}, sample_dims=[])
    result = Result(xr.DataTree.from_dict({"constant_data": ds}))
    assert "<details" not in result._repr_html_()
    result.attrs["config"] = "not json {"
    assert isinstance(result._repr_html_(), str)
