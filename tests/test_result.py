import json
from html import escape

import xarray as xr
from arviz_base import dict_to_dataset

from ringdown.result import Result


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


def test_repr_html_without_config():
    # repr must not raise or inject details when config is absent or invalid
    ds = dict_to_dataset({"scale": 2.0}, sample_dims=[])
    result = Result(xr.DataTree.from_dict({"constant_data": ds}))
    assert "<details" not in result._repr_html_()
    result.attrs["config"] = "not json {"
    assert isinstance(result._repr_html_(), str)
