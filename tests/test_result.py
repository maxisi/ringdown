import xarray as xr
from arviz_base import dict_to_dataset

from ringdown.result import Result


def test_strain_scale_length_one_constant_data():
    # dict_to_dataset promotes a scalar to shape (1,) with a dummy dim.
    ds = dict_to_dataset({"scale": 2.0}, sample_dims=[])
    result = Result(xr.DataTree.from_dict({"constant_data": ds}))
    assert result.strain_scale == 2.0
