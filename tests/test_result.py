import arviz as az
from ringdown.result import Result


def test_strain_scale_length_one_constant_data():
    # ArviZ dict_to_dataset promotes a scalar to shape (1,) with a dummy dim.
    ds = az.dict_to_dataset({"scale": 2.0}, default_dims=[])
    result = Result(az.InferenceData(constant_data=ds))
    assert result.strain_scale == 2.0
