import numpy as np
import ringdown as rd
from ringdown.imr import IMRResult


def test_imr_result_construct_and_copy():
    f = np.arange(9, dtype=float) * 16.0
    psd = rd.PowerSpectrum(np.ones_like(f) * 1e-46, index=f, ifo="H1")
    result = IMRResult(
        {"mass_1": [30.0, 31.0], "mass_2": [20.0, 21.0]},
        attrs={"foo": 1},
        psds={"H1": psd},
    )
    assert isinstance(result, IMRResult)
    assert result.attrs.get("foo") == 1
    assert "H1" in result.psds

    copied = result.copy()
    assert isinstance(copied, IMRResult)
    assert copied.attrs.get("foo") == 1
    assert "H1" in copied.psds
    assert copied.psds["H1"].ifo == "H1"

    sliced = result.iloc[:1]
    assert isinstance(sliced, IMRResult)
    assert sliced.attrs.get("foo") == 1
    assert "H1" in sliced.psds


def test_from_pesummary_keeps_config(tmp_path):
    # regression test: attrs handling used to drop the config entirely
    # (dict.update returns None), leaving trigger_time etc. unset
    import h5py

    path = str(tmp_path / "pe.h5")
    samples = np.array(
        [(30.0, 20.0, 1126259462.4), (31.0, 21.0, 1126259462.4)],
        dtype=[("mass_1", float), ("mass_2", float),
               ("geocent_time", float)],
    )
    with h5py.File(path, "w") as f:
        g = f.create_group("C01:IMRPhenomXPHM")
        g["posterior_samples"] = samples
        c = g.create_group("config_file").create_group("config")
        c["trigger_time"] = 1126259462.391
        c["duration"] = 4.0

    result = IMRResult.from_pesummary(path)
    assert result.config
    assert result.trigger_time == 1126259462.391
    assert result.duration == 4.0

    # explicit attrs are merged with the config, not dropped,
    # and the caller's dict is not mutated
    attrs = {"foo": 1}
    result = IMRResult.from_pesummary(path, attrs=attrs)
    assert result.attrs.get("foo") == 1
    assert result.config
    assert attrs == {"foo": 1}

    # the construct() wrapper also keeps the config
    result = IMRResult.construct(path)
    assert result.config
    assert result.trigger_time == 1126259462.391
