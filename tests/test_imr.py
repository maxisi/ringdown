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
