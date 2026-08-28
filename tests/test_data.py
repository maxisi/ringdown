import numpy as np
import ringdown as rd


def test_power_spectrum_from_data_series():
    # scipy.signal.welch 1.16+ indexes with tuples; pandas Series raises KeyError
    # unless we pass an ndarray. Data is a Series subclass.
    rng = np.random.default_rng(0)
    fs = 4096
    t = np.arange(int(2.0 * fs)) / fs
    d = rd.Data(rng.normal(size=len(t)), index=t, ifo="H1")
    psd = rd.PowerSpectrum.from_data(d)
    assert len(psd) > 1
    assert np.all(np.isfinite(psd))
    acf = d.get_acf()
    assert len(acf) > 1
    assert np.all(np.isfinite(acf))
