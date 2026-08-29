import numpy as np
import pytest
import ringdown as rd
from ringdown.waveforms.core import Signal
from ringdown.waveforms.ringdown import Ringdown


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


def test_data_ifo_survives_interpolate_and_sort():
    d = rd.Data([1.0, 2.0, 3.0, 4.0], index=[0.0, 0.25, 0.5, 0.75], ifo="H1")
    d2 = d.interpolate_to_index([0.0, 0.5])
    assert isinstance(d2, rd.Data)
    assert d2.ifo == "H1"

    shuffled = rd.Data([1.0, 2.0, 3.0], index=[3, 1, 2], ifo="L1")
    sorted_d = shuffled.sort_index()
    assert isinstance(sorted_d, rd.Data)
    assert sorted_d.ifo == "L1"
    assert sorted_d.index.is_monotonic_increasing


def test_data_condition():
    rng = np.random.default_rng(1)
    fs = 4096
    t0 = 1.0
    t = np.arange(int(4.0 * fs)) / fs
    d = rd.Data(rng.normal(size=len(t)), index=t, ifo="H1")
    c = d.condition(t0=t0, ds=2, f_min=20, trim=0.25)
    assert isinstance(c, rd.Data)
    assert c.ifo == "H1"
    assert c.index.is_monotonic_increasing
    assert len(c) < len(d)

    # roll+no-trim can leave the time series out of order and hits sort_index
    t0_unaligned = t[3]
    c2 = d.condition(t0=t0_unaligned, ds=2, trim=0, remove_mean=False)
    assert isinstance(c2, rd.Data)
    assert c2.ifo == "H1"
    assert c2.index.is_monotonic_increasing


def test_power_spectrum_gate_and_fill_power_of_two():
    f = np.linspace(0.0, 1024.0, 1025)
    vals = np.ones_like(f) * 1e-46
    vals[10] = 1e-20
    psd = rd.PowerSpectrum(vals, index=f, ifo="H1", fill_power_of_two=False)
    gated = psd.gate(inplace=False)
    assert gated is not psd
    assert isinstance(gated, rd.PowerSpectrum)
    assert gated.ifo == "H1"
    assert gated.iloc[10] < vals[10]

    psd.gate(inplace=True)
    assert psd.iloc[10] < vals[10]
    assert psd.ifo == "H1"

    # 0..1023 Hz at 1 Hz spacing: constructor should append 1024 Hz
    f_odd = np.arange(1024, dtype=float)
    filled = rd.PowerSpectrum(np.ones_like(
        f_odd) * 1e-46, index=f_odd, ifo="V1")
    assert filled.index[-1] == 1024
    assert filled.ifo == "V1"

    psd2 = rd.PowerSpectrum(
        np.ones_like(f_odd) * 1e-46,
        index=f_odd,
        ifo="V1",
        fill_power_of_two=False,
    )
    assert psd2.index[-1] == 1023
    psd2.fill_power_of_two()
    assert psd2.index[-1] == 1024
    assert psd2.ifo == "V1"


def test_signal_parameters_survive_slicing():
    s = Signal(
        [0.0, 1.0, 2.0],
        index=[0.0, 1.0, 2.0],
        parameters={"geocent_time": 1.0},
    )
    sl = s.iloc[1:]
    assert isinstance(sl, Signal)
    assert sl.parameters.get("geocent_time") == 1.0

    r = Ringdown(
        [0.0, 1.0, 2.0],
        index=[0.0, 1.0, 2.0],
        parameters={"geocent_time": 1.0},
        modes=[(1, -2, 2, 2, 0)],
    )
    rsl = r.iloc[1:]
    assert isinstance(rsl, Ringdown)
    assert rsl.parameters.get("geocent_time") == 1.0
    assert len(rsl.modes) == 1


def test_gwpy_timeseries_value_and_times():
    # Series.read(kind='frame') and Series.fetch copy GWpy .value / .times
    pytest.importorskip("gwpy")
    from gwpy.timeseries import TimeSeries

    ts = TimeSeries([0.0, 1.0, 2.0], sample_rate=4)
    d = rd.Data(ts.value, index=np.array(ts.times), ifo="H1")
    assert isinstance(d, rd.Data)
    assert d.ifo == "H1"
    assert len(d) == 3
    assert np.allclose(d.to_numpy(), [0.0, 1.0, 2.0])
