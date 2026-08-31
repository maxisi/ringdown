"""Tests for sampler keyword handling in :mod:`ringdown.fit`."""

from configparser import ConfigParser

import pytest

from ringdown import fit as rdfit
from ringdown.cli import ringdown_fit


def _patch_device(monkeypatch, backend, device_count):
    monkeypatch.setattr(rdfit.jax, "default_backend", lambda: backend)
    monkeypatch.setattr(
        rdfit.jax, "local_device_count", lambda: device_count
    )


@pytest.mark.parametrize(
    "backend, device_count, expected",
    [
        # fewer accelerators than chains: NumPyro's 'parallel' would fall
        # back to sequential sampling, so vectorize instead
        ("gpu", 1, "vectorized"),
        ("gpu", 2, "vectorized"),
        # enough accelerators for one chain each: leave 'parallel' alone
        ("gpu", 4, None),
        ("gpu", 8, None),
        # CPU keeps NumPyro's default regardless of device count
        ("cpu", 1, None),
        ("cpu", 4, None),
    ],
)
def test_chain_method_default(monkeypatch, backend, device_count, expected):
    _patch_device(monkeypatch, backend, device_count)
    _, sampler_kws, _ = rdfit.get_sampling_kwargs(num_chains=4)
    assert sampler_kws.get("chain_method") == expected


def test_chain_method_not_set_for_single_chain(monkeypatch):
    _patch_device(monkeypatch, "gpu", 1)
    _, sampler_kws, _ = rdfit.get_sampling_kwargs(num_chains=1)
    assert "chain_method" not in sampler_kws


@pytest.mark.parametrize(
    "kws",
    [
        {"chain_method": "sequential"},
        {"sampler": {"chain_method": "sequential"}},
    ],
)
def test_explicit_chain_method_is_respected(monkeypatch, kws):
    _patch_device(monkeypatch, "gpu", 1)
    _, sampler_kws, _ = rdfit.get_sampling_kwargs(num_chains=4, **kws)
    assert sampler_kws["chain_method"] == "sequential"


def test_cpu_device_count_is_clamped(monkeypatch, tmp_path):
    device_counts = []

    class Fit:
        result = type("Result", (), {"to_netcdf": lambda *_: None})()
        prior = result

        def run(self, **kwargs):
            pass

    monkeypatch.setattr(ringdown_fit.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        ringdown_fit.rd.utils, "load_config", lambda _: ConfigParser()
    )
    monkeypatch.setattr(
        ringdown_fit.rd.Fit, "from_config", lambda _: Fit()
    )
    monkeypatch.setattr(ringdown_fit.numpyro, "set_platform", lambda _: None)
    monkeypatch.setattr(
        ringdown_fit.numpyro, "set_host_device_count", device_counts.append
    )
    monkeypatch.setattr(ringdown_fit.jax_config, "update", lambda *_: None)

    ringdown_fit.main([
        "--device-count", "4", "--output", str(tmp_path / "fit.nc"), "input.ini"
    ])

    assert device_counts == [2]
