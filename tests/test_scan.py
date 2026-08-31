"""Tests for the scan command-line interface."""

from configparser import ConfigParser

from ringdown.cli import ringdown_scan


def test_cpu_device_count_is_clamped(monkeypatch, tmp_path):
    device_counts = []

    class FitSequence:
        targets = []

    config_path = tmp_path / "input.ini"
    config_path.touch()

    monkeypatch.setattr(ringdown_scan.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        ringdown_scan.rd.utils, "load_config", lambda _: ConfigParser()
    )
    monkeypatch.setattr(
        ringdown_scan.rd.FitSequence, "from_config", lambda _: FitSequence()
    )
    monkeypatch.setattr(ringdown_scan.numpyro, "set_platform", lambda _: None)
    monkeypatch.setattr(
        ringdown_scan.numpyro, "set_host_device_count", device_counts.append
    )
    monkeypatch.setattr(ringdown_scan.jax_config, "update", lambda *_: None)

    ringdown_scan.main(["--device-count", "4", str(config_path)])

    assert device_counts == [2]
