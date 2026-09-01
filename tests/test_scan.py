"""Tests for the scan command-line interface."""

from configparser import ConfigParser

from ringdown.cli import ringdown_scan


def test_cli_delegates_configuration_to_setup(monkeypatch, tmp_path):
    setup_calls = []

    class FitSequence:
        targets = []

    config_path = tmp_path / "input.ini"
    config_path.touch()

    monkeypatch.setattr(
        ringdown_scan.rd.utils, "load_config", lambda _: ConfigParser()
    )
    monkeypatch.setattr(
        ringdown_scan.rd.FitSequence, "from_config", lambda _: FitSequence()
    )
    monkeypatch.setattr(
        ringdown_scan.rd, "setup", lambda **kws: setup_calls.append(kws)
    )

    ringdown_scan.main(["--device-count", "4", str(config_path)])

    # device clamping etc. are tested in test_setup.py; here we only check
    # that the CLI hands its arguments through
    assert setup_calls == [{"platform": "cpu", "num_devices": 4, "x64": True}]
