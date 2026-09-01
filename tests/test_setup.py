"""Tests for :func:`ringdown.setup` and the import-time threading default."""

import os
import subprocess
import sys

import jax
import numpyro
import pytest

import ringdown as rd
from ringdown import _setup


@pytest.fixture
def calls(monkeypatch):
    """Divert every side effect of setup() into a record, so no test
    ever touches the real jax/numpyro configuration."""
    record = {"platform": [], "devices": [], "config": []}
    monkeypatch.setattr(numpyro, "set_platform", record["platform"].append)
    monkeypatch.setattr(
        numpyro, "set_host_device_count", record["devices"].append
    )
    monkeypatch.setattr(
        jax.config, "update", lambda *args: record["config"].append(args)
    )
    monkeypatch.setattr(_setup, "_backends_are_initialized", lambda: False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.delenv("RINGDOWN_DEVICE_COUNT", raising=False)
    # register the current value for restoration, since setup() writes it
    monkeypatch.setenv(
        "OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1")
    )
    return record


def test_cpu_defaults(calls):
    rd.setup()
    assert calls["platform"] == ["cpu"]
    assert calls["devices"] == [4]
    assert calls["config"] == [("jax_enable_x64", True)]
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_gpu_defaults_and_cuda_mapping(calls):
    # numpyro 0.21 rejects 'gpu': setup() must hand it 'cuda'
    rd.setup(platform="gpu")
    assert calls["platform"] == ["cuda"]
    assert calls["devices"] == [1]
    assert calls["config"] == [("jax_enable_x64", False)]


def test_cuda_alias_gets_gpu_defaults(calls):
    rd.setup(platform="cuda")
    assert calls["platform"] == ["cuda"]
    assert calls["devices"] == [1]
    assert calls["config"] == [("jax_enable_x64", False)]


def test_device_count_from_environment(calls, monkeypatch):
    monkeypatch.setenv("RINGDOWN_DEVICE_COUNT", "2")
    rd.setup()
    assert calls["devices"] == [2]


def test_cpu_device_count_is_clamped(calls, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    rd.setup(num_devices=4)
    assert calls["devices"] == [2]


def test_explicit_arguments_override_defaults(calls):
    rd.setup(platform="gpu", num_devices=3, x64=True, num_threads=2)
    assert calls["platform"] == ["cuda"]
    assert calls["devices"] == [3]
    assert calls["config"] == [("jax_enable_x64", True)]
    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_late_call_with_different_config_raises(calls, monkeypatch):
    monkeypatch.setattr(_setup, "_backends_are_initialized", lambda: True)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(jax, "local_device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="before any jax operation"):
        rd.setup()  # requests 4 devices, only 1 active
    # nothing may have been reconfigured
    assert calls["platform"] == []
    assert calls["devices"] == []
    assert calls["config"] == []


def test_late_call_with_matching_config_is_noop(calls, monkeypatch):
    monkeypatch.setattr(_setup, "_backends_are_initialized", lambda: True)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(jax, "local_device_count", lambda: 4)
    rd.setup(x64=bool(jax.config.jax_enable_x64))
    assert calls["platform"] == []
    assert calls["devices"] == []
    assert calls["config"] == []


def _omp_after_import(env):
    code = "import os, ringdown; print(os.environ['OMP_NUM_THREADS'])"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_import_sets_omp_default():
    env = {
        k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"
    }
    assert _omp_after_import(env) == "1"


def test_import_respects_preset_omp():
    env = dict(os.environ, OMP_NUM_THREADS="8")
    assert _omp_after_import(env) == "8"
