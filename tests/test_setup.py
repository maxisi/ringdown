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
    # pin a deterministic baseline (restored after), since setup() both
    # reads and writes this variable
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
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
    # the host device count only applies to CPU: no call on GPU
    assert calls["devices"] == []
    assert calls["config"] == [("jax_enable_x64", False)]


def test_cuda_alias_gets_gpu_defaults(calls):
    rd.setup(platform="cuda")
    assert calls["platform"] == ["cuda"]
    assert calls["devices"] == []
    assert calls["config"] == [("jax_enable_x64", False)]


def test_gpu_ignores_num_devices_with_warning(calls, caplog):
    caplog.set_level("WARNING", logger="ringdown._setup")
    rd.setup(platform="gpu", num_devices=2)
    assert calls["platform"] == ["cuda"]
    assert calls["devices"] == []
    assert "not controlled on accelerator platforms" in caplog.text


def test_device_count_from_environment(calls, monkeypatch):
    monkeypatch.setenv("RINGDOWN_DEVICE_COUNT", "2")
    rd.setup()
    assert calls["devices"] == [2]


def test_cpu_device_count_is_clamped(calls, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    rd.setup(num_devices=4)
    assert calls["devices"] == [2]


def test_unknown_cpu_count_disables_clamp(calls, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    rd.setup(num_devices=4)
    assert calls["devices"] == [4]


def test_explicit_arguments_override_defaults(calls):
    rd.setup(num_devices=3, x64=False, num_threads=2)
    assert calls["platform"] == ["cpu"]
    assert calls["devices"] == [3]
    assert calls["config"] == [("jax_enable_x64", False)]
    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_setup_respects_exported_omp_threads(calls, monkeypatch):
    # OMP_NUM_THREADS=16 ringdown_fit ... must keep 16
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    rd.setup()
    assert os.environ["OMP_NUM_THREADS"] == "16"


def test_setup_defaults_omp_threads_to_one(calls, monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    rd.setup()
    assert os.environ["OMP_NUM_THREADS"] == "1"


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


def test_late_call_with_different_threads_raises(calls, monkeypatch):
    # already-loaded libraries keep their thread pools, so a late thread
    # change cannot take effect: it must raise, not silently no-op
    monkeypatch.setattr(_setup, "_backends_are_initialized", lambda: True)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(jax, "local_device_count", lambda: 4)
    with pytest.raises(RuntimeError, match="before any jax operation"):
        rd.setup(x64=bool(jax.config.jax_enable_x64), num_threads=8)
    # nothing may have been reconfigured, the environment included
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert calls["platform"] == []
    assert calls["devices"] == []
    assert calls["config"] == []


def test_late_call_with_matching_threads_is_noop(calls, monkeypatch):
    monkeypatch.setattr(_setup, "_backends_are_initialized", lambda: True)
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(jax, "local_device_count", lambda: 4)
    rd.setup(x64=bool(jax.config.jax_enable_x64), num_threads=1)
    assert calls["platform"] == []
    assert calls["devices"] == []
    assert calls["config"] == []


def test_late_call_on_multi_gpu_machine_is_noop(calls, monkeypatch):
    # re-running setup(platform='gpu') on a 2-GPU machine must not raise:
    # the device count is not controlled on accelerators, so it plays no
    # part in the match
    monkeypatch.setattr(_setup, "_backends_are_initialized", lambda: True)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jax, "local_device_count", lambda: 2)
    rd.setup(platform="gpu", x64=bool(jax.config.jax_enable_x64))
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


def _import_order_warnings(env, prelude):
    """Run `prelude` then `import ringdown` in a fresh interpreter and
    return how many import-order RuntimeWarnings the import raised."""
    code = (
        "import warnings\n"
        f"{prelude}\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import ringdown\n"
        "hits = [w for w in caught\n"
        "        if issubclass(w.category, RuntimeWarning)\n"
        "        and 'before ringdown was imported' in str(w.message)]\n"
        "print(len(hits))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def test_import_after_jax_operation_warns():
    # a jax operation initializes the backends, freezing the threading
    # configuration before the OMP default can land
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    prelude = "import jax.numpy as jnp; jnp.zeros(1) + 1"
    assert _import_order_warnings(env, prelude) == 1


def test_import_after_idle_jax_does_not_warn():
    # merely importing jax does not initialize its backends (lazy), so
    # the OMP default still lands in time
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    assert _import_order_warnings(env, "import jax") == 0


def test_import_after_jax_operation_with_omp_set_does_not_warn():
    # with OMP_NUM_THREADS exported, the default is moot but harmless
    env = dict(os.environ, OMP_NUM_THREADS="1")
    prelude = "import jax.numpy as jnp; jnp.zeros(1) + 1"
    assert _import_order_warnings(env, prelude) == 0
