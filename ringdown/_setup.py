"""One-call configuration of jax and numpyro for ringdown analyses."""

__all__ = ["setup"]

import logging
import os

logger = logging.getLogger(__name__)

# platform aliases: jax reports the CUDA/ROCm backends as 'gpu', while
# numpyro's set_platform accepts 'cuda'/'rocm' but not 'gpu'
_GPU_ALIASES = ("gpu", "cuda", "rocm")


def _backends_are_initialized() -> bool:
    """Whether jax has already initialized its backends, freezing the
    platform, device count and threading configuration."""
    try:
        from jax._src.xla_bridge import backends_are_initialized
    except ImportError:
        # private API: if a future jax moves it, skip the guard
        return False
    return backends_are_initialized()


def setup(
    platform: str = "cpu",
    num_devices: int | None = None,
    x64: bool | None = None,
    num_threads: int = 1,
):
    """Configure jax and numpyro for a ringdown analysis.

    Sets the compute platform, the number of devices available for
    parallel chains, the floating-point precision and the number of
    BLAS/OpenMP threads, replacing the manual boilerplate of environment
    variables and :mod:`numpyro` calls. Call it right after importing
    :mod:`ringdown`, before any jax operation runs::

        import ringdown as rd
        rd.setup()                    # CPU, float64, 4 host devices
        rd.setup(platform='gpu')      # GPU, float32, 1 device

    All of these settings freeze when jax initializes its backends (at
    the first jax operation) and fail silently afterwards, so this
    function raises if called too late with a configuration different
    from the active one; a late call that matches the active
    configuration is a no-op, so re-running a notebook cell is harmless.

    Arguments
    ---------
    platform : str
        compute platform: 'cpu' or 'gpu' (also accepts 'cuda', 'rocm' or
        'tpu'); defaults to 'cpu'.
    num_devices : int, optional
        number of devices to make available, e.g., to run chains in
        parallel; on 'cpu' this many host devices are created out of the
        machine's cores (clamped to the CPU count). Defaults to the
        ``RINGDOWN_DEVICE_COUNT`` environment variable if set, otherwise
        4 on CPU and 1 on GPU.
    x64 : bool, optional
        enable double precision; defaults to true on CPU and false on
        GPU, the recommended settings (see the performance
        documentation).
    num_threads : int
        value for ``OMP_NUM_THREADS``, capping the threads used by each
        BLAS/OpenMP operation; the default of 1 avoids oversubscribing
        the machine when running parallel chains (importing ringdown
        already sets this default; the explicit argument overrides it).
    """
    import jax
    import numpyro

    platform = str(platform).lower()
    is_gpu = platform in _GPU_ALIASES
    # numpyro 0.21 rejects 'gpu'; its CUDA name is 'cuda'
    numpyro_platform = "cuda" if platform == "gpu" else platform

    if num_devices is None:
        if "RINGDOWN_DEVICE_COUNT" in os.environ:
            num_devices = int(os.environ["RINGDOWN_DEVICE_COUNT"])
        else:
            num_devices = 1 if is_gpu else 4

    cpu_count = os.cpu_count()
    if platform == "cpu" and num_devices > cpu_count:
        logging.warning(
            f"requested device count ({num_devices}) "
            "greater than the number of available CPUs. "
            "Setting it to the maximum number of CPUs "
            f"({cpu_count})."
        )
        num_devices = cpu_count

    if x64 is None:
        x64 = not is_gpu

    if _backends_are_initialized():
        active_platform = jax.default_backend()
        matches = (
            (active_platform in _GPU_ALIASES) == is_gpu
            and (is_gpu or active_platform == platform)
            and jax.local_device_count() == num_devices
            and bool(jax.config.jax_enable_x64) == bool(x64)
        )
        if matches:
            logger.info(
                "jax already initialized with the requested configuration"
            )
            return
        raise RuntimeError(
            "jax has already initialized its backends, so the requested "
            "configuration cannot take effect (jax and numpyro would "
            "silently ignore it). Call ringdown.setup() before any jax "
            "operation, right after importing ringdown. Active "
            f"configuration: platform={active_platform}, "
            f"num_devices={jax.local_device_count()}, "
            f"x64={bool(jax.config.jax_enable_x64)}; requested: "
            f"platform={platform}, num_devices={num_devices}, x64={x64}."
        )

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    numpyro.set_platform(numpyro_platform)
    numpyro.set_host_device_count(num_devices)
    jax.config.update("jax_enable_x64", bool(x64))

    logger.info(
        f"configured jax: platform={platform}, num_devices={num_devices}, "
        f"x64={x64}, OMP_NUM_THREADS={num_threads}"
    )
