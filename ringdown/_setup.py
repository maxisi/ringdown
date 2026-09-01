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
    num_threads: int | None = None,
):
    """Configure jax and numpyro for a ringdown analysis.

    Sets the compute platform, the number of devices available for
    parallel chains, the floating-point precision and the number of
    BLAS/OpenMP threads, replacing the manual boilerplate of environment
    variables and :mod:`numpyro` calls. Call it right after importing
    :mod:`ringdown`, before any jax operation runs::

        import ringdown as rd
        rd.setup()                    # CPU, float64, 4 host devices
        rd.setup(platform='gpu')      # GPU, float32, visible devices

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
        number of CPU host devices to make available, e.g., to run
        chains in parallel; on 'cpu' this many host devices are created
        out of the machine's cores (clamped to the CPU count). Defaults
        to the ``RINGDOWN_DEVICE_COUNT`` environment variable if set,
        otherwise 4. This only applies to the CPU platform: on GPU/TPU
        the device count is not controlled here (device visibility is
        environment-controlled, e.g., via ``CUDA_VISIBLE_DEVICES``), so
        an explicit value is ignored with a warning.
    x64 : bool, optional
        enable double precision; defaults to true on CPU and false on
        GPU, the recommended settings (see the performance
        documentation).
    num_threads : int, optional
        value for ``OMP_NUM_THREADS``, capping the threads used by each
        BLAS/OpenMP operation. Defaults to the value already in the
        environment (importing ringdown sets it to 1 unless it was
        already exported), so an exported ``OMP_NUM_THREADS`` is
        respected. Note that BLAS/OpenMP libraries read this variable
        when they load (during ``import ringdown``), so an explicit
        value passed here cannot re-thread libraries that are already
        loaded: it only affects whatever reads the environment
        afterwards, such as XLA at backend initialization or spawned
        subprocesses. A late (post-initialization) call never writes
        the environment: if the requested thread count differs from
        the active ``OMP_NUM_THREADS`` it raises like any other
        mismatch, since it could no longer take effect.
    """
    import jax
    import numpyro

    platform = str(platform).lower()
    is_gpu = platform in _GPU_ALIASES
    # numpyro 0.21 rejects 'gpu'; its CUDA name is 'cuda'
    numpyro_platform = "cuda" if platform == "gpu" else platform

    is_cpu = platform == "cpu"

    # the device count is only controllable on the CPU host platform:
    # numpyro.set_host_device_count() sets XLA_FLAGS=
    # --xla_force_host_platform_device_count, which GPU/TPU backends ignore
    if is_cpu:
        if num_devices is None:
            if "RINGDOWN_DEVICE_COUNT" in os.environ:
                num_devices = int(os.environ["RINGDOWN_DEVICE_COUNT"])
            else:
                num_devices = 4
        cpu_count = os.cpu_count()
        if cpu_count and num_devices > cpu_count:
            logger.warning(
                f"requested device count ({num_devices}) "
                "greater than the number of available CPUs. "
                "Setting it to the maximum number of CPUs "
                f"({cpu_count})."
            )
            num_devices = cpu_count
    elif num_devices is not None:
        logger.warning(
            "num_devices is not controlled on accelerator platforms "
            f"(requested {num_devices} on '{platform}'): visible devices "
            "are selected via the environment, e.g., CUDA_VISIBLE_DEVICES; "
            "ignoring it."
        )
        num_devices = None

    if x64 is None:
        x64 = not is_gpu

    if num_threads is None:
        # respect an exported OMP_NUM_THREADS (or the import-time default)
        num_threads = os.environ.get("OMP_NUM_THREADS", "1")

    if _backends_are_initialized():
        active_platform = jax.default_backend()
        # if the variable was unset since import there is nothing to
        # compare against, so treat the thread count as matching
        active_threads = os.environ.get("OMP_NUM_THREADS", str(num_threads))
        matches = (
            (active_platform in _GPU_ALIASES) == is_gpu
            and (is_gpu or active_platform == platform)
            # the device count is only controlled on CPU, so only there
            # can a mismatch mean a stale configuration
            and (not is_cpu or jax.local_device_count() == num_devices)
            and bool(jax.config.jax_enable_x64) == bool(x64)
            # already-loaded libraries keep their thread pools, so a
            # different thread count could not take effect either
            and str(num_threads) == active_threads
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
            f"x64={bool(jax.config.jax_enable_x64)}, "
            f"OMP_NUM_THREADS={active_threads}; requested: "
            f"platform={platform}, num_devices={num_devices}, x64={x64}, "
            f"num_threads={num_threads}."
        )

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    numpyro.set_platform(numpyro_platform)
    if is_cpu:
        numpyro.set_host_device_count(num_devices)
    jax.config.update("jax_enable_x64", bool(x64))

    logger.info(
        f"configured jax: platform={platform}, num_devices={num_devices}, "
        f"x64={x64}, OMP_NUM_THREADS={num_threads}"
    )
