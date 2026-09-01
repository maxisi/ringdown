Configuration
=============

*ringdown* configures jax and numpyro for you through a single call. To run on a
CPU with four host devices (so chains can sample in parallel) and double
precision:

.. code-block:: python

   import ringdown as rd
   rd.setup()

To run on a GPU with single precision instead:

.. code-block:: python

   import ringdown as rd
   rd.setup(platform='gpu')

:func:`ringdown.setup` sets the platform (``numpyro.set_platform``), the CPU
host device count (``numpyro.set_host_device_count``; the ``num_devices``
argument defaults to the ``RINGDOWN_DEVICE_COUNT`` environment variable, or 4
— on GPU/TPU the device count is not controlled here: visible devices are
selected through the environment, e.g., ``CUDA_VISIBLE_DEVICES``) and the
precision (``x64`` defaults to true on CPU and false on GPU). All
of these settings freeze as soon as jax initializes its backends, and jax and
numpyro silently ignore later changes — so call it right after importing
*ringdown*, before any jax operation; ``setup`` raises if called too late with
a configuration different from the active one.

In addition, importing *ringdown* caps BLAS/OpenMP threading by setting
``OMP_NUM_THREADS=1`` (unless the variable is already set), so that parallel
chains do not oversubscribe the machine with one thread pool per core each. To
override, export ``OMP_NUM_THREADS`` before importing *ringdown*, or pass
``num_threads`` to ``setup``.

.. autofunction:: ringdown.setup


The best choice of hardware and precision depends on the size of your analysis; see
`Performance`_ below. For a typical single fit a multi-core CPU in double precision is
usually the fastest option, while GPUs pay off for large analyses or for running many
chains at once. If you have multiple GPUs, ``numpyro`` can use them in parallel to run
different chains, just as with CPUs.

.. warning::

   Caveat emptor: depending on the autocovariance function (ACF), using ``float32`` can cause numerical problems when computing the likelihood. In physical units the strain is of order :math:`10^{-22}`, and intermediate quantities in the likelihood span a far wider dynamic range than ``float32`` can represent (its ceiling is around :math:`10^{38}`), so the computation can overflow to ``NaN`` even when the problem is perfectly well conditioned. *ringdown* will automatically rescale the strain (``strain_scale="auto"``) in an attempt to prevent this, but you should use this feature at your own risk.


Performance
-----------

The cost of a fit is dominated by the marginalized likelihood, which is built from
triangular solves on matrices of size ``n_analyze``. For a typical analysis these matrices
are small (a few hundred samples), so the computation is limited by kernel latency rather
than by arithmetic throughput, and a single GPU cannot be kept busy by a single chain.

**A single fit at typical size** (a few hundred analysis samples, two detectors, two
modes) runs fastest on a multi-core CPU in ``float64``. Comparing each platform in its
best configuration on the same machine, one GPU was slower than the CPU by roughly 2x on
an NVIDIA H100 and roughly 3x on an RTX A6000; letting the GPU use ``float32`` narrows the
gap to about 2x on the H100 but does not reverse it. This ranking was insensitive to the
details of the likelihood implementation.

GPUs become worthwhile in two regimes:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Analysis
     - Suggested setup
   * - Single fit, few hundred samples, up to ~3 modes
     - CPU, ``float64``, chains across host devices
   * - Many chains of the same fit
     - CPU across host devices up to a few tens of chains; beyond that, a GPU with
       vectorized chains (crossover: 4-16 chains on an H100, 16-64 on an A6000)
   * - Large analyses (``n_analyze`` ≳ 1000 **and** ≳ 4 modes)
     - GPU; measured 2-4x faster than the same-node CPU

Running many chains
   The comparison that matters is a GPU batching its chains against a CPU running them in
   parallel across host devices, as configured above. At the default four chains and typical
   size the CPU wins outright: about 2x faster than an H100 and 3x faster than an A6000, in
   either GPU precision. The crossover only arrives in the tens of chains, measured between 4
   and 16 chains on an H100 and between 16 and 64 on an A6000. The exception is the
   large-analysis row above, where the GPU already leads per chain and so wins at any chain
   count, four included; that extrapolates the per-gradient measurements, since chain scaling
   itself was measured only at typical size.

   On GPU, *ringdown* selects ``chain_method='vectorized'`` automatically whenever
   ``num_chains`` exceeds the number of visible devices, batching the chains into a single
   computation; the logic lives in ``get_sampling_kwargs`` in ``ringdown/fit.py``.
   (``numpyro``'s own ``'parallel'`` default instead finds one device and falls back to
   drawing them **sequentially**.) To override, or when driving ``numpyro`` directly:

   .. code-block:: ini

      [run]
      num_chains = 64
      chain_method = 'vectorized'

   Measured sampling throughput relative to a single chain on the same GPU was about 2-3x at
   4 chains, 6-7x at 16 and about 20x at 64.

Precision
   The value of ``float32`` depends on the class of GPU. Consumer and workstation cards
   have heavily reduced double-precision throughput (1:32 on workstation Ampere cards such
   as the RTX A6000), and there ``float32`` was roughly 2-3x faster and is what makes the
   card usable at all. Datacenter cards (A100, H100) run ``float64`` at 1:2, and
   ``float32`` bought only about 1.1x at typical analysis sizes, so prefer ``float64``
   there and keep the extra precision. Note also that ``float32`` degrades gradients far
   more than it degrades the log-likelihood itself, which is what the sampler is sensitive
   to. On CPU, always use ``float64``.

Threads
   The ``OMP_NUM_THREADS=1`` setting shown above measured roughly 1.5x slower per gradient
   than leaving threading unrestricted, since a single chain no longer spreads across
   cores. In exchange it lets ``numpyro.set_host_device_count`` run the chains in parallel
   without oversubscribing the machine, which is the better trade at the default four
   chains.
