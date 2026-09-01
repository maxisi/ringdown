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

:func:`ringdown.setup` sets the platform (``numpyro.set_platform``), the device
count (``numpyro.set_host_device_count``; the ``num_devices`` argument defaults
to the ``RINGDOWN_DEVICE_COUNT`` environment variable, or 4 on CPU and 1 on
GPU) and the precision (``x64`` defaults to true on CPU and false on GPU). All
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


You will see significant performance enhancements when running on a GPU with 32-bit precision. If you have multiple GPUs, ``numpyro`` can use them in parallel to run different chains, just as with CPUs. Sampling one chain for a GW150914-like system takes O(s) on an Nvidia A100 GPU.

.. warning::

   Caveat emptor: depending on the autocovariance function (ACF), using ``float32`` can cause numerical problems when computing the likelihood; *ringdown* will automatically rescale the strain in an attempt to prevent this, but you should use this feature at your own risk.
