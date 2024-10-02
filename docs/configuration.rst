Configuration
===========

In order to run Jax on a CPU with four cores and use double precision, you can do the following:

.. code-block:: python

   # disable numpy multithreading to avoid conflicts
   # with jax multiprocessing in numpyro
   import os
   os.environ["OMP_NUM_THREADS"] = "1"

   # import jax and set it up to use double precision
   from jax import config
   config.update("jax_enable_x64", True)

   # import numpyro and set it up to use 4 CPU devices
   import numpyro
   numpyro.set_host_device_count(4)
   numpyro.set_platform('cpu')


To run on a GPU with single precision you can instead do:

.. code-block:: python

   # import jax and set it up to use double precision
   from jax import config
   config.update("jax_enable_x64", False)

   # import numpyro and set it up to use 4 CPU devices
   import numpyro
   numpyro.set_platform('gpu')


You will see significant performance enhancements when running on a GPU with 32-bit precision. If you have multiple GPUs, ``numpyro`` can use them in parallel to run different chains, just as with CPUs. Sampling one chain for a GW150914-like system takes O(s) on an Nvidia A100 GPU.

.. warning::

   Caveat emptor: depending on the autocovariance function (ACF), using ``float32`` can cause numerical problems when computing the likelihood; *ringdown* will automatically rescale the strain in an attempt to prevent this, but you should use this feature at your own risk.
