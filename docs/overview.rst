ringdown overview
========

The `ringdown` package is composed of three modules: :mod:`ringdown.fit`, :mod:`ringdown.data` and :mod:`ringdown.waveforms`. The single most important object in the package is the :class:`ringdown.fit.Fit` class, which contains all information and settings required to set up and execute an analysis. Several examples of its use are provided in the :ref:`gallery`.

The :class:`ringdown.fit.Fit` object interfaces with a number of `Stan <https://mc-stan.org>`_ models for sampling. These models define the likelihood, prior and signal templates, and are contained in text files shipped with `ringdown` and accessed internally. The available models are described in :ref:`models`. For more information about sampling with Stan, or Hamiltonian Monte-Carlo in general, see the Stan docs.

Auxiliary to :mod:`ringdown.fit`, the :mod:`ringdown.data` and :mod:`ringdown.waveforms` modules contain utilities for handling strain data (time and frequency series, power spectral densities, autocovariance functions) and templates (both ringdown-only and inspiral-merger-ringdown). This functionality is intentionally minimal; many other widely-used packages exist for generic manipulation of strain data, e.g., `GWpy <https://gwpy.github.io>`_, `PyCBC <https://pycbc.org>`_.
Like those packages, `ringdown` relies heavily on `LALSuite <https://pypi.org/project/lalsuite/>`_ for waveform handling and related operations.

The above functionality can be used to build your own workflows. Additionally, `ringdown` ships with a single executable `ringdown_fit` which can be used to automatically set up and run a ringdown fit starting from a configuration file. This can easily be used to create automated workflows if desired.
