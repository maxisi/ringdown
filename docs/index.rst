.. ringdown documentation master file, created by
   sphinx-quickstart on Mon Dec  6 15:33:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ringdown
========

**ringdown** is a Python package for the Bayesian analysis of black hole ringdowns in data from gravitational-wave detectors, like `LIGO <https://www.ligo.caltech.edu>`_ and `Virgo <https://www.virgo-gw.eu>`_.

`ringdown` contains basic utilities to manipulate strain data and easily set up your own inference analyses quickly. The current implementation relies on `numpyro <https://num.pyro.ai>`_ for sampling multidimensional posterior probability densities.

This documentation is being actively developed. Usage examples can be found :doc:`here <gallery>` and API documentation in the :ref:`modindex`.

.. image:: https://badge.fury.io/py/ringdown.svg
    :target: https://badge.fury.io/py/ringdown
.. image:: https://img.shields.io/badge/GitHub-ringdown-blue
    :target: https://github.com/maxisi/ringdown
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5094067.svg
   :target: https://doi.org/10.5281/zenodo.5094067

Example
-------

You can get an analysis going with just a few lines of code. For example, assuming you have already downloaded GW150914 data from `GWOSC <https://www.gw-openscience.org>`_::

    import ringdown as rd
    fit = rd.Fit(modes=[(1,-2,2,2,0), (1,-2,2,2,1)])
    fit.load_data('{i}-{i}1_GWOSC_16KHZ_R1-1126259447-32.hdf5', ifos=['H1', 'L1'], kind='gwosc')
    fit.set_target(1126259462.4083147, ra=1.95, dec=-1.27, psi=0.82, duration=0.05)
    fit.condition_data(ds=8)
    fit.update_prior(a_scale_max=1e-21, m_min=50, m_max=150)
    fit.run()

Or run using a configuration file from the :doc:`command line <exe_ringdown_fit>`:

.. code-block:: console

    ringdown_fit config.ini

See full-fledged examples in the :doc:`gallery`.


How to cite
-----------

We ask that scientific users of this code cite the corresponding Zenodo entry (see blue DOI badge above), as well as `Isi & Farr (2021) <https://arxiv.org/abs/2107.05609>`_:

.. code-block:: bash

  @article{Isi:2021iql,
      author = "Isi, Maximiliano and Farr, Will M.",
      title = "{Analyzing black-hole ringdowns}",
      eprint = "2107.05609",
      archivePrefix = "arXiv",
      primaryClass = "gr-qc",
      reportNumber = "LIGO-P2100227",
      month = "7",
      year = "2021"
  }

Contents
--------

.. toctree::
   :maxdepth: 2

   overview
   models
   gallery
   exe_ringdown_fit
   modules

Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
