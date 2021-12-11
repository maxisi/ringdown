.. ringdown documentation master file, created by
   sphinx-quickstart on Mon Dec  6 15:33:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ringdown
========

**ringdown** is a Python-based package for the Bayesian analysis of black hole ringdowns in data from gravitational-wave detectors, like `LIGO <https://www.ligo.caltech.edu>`_ and `Virgo <https://www.virgo-gw.eu>`_.

ringdown contains basic utilities to manipulate strain data and easily set up your own inference analyses. The current implementation relies on `Stan <https://mc-stan.org>`_ for sampling.

This documentation is being actively developed. In the meantime, usage examples can be found `here <https://github.com/maxisi/ringdown/tree/main/examples>`_ and API documentation in the :ref:`modindex`.

.. image:: https://badge.fury.io/py/ringdown.svg
    :target: https://badge.fury.io/py/ringdown
.. image:: https://img.shields.io/badge/GitHub-ringdown-blue
    :target: https://github.com/maxisi/ringdown
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5094067.svg
   :target: https://doi.org/10.5281/zenodo.5094067


How to cite
===========

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

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
