ringdown_scan
=============

Runs a sequence of ringdown fits of an event at different start times from a
single configuration file (see :doc:`config_file`), executing the fits
directly rather than setting up batch jobs like :doc:`ringdown_pipe
<exe_ringdown_pipe>`.

.. argparse::
   :filename: ../ringdown/cli/ringdown_scan.py
   :func: get_parser
   :prog: ringdown_scan
