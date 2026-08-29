ringdown_pp_pipe
================

Sets up a probability-probability (PP) test: a collection of ringdown fits of
synthetic injections drawn from the prior, based on a common configuration
file (see :doc:`config_file`), together with SLURM submission files.

.. argparse::
   :filename: ../ringdown/cli/ringdown_pp_pipe.py
   :func: get_parser
   :prog: ringdown_pp_pipe

An example configuration file, ``ringdown_pp_pipe_example.ini``, can be found
in the ``etc/`` directory of the source repository.
