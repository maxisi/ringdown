ringdown_pipe
=============

Sets up a pipeline of ringdown fits over a series of analysis start times,
creating one run directory per target time from a common configuration file
(see :doc:`config_file`, in particular the ``[pipe]`` section), together with
SLURM submission files.

.. argparse::
   :filename: ../ringdown/cli/ringdown_pipe.py
   :func: get_parser
   :prog: ringdown_pipe

An example configuration file, ``ringdown_pipe_example.ini``, can be found in
the ``etc/`` directory of the source repository.
