Configuration files
===================

The executables shipped with `ringdown` (e.g., :doc:`ringdown_fit
<exe_ringdown_fit>`) are driven by INI-style configuration files, parsed with
:mod:`configparser` and consumed by :meth:`Fit.from_config
<ringdown.fit.Fit.from_config>`. The same files can be used directly from
Python::

    import ringdown as rd
    fit = rd.Fit.from_config("config.ini")

Complete example files (``ringdown_fit_example.ini``,
``ringdown_fit_fake_example.ini``, ``ringdown_pipe_example.ini``,
``ringdown_pp_pipe_example.ini``) can be found in the ``etc/`` directory of
the `source repository <https://github.com/maxisi/ringdown/tree/main/etc>`_.

Value parsing
-------------

Option values are strings that :meth:`Fit.from_config
<ringdown.fit.Fit.from_config>` parses with
:func:`ringdown.utils.try_parse`: each value is interpreted as a float if
possible, otherwise through :func:`ast.literal_eval` (so booleans, tuples,
lists and dictionaries can be written as Python literals, e.g. ``modes = (1,
-2, 2, 2, 0),(1, -2, 2, 2, 1)``); the special value ``inf`` becomes
``np.inf``; anything else is kept as a plain (unquoted) string. The ``ifos``
option can be a Python list or a comma-separated string (e.g. ``ifos =
H1,L1``). Options defined in a ``[DEFAULT]`` section are only used for
:mod:`configparser` interpolation into other options and are not themselves
passed to any method.

Sections
--------

Most sections simply collect keyword arguments that are forwarded to a
:class:`Fit <ringdown.fit.Fit>` method; see the linked methods for the
available options.

``[model]``
    Options passed to the :class:`Fit <ringdown.fit.Fit>` constructor: the
    ``modes`` specification plus any model settings accepted by
    :meth:`Fit.update_model <ringdown.fit.Fit.update_model>` (i.e., arguments
    of :func:`ringdown.model.make_model`).

``[prior]``
    Merged into the ``[model]`` options; takes any option accepted by
    :meth:`Fit.update_model <ringdown.fit.Fit.update_model>` (e.g.,
    ``a_scale_max``, ``m_min``, ``m_max``). The split into two sections is
    purely organizational.

``[target]``
    Options passed to :meth:`Fit.set_target <ringdown.fit.Fit.set_target>`
    (e.g., ``t0``, ``ra``, ``dec``, ``psi``, ``duration``). A section
    specifying ``ra`` but no ``t0`` is ignored by :meth:`Fit.from_config
    <ringdown.fit.Fit.from_config>` (such configs define targets for a fit
    sequence, as produced by ``ringdown_pipe``).

``[data]``
    Options passed to :meth:`Fit.load_data <ringdown.fit.Fit.load_data>`
    (e.g., ``ifos``, ``path``, ``kind``, ``channel``). If neither ``[data]``
    nor ``[fake-data]`` is present, parsing stops after the model and IMR
    setup and the remaining sections are ignored.

``[fake-data]``
    Options passed to :meth:`Fit.fake_data <ringdown.fit.Fit.fake_data>` to
    generate synthetic data instead of loading strain from disk.

``[injection]``
    Options passed to :meth:`Fit.inject <ringdown.fit.Fit.inject>` to add a
    simulated signal. A ``path`` option points to a JSON file with injection
    parameters, which are merged with (and overridden by) the options in this
    section. The ``no_noise`` and ``post_cond`` flags control whether the
    injection replaces the data entirely and whether it is added after
    conditioning.

``[condition]``
    Options passed to :meth:`Fit.condition_data
    <ringdown.fit.Fit.condition_data>` (e.g., ``ds``, ``f_min``); skipped if
    :meth:`Fit.from_config <ringdown.fit.Fit.from_config>` is called with
    ``no_cond=True``.

``[acf]``
    If a ``path`` (or ``from_imr_result``) option is given, options are passed
    to :meth:`Fit.load_acfs <ringdown.fit.Fit.load_acfs>`; otherwise, they are
    passed to :meth:`Fit.compute_acfs <ringdown.fit.Fit.compute_acfs>` to
    estimate ACFs from the data.

``[imr]``
    Options to load a reference inspiral-merger-ringdown (IMR) posterior. If
    ``initialize_fit = True``, the fit is constructed with
    :meth:`Fit.from_imr_result <ringdown.fit.Fit.from_imr_result>` from the
    result at ``path`` (or ``imr_result``), automatically deriving settings
    like the target from the IMR posterior; this requires an integer random
    seed given by the ``prng`` (or ``seed``) option, used to subselect IMR
    samples. Otherwise, the result is attached as a reference via
    :meth:`Fit.add_imr_result <ringdown.fit.Fit.add_imr_result>`. Remaining
    options are forwarded to those methods.

In addition, the executables read sections that are not consumed by
:meth:`Fit.from_config <ringdown.fit.Fit.from_config>` itself: a ``[run]``
section with sampler options forwarded to :meth:`Fit.run
<ringdown.fit.Fit.run>` (read by :doc:`ringdown_fit <exe_ringdown_fit>` and
:doc:`ringdown_scan <exe_ringdown_scan>`), and a ``[pipe]`` section defining
the set of analysis start times and output location for :doc:`ringdown_pipe
<exe_ringdown_pipe>` and :doc:`ringdown_pp_pipe <exe_ringdown_pp_pipe>`.
