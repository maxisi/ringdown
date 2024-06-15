Models
======

The :class:`Fit <ringdown.fit.Fit>` object interfaces with a of `numpyro
<https://num.pyro.ai>`_ model for sampling. This model defines the likelihood,
prior and signal templates and is constructed according to a number of options
per the :ref:`modindex`. The model can run in three broad configurations.

Generic damped sinusoids
------------------------

Fit an arbitrary number of damped sinusoids parameterized in terms of their
amplitudes :math:`A_n`, phases :math:`\phi_n`, frequencies :math:`f_n` and
damping rates :math:`\gamma_n = 1/\tau_n`, where :math:`\tau_n` is the damping
time. The signal model is thus

.. math::
   h(t) = \sum_n A_n \cos(2\pi f_n t - \phi_n) \exp(-\gamma_n t)

although this is internally parameterized in terms of cosine and sine
quadratures, :math:`A_x = A \cos \phi` and :math:`A_y = A \sin \phi`, for
efficiency. Importantly, label switching problems are avoided by defining
:math:`\f_n < \f_{n+1}` or :math:`\gamma_n < \gamma_{n+1}`.

Priors are flat in :math:`A_n, \phi_n, f_n, \gamma_n`, modulo the
:math:`\f_n < \f_{n+1}` or :math:`\gamma_n < \gamma_{n+1}` restriction.

|:point_right:| **See this model in action!** :doc:`examples/single_damped_sinusoid`.

Kerr ringdowns
--------------

Fit an arbitrary number of Kerr ringdown modes with arbitrary elliptical
polarization as in `Isi & Farr (2021) <https://arxiv.org/abs/2107.05609>`_.
Modes are specified by their :math:`(p, s, \ell, |m|, n)` numbers: prograde vs
retrograde (:math:`p = \pm 1`), spin-weight (:math:`s = -2` for GWs), angular
harmonic numbers (:math:`\ell` and :math:`0 \leq |m| \leq \ell`), and overtone
number (:math:`n`).

The waveform is such that the two GW polarizations for each
:math:`j \equiv (+1, -2, \ell, |m|, n)` mode are given by six
parameters :math:`A_j, \epsilon_j, \theta_j,\phi_j` following

.. math::
   \begin{eqnarray}
   h^{(+)}_{j} &= h^c_{j}\, \cos \theta_{j} - \epsilon_{j} h^s_{j}\, \sin\theta_{j}\, , \\
   h^{(\times)}_j &= h^c_{j}\, \sin \theta_j + \epsilon_j h^s_{j}\, \cos\theta_j\, ,
   \end{eqnarray}

for cosine and sine quadratures

.. math::
   \begin{eqnarray}
   h^c_j &\equiv A_j\, e^{-t/\tau_j} \cos(\omega_j t - \phi_j) \, , \\
   h^s_j &\equiv A_j\, e^{-t/\tau_j} \sin(\omega_j t - \phi_j) \, .
   \end{eqnarray}

For each detector the template will thus be

.. math::
   h = \sum_j \left( F_+ h^{(+)}_{j} + F_\times h^{(\times)}_{j} \right)\, ,

summing over the requested mode indices :math:`j`. The antenna patterns
:math:`(F_+, F_\times)` are determined automatically by the
:class:`Fit<ringdown.fit.Fit>` object based on the target sky location and polarization
angle; these are currently fixed, and their only effect is to scale the relative
amplitudes at different detectors (otherwise, they are degenerate with the mode
amplitudes and phases).

In the ``mchi`` model, the mode frequencies and damping rates are parameterized
by two parameters: the Kerr black-hole mass :math:`M` and dimensionless spin
magnitude :math:`\chi`. To replicate this functional dependence efficiently, the
model makes use of fitting coefficients precomputed through the `qnm
<https://qnm.readthedocs.io/en/latest/>`_ package.

The priors are uniform in :math:`M` and :math:`\chi`. The priors can also be
made uniform on :math:`A_j` and :math:`\epsilon_j` using the ``flat_A`` and
``flat_A_ellip`` options (see :meth:`Fit.update_prior
<ringdown.fit.Fit.update_prior>`); by default, however, they correspond to
Gaussian priors on the cosine and sine quadratures of each polarization (see
Appendix of `Isi & Farr (2021) <https://arxiv.org/abs/2107.05609>`_).

This model supports deviations from the Kerr spectrum, which can be turned on
via the ``df`` and ``dg`` options. This activates deviation
parameters :math:`\delta f_j` and :math:`\delta\gamma_j` that modify the
frequencies and damping times such that

.. math::
   \begin{eqnarray}
   f_j &= f_j(M,\chi) \exp(\delta f_j) \, , \\
   \gamma_j &= \gamma_j(M,\chi) \exp(\delta \gamma_j) \, .
   \end{eqnarray}

|:point_right:| **See this model in action!** :doc:`examples/GW150914`.

Kerr ringdowns with restricted polarizations
--------------------------------------------

Fit an arbitrary number of Kerr overtones with
polarizations parameterized by a single "inclination" parameter
:math:`\cos\iota`, controlling the degree of circular polarization for all
modes. This is equivalent to assuming all :math:`m=+2` and :math:`m=-2`
components are equally excited, so that the ellipticity of the observed signal
is only a function of the viewing angle (see appendix in `Isi & Farr (2021)
<https://arxiv.org/abs/2107.05609>`_); we might expect this in the case of
nonprecessing systems, which possess equatorial reflection symmetry (hence the
naming ``aligned``).

In this more restricted version of the mass-spin model above, the polarizations are given by

.. math::
   \begin{eqnarray}
   h^{(+)} &= \sum_{\ell m n} Y^+_{\ell |m| n}(\cos\iota) A_{\ell |m| n} \cos(\omega_{\ell |m| n} t - \phi__{\ell |m| n})
   \exp(-t \gamma_{\ell |m| n})\, , \\
   h^{(\times)} &= \sum_{\ell |m| n} A_{\ell |m| n} Y^\times_{\ell |m| n}(\cos\iota) \sin(\omega_{\ell |m| n} t - \phi_{\ell |m| n}) \exp(-t/\tau_{\ell |m| n})\, .
   \end{eqnarray}

where :math:`Y^{+/\times}_{\ell |m| n}(\cos\iota)` are given in Eq. (31) of
`Isi (2022) <https://arxiv.org/abs/2208.03372>`_. Other options for this model
are analogous to those in the generic mass-spin configuration.
