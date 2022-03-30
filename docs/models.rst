Models
======

The :class:`ringdown.fit.Fit` object interfaces with a number of `Stan <https://mc-stan.org>`_ models for sampling. These models define the likelihood, prior and signal templates, and are contained in text files shipped with `ringdown` and accessed internally. The available models are described below.

ftau
----

Fit an arbitrary number of damped sinusoids parameterized in terms of their amplitudes :math:`A_n`, phases :math:`phi_n`, frequencies :math:`f_n` and damping rates :math:`\gamma_n = 1/\tau_n`, where :math:`\tau_n` is the damping time. The signal model is thus

.. math::
   h(t) = \sum_n A_n \cos(2\pi f_n t - \phi) \exp(-\gamma_n t)

although this is internally parameterized in terms of cosine and sine quadratures, :math:`A_x = A \cos \phi` and :math:`A_y = A \sin \phi`, for efficiency. Importantly, label switching problems are avoided by defining :math:`\gamma_n < \gamma_{n+1}`.

.. warning::
   This model is not currently suitable for the analysis of real data. The
   `ftau` model currently has no notion of  polarizations, so it cannot support
   multiple detectors. Additionally, the :math:`\gamma_n < \gamma_{n+1}` can
   make it innefficient for cases in which :math:`\gamma_n \approx
   \gamma_{n+1}` (as would be the case, for example, for different angular
   harmonics).

Priors are flat in :math:`A_n, \phi_n, f_n, \tau_n`, modulo the :math:`\gamma_n < \gamma_{n+1}` restriction.

mchi
----

Fit an arbitrary number of Kerr ringdown modes with arbitrary elliptical polarization as in `Isi & Farr (2021) <https://arxiv.org/abs/2107.05609>`_.
Modes are specified by their :math:`(p, s, \ell, |m|, n)` numbers: prograde vs retrograde (:math:`p = \pm 1`), spin-weight (:math:`s = -2` for GWs), angular harmonic numbers :math:`\ell` and :math:`0 \leq |m| \leq \ell`, and overtone number :math:`n`.
The waveform is such that the two GW polarizations for each :math:`j \equiv (+1, -2, \ell, |m|, n)` mode are given by

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

