Welcome to the lsdensities' documentation.
==========================================

**Lattice Spectral Densities** is a python library providing tools for the calculation of smeared spectral densities from lattice correlators.
The library uses the `HLT method <https://arxiv.org/pdf/2409.04413>`_ in both its frequentist and Bayesian formulations. It is based on `mpmath <https://mpmath.org/>`_ for handling arbitrary precision arithmetics.

The library takes a measurements for a lattice correlator :math:`C(t)`, which is sampled at a finite number of points, and returns the associated spectral density :math:`\rho(E)` smeared with a chosen kernel:

.. math::

    \rho(\sigma;\omega) = \int dE \, \mathcal{S}_\sigma(\omega-E) \, \rho(E) \, .

The smearing kernel :math:`\mathcal{S}(\omega-E)` is a Gaussian function by default. Different options are available, and others can be implemented or requested.

It is assumed that the relation between the correlator and the spectral density is of the following type:

.. math::

    C(t) = \int_0^\infty dE \, \rho(E) \, b_T(t, E)

where :math:`b_T(t, E)` can be either :math:`e^{-t E}` or :math:`e^{-t E} + e^{(-T+t) E}`. Other options can be implemented or requested.

The smeared spectral density is computed from a linear combination of the correlators,

.. math::

    \rho(\sigma, \omega) = \sum_{\tau = 1}^{\tiny \tau_{\rm max} } g_t(E) C(a \tau) \, , \;\;\;\;\; 0 < \tau \leq \tau_{\max} \; , \;\;\; \;\; \tau = t / a \; ,

where :math:`a` is the lattice spacing and :math:`\tau_{\max}` the number of data points. The coefficients are computed according to the following expression:

.. math::

    \vec{g}(\sigma;\omega) = \argmin_{\vec{g} \in \mathbb{R}^{\tau_{\rm max}}} \int_0^\infty dE \, e^{\alpha E} | \sum_{r=1}^{\tau_{\rm max}} g_\tau(\omega) b_T(a \tau, E) - \mathcal{S}_\sigma(E,\omega) |^2 \\  + \lambda \; \sum_{\tau_1, \tau_2=1}^{\tau_{\rm max}} g_{\tau_1}(\omega) \, B_{\tau_1 \tau_2}\,  g_{\tau_2}(\omega) \, , \;\;\;\;\; \lambda \in (0,\infty)

.. _what_is_lambda-label:

At :math:`\lambda=0` this expression provides the exact solution for the problem,
which is however deeply unstable when the correlator is affected by noise.
The class `InverseProblemWrapper.py` provides routines that automatically
optimise the value of :math:`\lambda`.

.. _what_is_alpha-label:

Different values of :math:`\alpha < 2` can be also chosen. This allows to work with different
norms. Different choices can have better convergence properties at different energies.



Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction
   InverseProblemWrapper
   rhoUtils
