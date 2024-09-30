Welcome to the lsdensities' documentation.
====================================

**Lattice Spectral Densities** is a python library providing tools for the calculation of smeared spectral densities from lattice correlators.
The library uses the `HLT method <https://arxiv.org/pdf/2409.04413>`_ in both its frequentist and Bayesian formulations. It is based on `mpmath <https://mpmath.org/>`_ for handling arbitrary precision arithmetics.

The library takes a measurements for a lattice correlator, which is sampled at a finite number of points and returns the associated spectral density smeared with a chosen kernel, which is a Gaussian function by default.
It is assumed that the relation between the correlator and the spectral density is of the following type:

.. math::

    C(t) = \int_0^\infty dE \, \rho(E) \, b_T(t, E)

where :math:`b_T(t, E)` can be either :math:`e^{-t E}` or :math:`e^{-t E} + e^{(-T+t) E}`. Other options can be implemented or requested.

The smeared spectral density is computed from a linear combination of the correlators,

.. math::

    \rho_\sigma(E) = \sum_{t=1}^{t_{\text{max}}} g_t(E,\sigma) \,  C(t)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction
   InverseProblemWrapper