The procedure
=============

Our method consists in a set of rules that aim to provide an unbiased estimate for a smeared spectral density. The lsdensities
library provides functions for the implementation of these rules.

    1. Establish a parametric map between the input data :math:`C(t)` and a smeared spectral density :math:`\rho_\sigma(\omega)`.
    Given a set of parameters :math:`\mathbb{p} = \lambda, \alpha`.

        .. math::

            C(t) \underset{\mathbb{p}}{\mapsto} \rho^{\mathbb{p}}_\sigma(E)

        This is implemented through the equation

        .. math::

            \vec{g}(\sigma;\omega) = \argmin_{\vec{g} \in \mathbb{R}^{\tau_{\rm max}}} \int_0^\infty dE \, e^{\alpha E} | \sum_{r=1}^{\tau_{\rm max}} g_\tau(\omega) b_T(a \tau, E) - \mathcal{S}_\sigma(E,\omega) |^2 \\  + \lambda \; \sum_{\tau_1, \tau_2=1}^{\tau_{\rm max}} g_{\tau_1}(\omega) \, B_{\tau_1 \tau_2}\,  g_{\tau_2}(\omega) \, , \;\;\;\;\; \lambda \in (0,\infty)


        For a generic set of parameters :math:`\mathbb{p}`, in particular for :math:`\lambda`, this will not be the solution we are after. However, for a subset of
        values of :math:`\mathbb{p}`, the map provides a solution that is compatible, within the statistical error of the input data,
        with the spectral density associated with :math:`C(t)` and smeared with the desired kernel.

The second operation that must be performed is then:

    2. Identification of the value of :math:`\lambda` (and :math:`\alpha` if desired) such that the solution is compatible with the input within errors.

        This task can be implemented manually by the user, but this library manages it automatically via the :ref:`InverseProblemWrapper <InverseProblemWrapper-label>` class in two different ways.
        Two optimal values of :math:`\lambda` will be suggested
        based on two different interpretations.

        One relies on the identification of a plateau in the values of :math:`\rho_\sigma`
        as a function of :math:`\lambda`, as the latter variable is reduced.
        The results are collected in :ref:`InverseProblemWrapper.rhoResultHLT <rhoResultHLT-label>`.

        The other suggested value is based on Bayesian inference and is the result of the maximisation of a likelihood function.
        The results are collected in :ref:`InverseProblemWrapper.rhoResultBayes <rhoResultBayes-label>`.

        If everything works correctly, the two values should be compatible.

        The search for the optimal value of :math:`\lambda`
        requires input values to be passed through the :ref:`AlgorithmParameters <AlgorithmParameters-label>` class.
