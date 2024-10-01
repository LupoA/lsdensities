lsdensities/InverseProblemWrapper.py
====================================

Overview
--------

The file contains the main class `InverseProblemWrapper` and a few auxiliary classes.

The optimisation process within InverseProblemWrapper seeks to obtain the smeared spectral density at different, decreasing values of `lambda`. Once a plateau is identified, meaning that subsequent values of lambda provide compatibles
estimates for the smeared density, a value for `lambda` inside the plateau and the corresponding smeared
spectral densities are pulled out, and identified as the result. As further check, the result is compared with a second value of `lambda`, looking for differences to add as systematic errors. The auxiliary `AlgorithmParameters` class
encapsulates the parameters the control this scan over `lambda`. They instruct the class on what should be interpreted as a satisfying plateau, how many points in the plateau should be evaluated, where to perform the control for systematic error and so on.


Up to three different values of `alpha` can be used. If additional values are used, the algorithm will seek for a common plateau among different values of `alpha`. We remind that `lambda` and `alpha` refers to the variables entering the
expression for the coefficients,

.. math::

    \vec{g}(\sigma;\omega) = \argmin_{\vec{g} \in \mathbb{R}^{\tau_{\rm max}}} \int_0^\infty dE \, e^{\alpha E} | \sum_{r=1}^{\tau_{\rm max}} g_\tau(\omega) b_T(a \tau, E) - \mathcal{S}_\sigma(E,\omega) |^2 \\  + \lambda \; \sum_{\tau_1, \tau_2=1}^{\tau_{\rm max}} g_{\tau_1}(\omega) \, B_{\tau_1 \tau_2}\,  g_{\tau_2}(\omega) \, , \;\;\;\;\; \lambda \in (0,\infty)

The matrix B is proportional to the covariance matrix of the correlator. Part of the proportionality factor is the norm (in the sense defined above, i.e. :math:`alpha` and energy dependent) of the smearing kernel. This is pre-computed at all requested energies by the private class _NormaliseMeasure.
The above equation amounts to solve a linear system,

.. math::
    g_{\tau_1}(\sigma;\omega) = [\Sigma(\alpha) + c(\lambda, \alpha, \omega) \text{Cov}]^{-1}_{\tau_1 \tau_2} \, f_{\tau_1}(\omega,\sigma,\alpha)

The matrix :math:`\Sigma` is also pre-computed by the `SigmaMatrix` class. The normalising facotr `c` is computed within the InverseProblemWrapper class, and the vector `f` can be found in `lsdensities/core.py`.

In the following we describe the auxiliary classes first, then InverseProblemWrapper.

class AlgorithmParameters
-------------------------

This auxiliary class contains the parameter that determine the search for the optimal value of `lambda`.

**Parameters**

    - ``alphaA`` (`float`, optional): The first alpha value. Defaults to `0`. Must be less than 2.
    - ``alphaB`` (`float`, optional): The second alpha value, distinct from `alphaA`. Defaults to `1/2`. Must be less than 2.
    - ``alphaC`` (`float`, optional): The third alpha value, distinct from both `alphaA` and `alphaB`. Defaults to `1.99`. Must be less than 2.
    - ``lambdaMax`` (`float`, optional): The starting value for the scan over `lambda`. Defaults to `50`.
    - ``lambdaStep`` (`float`, optional): The step size for scanning over `lambda`. Defaults to `25`. After one iteration, lambda is replaced by lambda - lambdaStep. If a negative value is hit, lambdaStep is rescaled. When lambdaMin is hit, the algorithm stops.
    - ``lambdaScanCap`` (`int`, optional): The number of subsequent compatible measurements required to stop the scan. Defaults to `6`.
    - ``plateau_id`` (`int`, optional): A plateau consists of subsequently compatible measurements. This parameter determines which of this values is taken as the solution. Defaults to `1`, meaning that from a chain of compatible values, the first one (corresponding to the larger value of lambda) is chosen.
    - ``kfactor`` (`float`, optional): A factor used to estimate systematics by repeating the calculation at `lambda = kfactor * lambda(reference)`. Defaults to `0.1`.
    - ``lambdaMin`` (`float`, optional): The minimum value for `lambda` in the scan. Defaults to `1e-6`.
    - ``comparisonRatio`` (`float`, optional): Used to define compatibility between measurements. Measurements are considered compatible if they agree within `comparisonRatio * uncertainty`. Defaults to `0.4`, meaning that values are considered compatible if they agree within "0.4 sigma".
    - ``resize`` (`int`, optional): If `lambda` hits zero before reaching `lambdaMin`, the step size is resized. This allows the algorithm to sample values of `lambda` at different scales. Defaults to `4`.

    With default parameters, the algorithm will scan the following values of `lambda`: 50, 25, 18.75, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.1953125, 0.09765625, 0.048828125, and so on until lambdaMin is hit, or a plateau is identified.

**Example usage**

Here’s how you can initialize and use the `AlgorithmParameters` class:

.. code-block:: python

    from lsdensities import AlgorithmParameters

    # Initialize the parameters for the scan
        inverse_problem_parameters = AlgorithmParameters(
        alphaA=0,
        alphaB=1.99,
        alphaC=0.5,
        lambdaMax=lambdaMax,
        lambdaStep=lambdaMax / 2,
        lambdaScanCap=8,
        kfactor=0.1,
        lambdaMin=5e-2,
        comparisonRatio=0.3,
    )

    print(f"Lambda Max: {params.lambdaMax}")
    print(f"Alpha A: {params.alphaA}")
    print(f"Comparison Ratio: {params.comparisonRatio}")


class InverseProblemWrapper
---------------------------

Performs the scan over lambda, selects a solution, which is stored in internal variables together with other output values.

The class takes the following input parameters:

**Parameters**

    - ``par`` (`Inputs`): An instance of the `Inputs` class containing required parameters about the lattice.
    - ``algorithmPar`` (`AlgorithmParameters`): An instance of the AlgorithmParameters class containing the parameter for the selection of `lambda`
    - ``matrix_bundle`` (`MatrixBundle`): Instance of the MatrixBoundle class, which contains the covariance matrix of the correlator and its normalisation factor.
    - ``correlator`` (Obs): An instance of the Obs class which contains the measurements of the lattice correlators, together with other related features.
    - ``energies`` (np.array) Numpy array containing the energies, typically ``np.linspace(par.emin, par.emax, par.Ne)``.

The class has a large number of attributes. We report the most important ones that the user may need to access.

The class has the following methods:

    **lambdaResultHLT**
        Array for which each entry is the optimal value of `lambda` obtained from the plateau search. Different entries correspond to different energies.

        Type: np.float64

    **rhoResultHLT**
        Array for which each entry is the smeared spectral density corresponding to the optimal value of `lambda` given by ``self.lambdaResultHLT``. Different entries correspond to different energies.

        Type: np.float64

    **drho_result**
        Array for which each entry is the statistical error on the smeared spectral density stored in ``self.rhoResultHLT``. Different entries correspond to different energies.

        Type: np.float64

    **rho_sys_err_HLT**
        Array for which each entry is the systematic error on the smeared spectral density stored in ``self.rhoResultHLT``. Different entries correspond to different energies.

        Type: np.float64

    **lambdaResultBayes**
        Array for which each entry is the optimal value of `lambda` obtained minimising the negative log likelihood (NLL). Different entries correspond to different energies.

        Type: np.float64

    **rhoResultBayes**
        Array for which each entry is the smeared spectral density corresponding to the optimal value of `lambda` given by ``self.lambdaResultBayes``. Different entries correspond to different energies.

        Type: np.float64

    **drho_bayes**
        Array for which each entry is the statistical error (sqrt of the width of the posterior distribution) on the smeared spectral density stored in ``self.rhoResultBayes``. Different entries correspond to different energies.

        Type: np.float64

    **rho_sys_err_Bayes**
        Array for which each entry is the systematic error on the smeared spectral density stored in ``self.rhoResultBayes``. Different entries correspond to different energies.

        Type: np.float64

    **gt_HLT**
        For each energy, a list containing the linear coefficients :math:`g_\tau` generating the solution ``self.rhoResultHLT``. Its structure is ``[[] for _ in range(self.par.Ne)]``, meaning that
        it is effectively a 2D array, where one dimension runs over the data index (time) and the other labels different energies.

        Type: List of lists

    **gt_Bayes**
        For each energy, a list containing the linear coefficients :math:`g_\tau` generating the solution ``self.rhoResultBayes``. Its structure is ``[[] for _ in range(self.par.Ne)]``, meaning that
        it is effectively a 2D array, where one dimension runs over the data index (time) and the other labels different energies.

        Type: List of lists

The class features a number of methods. The main one that is intended to be accessed externally is InverseProblemWrapper.run(). This function performs the scan over
`lambda` and selects the solution. A number of preparatory functions needs to be however called.

    **fillEspaceMP** ()
        Fills internal variables containing the energies at which we requested to solve the inverse problem. Additionally fills a dictionary, so that
        the integer index can be accessed from the value of the energies.

        This should be made private.

    **prepareHLT** ()
        runs ``fillEspaceMP()``, computes and stores the :math:`\Sigma` matrix and normalising factors.


    **run** (savePlots=True, livePlots=False)
        For each energy in ``self.energies``, it calls the method ``self.scanParameters`` which performs a scan over `lambda` and finds the best values, both frequentist and Bayesian.
        After this is done, it computes the systematic error by repeating the calculation at a different value of `lambda`, which was prescribed in the ``AlgorithmicParameters`` class passed
        as an input. Finally, it prints various results in an output file. Depending on the boolean argument, it stores a number of plots in the output directory. livePlots will make the plots appear as the application is executed.

class SigmaMatrix
-----------

Class computing and storing elements of the matrix :math:`\Sigma`. If the periodicity is set to EXP, this corresponds to

.. math::
    \Sigma_{\tau_1 \tau_2} = \frac{1}{\tau_1 + \tau_2 + 2 - \alpha}

If the periodicity is set to COSH, the expression is generalised appropriately.

.. warning::
    The presence of a +2 is due to the fact that we do not use the correlator evaluated at :math:`t=0`. No additional shift is required.


The class takes the following parameters

    **Parameters**
        - ``par`` (`Inputs`): An instance of the `Inputs` class containing required parameters.
        - ``alphaMP`` (`mpmath.mpf`, optional): The `alpha` parameter. Defaults to `0`.

The class has the following attributes

    **par**
        The parameters passed as inputs

        Type: Inputs class

    **tmax**
        Simply par.tmax (redundant)

        Type: int

    **matrix**
        The matrix :math:`\Sigma`

        Type: mp.matrix(par.tmax, par.tmax)

The class has the following methods

    **evaluate**
        Fills the entries of ``self.matrix``. Does not `return`.

class _NormaliseMeasure (Private)
---------------------------

The `_NormaliseMeasure` class pre-evaluated part of the normalising facotrs required by the InverseProblemWrapper class. It is an array (one-dimensional matrix) of mpf numbers, one for each energy requested.
Values can be accessed by using the integer index, or by exact energy value through a dictionary. The value computed is historically called :math:`A_0`

.. math::
    A_0(\omega) = \int_{E_0}^\infty dE \,  e^{\alpha E} \, | \mathcal{S}_\sigma(\omega-E) |^2

The class takes the following parameters

    **Parameters**
        - ``par`` (`Inputs`): An instance of the `Inputs` class containing required parameters.
        - ``alpha`` (`mpmath.mpf`, optional): The `alpha` parameter. Defaults to `0`.
        - ``emin`` (`mpmath.mpf`, optional): The lower bound of the integral. Defaults to `0`.

The class has the following attributes

    **valute_at_E**
            The matrix that stores the evaluated values of :math:`A_0` for all energies in `espace_mp`.

            Type: `mpmath.matrix`
    **valute_at_E_dictionary**
            A dictionary where the keys are energy values and the values are the corresponding evaluated :math:`A_0`.

            Type: `dict`
    **is_filled**
            A flag that indicates whether the :math:`A_0` values have been computed and stored.

            Type: `bool`
    **alphaMP**
            The multiple-precision value of `alpha`, which is used in the :math:`A_0` evaluation.

            Type: `mpmath.mpf`
    **eminMP**
            The multiple-precision minimum energy used in the calculation.

            Type: `mpmath.mpf`
    **par**
            An instance of the `Inputs` class containing some required parameters.

            Type: Inputs

The class has the following methods

    **evaluate** `(espace_mp)`
            computes :math:`A_0` for all energy values in `espace_mp` and assigns ``self.is_filles = True``. Does not `return`.

            **Parameters**
                - ``espace_mp`` (`mpmath.matrix`): A multi-precision matrix that contains the energy values at which :math:`A_0` is evaluated.
