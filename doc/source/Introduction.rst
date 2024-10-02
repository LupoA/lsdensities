Setting up lsdensities
======================

In your directory of choice, clone the repository,

.. code-block:: shell

    git clone https://github.com/LupoA/lsdensities

You can then install by running:

.. code-block:: shell

    cd lsdensities
    pip install -e .

Checking that it works
----------------------

Moving to the example directory

.. code-block:: shell

    cd lsdensities/examples/

Run the minimal example,

.. code-block:: shell

    python minimal_example.py

Alternatively, in the test directory, you can run the tests, which should take seconds.

.. code-block:: shell

     cd lsdensities/tests/
     pytest

Usage examples
==============

There are two ways of using this library, depending on how the user wants to interface with the
optimisation of the parameter :ref:`lambda <what_is_lambda-label>`. One way is to use the library to compute
the spectral density at a fixed value of lambda. One can loop over different values of lambda and choose the value
that is deemed correct. Alternatively, the optimisation of lambda is automatised in the :ref:`InverseProblemWrapper <InverseProblemWrapper-label>` class. This is the recommended way.
In the class is used, only a handful of inputs and the data must be passed to the library, and the rest is handled automatically.
The user still has a large degree of freedom by tuning the input :ref:`parameters of the class <AlgorithmParameters-label>`.
Moreover, the :ref:`InverseProblemWrapper <InverseProblemWrapper-label>` class will output the smeared spectral density both from a Bayesian and a frequentist algorithm.

To immediately look the workflow without the InverseProblemWrapper, look at ``examples/minimal_example.py``.

To immediately look at the workflow using the InverseProblemWrapper, look at ``examples/runInverseProblem.py`` (requires a datafile).

We can start by looking at some common, basic features.
At the beginning, the desired level of numerical precision can be selected. This can be done with the command

.. code-block:: python

    num_digits = 64
    init_precision(num_digits)

which sets, in this example, the precision to approximately 64 numerical digits.

An :ref:`Inputs <Inputs-label>` object must be then defined. This object specify some basic inputs, such as the type of the smearing kernel, its smearing radius, or the energies at which we want to obtain the solution. The Input object is ubiquitous in lsdensities and it should be well understood.
While its attributed have default values, they should be set manually or via input files. For instance:

.. code-block:: python

    parameters = Inputs()
    parameters.periodicity = "EXP"  # "EXP" or  "COSH" depending
                                    # on the lattice having open
                                    # or periodic boundary conditions in time
    parameters.sigma = 0.25  # smearing radius in given energy units

Some inputs are dimensionful quantities, and the user should take care of using them consistently (e.g. everything is expressed in GeV).
Once the desired attributed for the :ref:`Inputs <Inputs-label>` object are specify, a function needs to called in order to fill some internal attributed based on the values provided.

.. code-block:: python

    parameters.assign_values()

An important variable set by this function is ``parameters.tmax`` which specifies the number of datapoints that will be actually used. If unspecified, it uses the maximum values, which is inferred by ``parameters.time_extent`` and ``parameter.periodicity``.

.. warning::
    running ``parameters.assign_values()`` is mandatory, and applications may not work if this function is not called. Attributed of parameters should not be modified after ``assign_values()`` is called.

A first look: solving against synthetic correlators (no datafile required) at a single value of :math:`\lambda`
---------------------------------------------------------------------------------------------------------------

In the following we create a single sample for a correlator and we extract the smeared spectral density at a given energy.
This basic application of the library is intended to familiarise the user with its basic feature, but it does not contain all the information that is necessary to run against a real dataset.


First, initialise the precision and the Input object. Since the data will be synthetic, we have to additionally specify the time extent of the lattice, which is normally red from the datafile.

.. code-block:: python
    init_precision(128)
    parameters = Inputs()
    parameters.time_extent = 32
    parameters.kerneltype = "FULLNORMGAUSS"  # Kernel smearing spectral density
    parameters.periodicity = "EXP"  # EXP / COSH for open / periodic boundary conditions
    parameters.sigma = 0.25  # smearing radius in given energy units
    parameters.assign_values()  # assigns internal variables based on given inputs

We then create the synthetic correlator to serve as input. This must be an `mp.matrix <https://mpmath.org/doc/current/matrices.html>`_ type.
The number of rows is the number of data points at which the correlator is computed (``parameters.tmax``).
The number of columns is the number of samples for the correlator, in this case one. The associated covariance matrix should have matching size.

.. code-block:: python
    lattice_correlator = mp.matrix(
        parameters.tmax, 1
    )
    lattice_covariance = mp.matrix(
        parameters.tmax
    )

We fill the correlator with a simple exponential function decaying according to a values MASS. We also populate the covariance matrix artificially.

.. code-block:: python

    MASS = 1 # in the given energy units
    for t in range(parameters.tmax):  # mock data
        lattice_correlator[t] = mp.exp(-mpf(t + 1) * mpf(str(MASS)))
        lattice_covariance[t, t] = lattice_correlator[t] * 0.02 # for this quick example
                                                                # we set the covariance to diagonal,
                                                                # with a value of e.g. 2% of the correlator

Notice that the argument of the exponential defining ``lattice_correlator`` is shifted by one, because the correlator at :math:`t=0` cannot be used.

The library then provides function to compute the smeared spectral density as a linear combination of the correlators,

.. math::

    \rho_\sigma(E) = \sum_{t=1}^{t_{\text{max}}} g_t(E,\sigma) \,  C(t)

The coefficients :math:`g_t(E,\sigma)` are compute through a linear system. To this end, we shall define the appropriate matrix:

.. code-block:: python

    ill_conditioned_matrix = hlt_matrix(parameters.tmax, alpha=0)

This has to be in general regularised. In this example it would not be necessary since we created our correlator was defined with mpmath to be exact up to 64 digits, but we do it for pedagogical reasons

.. code-block:: python

    regularising_parameter = mpf(str(1e-6))
    regularised_matrix = ill_conditioned_matrix + (
        regularising_parameter * lattice_covariance
    )
    matrix_inverse = regularised_matrix ** (-1)

The value of ``regularising_parameter`` regularises the solution but introduces a bias.
Its effect accounted for with a high degree of automation by the :ref:`InverseProblemWrapper <InverseProblemWrapper-label>` class.
In this first example, we simply set it to a small value.

Having computed and inverted the appropriate matrix, we can obtain the coefficients by using the ``coefficients_ssd`` function

.. code-block:: python
    energy = 0.5 # the energy at which we compute the smeared spectral density
    coeff = coefficients_ssd(
        matrix_inverse,
        parameters,
        energy,
        alpha=0,
    )

The result is then computed with the ``get_ssd_scalar`` function

.. code-block:: python
    result = get_ssd_scalar(
    coeff,  #   linear combination of data
    lattice_correlator,
    parameters,
    )

 In this example, the derived solution can be compared with the true value

.. code-block:: python
    true_value = gauss_fp(peak, energy, parameters.sigma, norm="full")

The derived solution should approach the true value as ``regularising_parameter`` is reduced towards zero and ``par.time_extent`` is increased.

.. warning::
    The time argument of the correlator must be shifted by one unit. This is because the correlator in zero `cannot enter the reconstruction process <https://arxiv.org/pdf/1903.06476>`_.
    When using a datafile, you do NOT need to remove the correlator at :math:`t=0` from it, because this is done automatically by the library.

A second look: scan over :math:`\lambda` and automatised workflow
-----------------------------------------------------------------

under construction


Example applications in the examples directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some applications implementing variations of the strategy described above are available in the ``examples`` directory.

The file ``minimal_example.py`` contains a slightly less verbose of the code reported above.

The file ``runExact.py`` contains a similar example where the smeared spectral density is computed at a large number of energies.
