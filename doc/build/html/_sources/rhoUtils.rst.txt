lsdensities/utils/rhoUtils.py
=============================

This modules contain some of the core classes used throughout the library, such as the Inputs class,
a container for all inputs related to the physics of the problem, or the Obs class, which aggregate in one place
features of observables (samples, mean, variance, covariance and so on). It also contains miscellaneous utility functions such
as loggers, a datafile reader, directories creation etc.

.. _Inputs-label:

Inputs class
------------

This is a container for a number of inputs that must be specified in order to work with this library.
Most functions do not accept the inputs separately, and the Inputs class must be passed to them,
in order to reduce the number of arguments.

The following attributes are available:

    **datapath**
        Path to the input data (correlator)

        Type: str

    **time_extent**
        Time extent of the lattice. If a datafile is red, this variable is inferred from the file.

        Type: int

    **num_samples**
        Sample size of the correlator. If a datafile is red, this variable is inferred from the file.

        Type: int

    **periodicity**
        String describing the periodicity in time of the correlator. Accepted values are "EXP" for open boundaries, "COSH" for periodic boundaries.

        Type: str

    **tmax**
        number of datapoints to be used. If not set, it will be inferred based on the values time_extent and periodicity.
        Note that the correlator at t=0 is excluded. For this reason, if periodicity == EXP,
        ``tmax = time_extent-1``. If periodicity == COSH, ``tmax = int(time_extent/2)``.

        Type: int

    **num_boot**
        Bootstrap samples. Relevant if you perform a bootstrap of the correlator within lsdensities.

        Type: int

    **sigma**
        Smearing radius of the kernel.

        Type: float

    **prec**
        Working numerical precision for mpmath.

        Type: int.

    **kerneltype**
        Smearing kernel. Implemented options are a Gaussian (with different normalisations) or a Cauchy Kernel.
        Accepted values are "FULLNORMGAUSS", "HALFNORMGAUSS", "CAUCHY". The first is a Gaussian normalised
        over the full real axis, the second is normalised in [0, inf), the third is a Cauchy kernel.

        Type: str

    **loglevel**
        Level of verbosity, handled by the logging python library. Accepted values are WARNING (default), INFO (verbose).

    **Na**
        Number of different values for the parameter :ref:`alpha <what_is_alpha-label>` used to determine best value of :ref:`lambda <what_is_lambda-label>`.
        Relevant if the user wants to work with :ref:`InverseProblemWrapper <InverseProblemWrapper-label>`.
        Accepted values are 1, 2 or 3.

        Type: int
    **emin**
        Smallest energy at which the smeared spectral density is computed.
        Relevant if the user wants to work with :ref:`InverseProblemWrapper <InverseProblemWrapper-label>`, which requires a set of energies.
        To use InverseProblemWrapper on a single energy, simply select `emin`=`emax` and `Ne=1`.

        Type: float

    **emax**
        Largest energy at which the smeared spectral density is computed.
        Relevant if the user wants to work with :ref:`InverseProblemWrapper <InverseProblemWrapper-label>`, which requires a set of energies.
        To use InverseProblemWrapper on a single energy, simply select `emin`=`emax` and `Ne=1`.

        Type: float

    **Ne**
        Number of energies at which the smeared spectral density is computed.
        Relevant if the user wants to work with :ref:`InverseProblemWrapper <InverseProblemWrapper-label>`, which requires a set of energies.
        To use InverseProblemWrapper on a single energy, simply select `emin`=`emax` and `Ne=1`.

        Type: int

    **A0cut**
        Smallest value of A[g]/A[0] accepted by :ref:`InverseProblemWrapper <InverseProblemWrapper-label>` during the search for the optimal value of :ref:`lambda <what_is_lambda-label>`.
        Relevant if the user wants to work with :ref:`InverseProblemWrapper <InverseProblemWrapper-label>`.

        Type: float

    **outdir**
        Directory where all the output will be stored.

        Type: str

    **logpath**
        Subdirectory of `outdir` containing logs

        Type: str

    **plotpath**
        Subdirectory of `outdir` containing plots

        Type: str

Examples usage
~~~~~~~~~~~~~~

Inputs can defined manually

.. code-block:: python

    parameters = Inputs()
    parameters.time_extent = 32
    parameters.kerneltype = "FULLNORMGAUSS"  # Kernel smearing spectral density
    parameters.periodicity = "EXP"  # EXP / COSH for open / periodic boundary conditions
    parameters.sigma = 0.25  # smearing radius in given energy units
    parameters.assign_values()

or via command line

.. code-block:: python

    par = parse_inputs()
    init_precision(par.prec)
    par.assign_values()
