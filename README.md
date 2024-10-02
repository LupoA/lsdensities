# LSDensities: Lattice Spectral Densities


**lsdensities** is a Python library for the calculation of
smeared spectral densities from lattice correlators.

Solutions can be obtained with the
<a href="https://arxiv.org/pdf/1903.06476.pdf">Hansen Lupo Tantalo</a> method
and <a href="https://arxiv.org/pdf/2311.18125.pdf">
Bayesian inference with Gaussian Processes</a>, or combinations of the two.

This library is based on <a href="https://mpmath.org/">mpmath</a>
for performing the high-precision arithmetic operations that are necessary
for the solution of the inverse problem.


## Authors

Niccolò Forzano, Alessandro Lupo.

## Documentation

Documentation is under construction. It can be found <a href="https://lupoa.github.io/lsdensities/index.html">here</a> in its preliminary form.

## Installation

One can download, build and install the package

```bash
pip install https://github.com/LupoA/lsdensities
```

## Usage

Preliminary tests can be found in the ``tests`` folder, and tested using the ``pytest`` command.

Usage examples can be found in the ``examples`` folder.

The most basic workflow is illustrated in `examples/runExact.py`,
which generates a high-precision correlator, and computes the corresponding spectral density smeared with one of the
available kernels.

A realistic example is shown in ```examples/runInverseProblem.py```, where input data for the correlator
needs to be provided.

The most complete class is `src/lsdensities/InverseProblemWrapper.py`, which
provides utilities for estimating errors and treating
the bias both in the HLT and in the Bayesian framework.

Function call example:

```python
from lsdensities.utils.rhoUtils import (
    init_precision,
    Inputs,
)
from mpmath import mp, mpf
from lsdensities.core import hlt_matrix
from lsdensities.transform import coefficients_ssd, get_ssd_scalar
from lsdensities.utils.rhoMath import gauss_fp

# compute the smeared spectral density at some energy,
# from a lattice correlator

init_precision(128)
parameters = Inputs()
parameters.time_extent = 32
parameters.kerneltype = 'FULLNORMGAUSS'  # Kernel smearing spectral density
parameters.periodicity = 'EXP'  # EXP / COSH for open / periodic boundary conditions
parameters.sigma = 0.25  # smearing radius in given energy units
peak = 1    #   energy level in the correlator
energy = 0.5     # energy at which the smeared spectral density
                 # is evaluated in given energy units
parameters.assign_values()  # assigns internal variables
                            # based on given inputs
                            # such as tmax = number of data points,
                            # which is inferred from time_extent and periodicity,
                            # if not specified

lattice_correlator = mp.matrix(parameters.tmax, 1)  #  vector; fill with lattice data
lattice_covariance = mp.matrix(parameters.tmax)     #  matrix; fill with data covariance

for t in range(parameters.tmax):    # mock data
    lattice_correlator[t] = mp.exp(-mpf(t + 1) * mpf(str(peak)))
    lattice_covariance[t,t] = lattice_correlator[t] * 0.02


regularising_parameter = mpf(str(1e-6))   # regularising parameters; must be tuned.
                                          # Automatic tuning is provided in InverseProblemWrapper.py
                                          # this example has exact data, so the parameters
                                          # can be made as small as zero,
                                          # in which case the result will be exact in
                                          # the limit of infinite tmax

regularised_matrix = hlt_matrix(parameters.tmax, alpha=0) + (regularising_parameter * lattice_covariance)
matrix_inverse = regularised_matrix**(-1)

coeff = coefficients_ssd(matrix_inverse,   # linear coefficients
                       parameters,
                       energy,
                       alpha=0)

result = get_ssd_scalar(coeff,     # linear combination of data
                                     lattice_correlator,
                                     parameters)

true_value = gauss_fp(peak, energy, parameters.sigma, norm="full")

print("Result: ", float(result))   # reconstructed smeared spectral density at E = energy
print("Exact results :", true_value)  # exact smeared spectral density at E = energy
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

Development requirements can be installed by using ``pip install -r requirements.txt``, and they are listed in ``requirements.txt``.

## References
For the main ideas: https://arxiv.org/pdf/1903.06476.pdf

For the Bayesian setup and the general treatment of the bias: https://arxiv.org/pdf/2409.04413

## License

[GPL](https://choosealicense.com/licenses/gpl-3.0/)
