# LSDensities: Lattice Spectral Densities


**lsdensities** is a Python library for the calculation of
smeared spectral densities from lattice correlators using variations of the <a href="https://arxiv.org/pdf/1903.06476.pdf">HLT</a> method in its <a href="https://inspirehep.net/files/5af0fb67df50ae0242c5a1a6820680cf">frequentist or Bayesian formulation</a>

Some features of this library:
- Stability</a> analysis to estimate the bias introduced by the Backus-Gilbert regulator.
- Error estimates via frequentist methods (bootstrap/jackknife...) or <a href="https://inspirehep.net/files/5af0fb67df50ae0242c5a1a6820680cf">Bayesian methods (via Gaussian Processes)</a>.
- Singular Value Decomposition and "<a href="https://arxiv.org/pdf/2605.14652">Eigenspace Analysis</a>"

This library is based on <a href="https://mpmath.org/">mpmath</a>
for performing the high-precision arithmetic operations that are necessary
for the solution of the inverse problem.


## Authors

Niccolò Forzano, Alessandro Lupo.

## Installation

One can download, build and install the package

```bash
pip install https://github.com/LupoA/lsdensities
```

## Usage

Some tests can be found in the ``tests`` folder, and tested using the ``pytest`` command.

Usage examples can be found in the ``examples`` folder.

The most basic workflow is illustrated in `examples/runExact.py`,
which generates a high-precision correlator, and computes the corresponding spectral density smeared with one of the
available kernels without regulators (the problem is not ill-posed when the data is not noisy).

A simple example of how to run the stability analysis is found in `examples/run_hlt_wnoise.py`, which uses synthetic data with synthetic noise.

Most inverse-problem related functionalities are in `src/lsdensities/inverse_problem_solvers` which contains routines for
- stability analysis `src/lsdensities/inverse_problem_solvers/stability_analysis.py`
- hlt method with frequentist and bayesian errors `src/lsdensities/inverse_problem_solvers/hlt_stability.py` 
- eigenspace analysis `src/lsdensities/inverse_prolem_solvers/hlt_eigenspace.py`

Function call example:

```python
from lsdensities.utils.common import (
    init_precision,
    Inputs,
)
from mpmath import mp, mpf
from lsdensities.core import cauchy_matrix
from lsdensities.transform import coefficients_ssd, get_ssd_scalar
from lsdensities.utils.math_utils import gauss_fp

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
                                          # Automatic tuning is provided in hlt_stability.py
                                          # this example has exact data, so the parameters
                                          # can be made as small as zero,
                                          # in which case the result will be exact in
                                          # the limit of infinite tmax

regularised_matrix = cauchy_matrix(parameters.tmax, alpha=0) + (regularising_parameter * lattice_covariance)
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
M. Hansen, A. Lupo, N.Tantalo, <a href="https://arxiv.org/pdf/1903.06476.pdf">"Extraction of spectral densities from lattice correlators"</a>, Phys.Rev.D 99 (2019) 9, 094508

L. Del Debbio, A. Lupo, M. Panero, N. Tantalo, <a href="https://inspirehep.net/files/5af0fb67df50ae0242c5a1a6820680cf">Bayesian solution to the inverse problem and its relation
to Backus–Gilbert methods</a>, Eur.Phys.J.C 85 (2025) 2, 185

A. Lupo, N. Tantalo, <a href="https://arxiv.org/pdf/2605.14652">Extraction of spectral densities from lattice correlators: decoupling signal from noise</a>

## License

[GPL](https://choosealicense.com/licenses/gpl-3.0/)
