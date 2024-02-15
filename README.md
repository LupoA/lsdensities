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

## Installation

One can download, build and install the package

```bash
pip install https://github.com/LupoA/lsdensities
```

## Usage

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
import lsdensities

# compute the smeared spectral density at some energy,
# from a lattice correlator

inputMatrix = lsdensities.core.Smatrix_mp(time_extent)

coeff = lsdensities.transform.h_Et_mp_Eslice(inputMatrix, 
                                             parameters, 
                                             energy)

res = lsdensities.transform.y_combine_central_Eslice_mp(coeff, 
                                                        lattice_correlator, 
                                                        parameters)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## References
For the mean ideas: https://arxiv.org/pdf/1903.06476.pdf

For the Bayesian setup and the general treatment of the bias: https://arxiv.org/pdf/2311.18125.pdf

## License

[GPL](https://choosealicense.com/licenses/gpl-3.0/)
