from lsdensities.utils.rhoUtils import (
    init_precision,
    Inputs,
)
from mpmath import mp, mpf
from lsdensities.core import Smatrix_mp
from lsdensities.transform import h_Et_mp_Eslice, y_combine_central_Eslice_mp
from lsdensities.utils.rhoMath import gauss_fp

# compute the smeared spectral density at some energy,
# from a lattice correlator

init_precision(128)
parameters = Inputs()
parameters.time_extent = 32
parameters.kerneltype = "FULLNORMGAUSS"  # Kernel smearing spectral density
parameters.periodicity = "EXP"  # EXP / COSH for open / periodic boundary conditions
parameters.sigma = 0.25  # smearing radius in given energy units
peak = 1  #   energy level in the correlator
energy = 0.5  # energy at which the smeared spectral density is evaluated in given energy units
parameters.assign_values()  # assigns internal variables based on given inputs
# such as tmax = number of data points, which is inferred from time_extent and periodicity, if not specified

lattice_correlator = mp.matrix(
    parameters.tmax, 1
)  #  vector; to be filled with lattice data
lattice_covariance = mp.matrix(
    parameters.tmax
)  #  matrix; to be filled with data covariance

for t in range(parameters.tmax):  # mock data
    lattice_correlator[t] = mp.exp(-mpf(t + 1) * mpf(str(peak)))
    lattice_covariance[t, t] = lattice_correlator[t] * 0.02


regularising_parameter = mpf(str(1e-6))  # regularising parameters; must be tuned.
# Automatic tuning is provided in lsdensities/InverseProblemWrapper.py
# this example has exact data, so the parameters
# can be made as small as zero, in which case the result will be exact in
# the limit of infinite tmax

regularised_matrix = Smatrix_mp(parameters.tmax, alpha_=0) + (
    regularising_parameter * lattice_covariance
)
matrix_inverse = regularised_matrix ** (-1)

coeff = h_Et_mp_Eslice(
    matrix_inverse,  #   linear coefficients
    parameters,
    energy,
    alpha_=0,
)

result = y_combine_central_Eslice_mp(
    coeff,  #   linear combination of data
    lattice_correlator,
    parameters,
)

true_value = gauss_fp(peak, energy, parameters.sigma, norm="full")

print(
    "Result: ", float(result)
)  #   reconstructed smeared spectral density at E = energy
print("Exact results :", true_value)  #   exact smeared spectral density at E = energy
