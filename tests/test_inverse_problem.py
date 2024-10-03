from lsdensities.utils.rhoUtils import (
    Inputs,
)
from mpmath import mp, mpf
from lsdensities.core import hlt_matrix
from lsdensities.transform import coefficients_ssd, get_ssd_scalar
from lsdensities.utils.rhoMath import gauss_fp


def run(time_extent):
    parameters = Inputs()
    parameters.time_extent = time_extent
    parameters.kerneltype = "FULLNORMGAUSS"
    parameters.periodicity = "EXP"
    parameters.sigma = 0.25
    parameters.prec = 128
    peak = 1
    energy = 0.5
    parameters.assign_values()

    lattice_correlator = mp.matrix(parameters.tmax, 1)
    lattice_covariance = mp.matrix(parameters.tmax)

    for t in range(parameters.tmax):  # mock data
        lattice_correlator[t] = mp.exp(-mpf(t + 1) * mpf(str(peak)))
        lattice_covariance[t, t] = lattice_correlator[t] * 0.02

    regularised_matrix = hlt_matrix(parameters.tmax, alpha=0)
    matrix_inverse = regularised_matrix ** (-1)

    coeff = coefficients_ssd(
        matrix_inverse,
        parameters,
        energy,
        alpha=0,
    )

    result = get_ssd_scalar(
        coeff,
        lattice_correlator,
        parameters,
    )

    true_value = gauss_fp(peak, energy, parameters.sigma, norm="full")

    diff = float(abs(result - true_value))
    print("time extent : ", time_extent, "diff = ", diff)

    if time_extent >= 32:
        assert diff < 1e-6
    elif time_extent < 32 and time_extent > 8:
        assert diff < 1e-1

    return float(diff)


def test_convergence():
    diff8 = run(8)
    diff16 = run(16)
    diff32 = run(32)
    diff48 = run(48)
    diff64 = run(64)

    assert diff8 > diff16
    assert diff16 > diff32
    assert diff32 > diff48
    assert diff48 > diff64


test_convergence()
