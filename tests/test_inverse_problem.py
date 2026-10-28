import pytest
from lsdensities.utils.common import (
    init_precision,
    Inputs,
)
from mpmath import mp, mpf
from lsdensities.core import cauchy_matrix
from lsdensities.transform import coefficients_ssd, get_ssd_scalar
from lsdensities.utils.math_utils import gauss_fp


def run(time_extent):
    init_precision(128)
    parameters = Inputs()
    parameters.time_extent = time_extent
    parameters.kerneltype = "FULLNORMGAUSS"
    parameters.periodicity = "EXP"
    parameters.sigma = 0.25
    peak = 1
    energy = 0.5
    parameters.assign_values()

    lattice_correlator = mp.matrix(parameters.tmax, 1)
    lattice_covariance = mp.matrix(parameters.tmax)

    for t in range(parameters.tmax):  # mock data
        lattice_correlator[t] = mp.exp(-mpf(t + 1) * mpf(str(peak)))
        lattice_covariance[t, t] = lattice_correlator[t] * 0.02

    regularised_matrix = cauchy_matrix(parameters.tmax, alpha=0)
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


def test_stability_analysis_power_kernel():
    # rho(E) = sum_i w_i delta(E - E_i): int rho(E) / E^3 dE = sum_i w_i / E_i^3.
    # e0 > 0 and alpha != 0 in every channel.
    import numpy as np

    from lsdensities.inverse_problem_solvers.hlt_stability import AlgorithmParameters, HLTWithBackusGilbert
    from lsdensities.utils.common import Obs

    init_precision(60)
    peaks, weights = np.array([0.5, 0.9, 1.6]), np.array([0.6, 1.0, 0.8])
    par = Inputs()
    par.time_extent = 16
    par.periodicity = "EXP"
    par.kerneltype = "INVPOW3"
    par.sigma = 0
    par.emin = par.emax = par.e0 = 0.3
    par.Ne = 1
    par.Na = 3
    par.num_boot = 200
    par.assign_values()

    t = np.arange(par.time_extent)
    exact = (weights * np.exp(-np.outer(t, peaks))).sum(axis=1)
    rng = np.random.default_rng(0)
    corr = Obs(T=par.time_extent, tmax=par.tmax, nms=par.num_boot, sample_type="bootstrap")
    corr.sample = rng.multivariate_normal(exact, np.diag((1e-4 * exact) ** 2), size=par.num_boot)
    corr.evaluate()
    corr.evaluate_covmatrix()
    corr.fill_mp_sample()

    hlt = HLTWithBackusGilbert(
        par=par,
        algorithmPar=AlgorithmParameters(
            alphaA=-0.5, alphaB=-1.0, alphaC=-1.5, lambdaMax=10, lambdaMin=1e-3, lambdaStep=2, lambdaScanCap=2
        ),
        B=corr.mpcov,
        bnorm=mpf(str(corr.central[1] ** 2)),
        correlator=corr,
        energies=[par.e0],
    )
    hlt.prepareHLT()
    hlt.run()
    true_value = float(np.sum(weights / peaks**3))
    assert abs(hlt.rhoResultHLT[0] - true_value) < 3 * hlt.rho_quadrature_err_HLT[0] + 1e-3 * true_value
    assert 0 <= hlt.aa0[0] <= 1
    # The flagged rho, lambda* and g_t come from the same step of the scan
    step = hlt.lambda_list[0].index(hlt.lambdaResultHLT[0])
    assert hlt.rhoResultHLT[0] == pytest.approx(float(hlt.channelA.rho_list[0][step]))
    rho_from_gt = sum(g * mpf(str(c)) for g, c in zip(hlt.gt_HLT[0], corr.central[1:]))
    assert float(rho_from_gt) == pytest.approx(hlt.rhoResultHLT[0], rel=1e-8)
