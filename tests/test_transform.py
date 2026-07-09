"""
Tests for src/lsdensities/transform.py.

Each function here is a linear combination over the tmax time slices; every
test below cross-checks the (vectorised/optimised) implementation against the
direct, unambiguous definition computed with a plain Python loop.
"""

import pytest
from mpmath import mp, mpf

from lsdensities.core import ft_mp, gte
from lsdensities.transform import (
    coefficients_ssd,
    combine_base_scalar,
    combine_fMf_scalar,
    combine_likelihood,
    get_ssd_averaged_scalar,
    get_ssd_scalar,
)
from lsdensities.utils.common import Inputs, _variance_scale_factor


@pytest.fixture
def params():
    par = Inputs()
    par.time_extent = 10
    par.periodicity = "EXP"
    par.kerneltype = "FULLNORMGAUSS"
    par.sigma = 0.3
    par.e0 = 0
    par.num_boot = 4
    par.assign_values()
    return par


@pytest.fixture(autouse=True)
def _set_precision():
    mp.dps = 30


def _ft(estar, i, par, alpha):
    return ft_mp(
        e=estar,
        t=mpf(i + 1),
        sigma_=par.mpsigma,
        alpha=alpha,
        e0=par.mpe0,
        type=par.periodicity,
        T=par.time_extent,
        ker_type=par.kerneltype,
    )


def test_coefficients_ssd_matches_direct_definition(params):
    # Regression test for the O(tmax^2) -> O(tmax) refactor: gt = matrix . f,
    # where f[j] = ft_mp(t=j+1) -- must not depend on the row index.
    mp.dps = 30
    M = mp.randmatrix(params.tmax, params.tmax)
    estar, alpha = mpf("0.5"), mpf("0.1")

    gt = coefficients_ssd(M, params, estar, alpha=alpha)

    gt_ref = mp.matrix(params.tmax, 1)
    for i in range(params.tmax):
        for j in range(params.tmax):
            gt_ref[i] += M[i, j] * _ft(estar, j, params, alpha)

    maxdiff = max(abs(gt[i] - gt_ref[i]) for i in range(params.tmax))
    assert maxdiff < mpf("1e-25")


def test_get_ssd_scalar_matches_dot_product(params):
    gt = mp.randmatrix(params.tmax, 1)
    corr = mp.randmatrix(params.tmax, 1)
    result = get_ssd_scalar(gt, corr, params)
    expected = sum((gt[i] * corr[i] for i in range(params.tmax)), mpf(0))
    assert result == expected


def test_get_ssd_averaged_scalar_matches_manual_average(params):
    gt = mp.randmatrix(params.tmax, 1)
    samples = mp.randmatrix(params.num_boot, params.tmax)

    avg, err = get_ssd_averaged_scalar(gt, samples, params)

    values = [
        sum((gt[i] * samples[b, i] for i in range(params.tmax)), mpf(0))
        for b in range(params.num_boot)
    ]
    expected_avg = sum(values) / params.num_boot
    # get_ssd_averaged_scalar defaults to sample_type="bootstrap": the error is
    # the ddof=1 sample std of the replicates, used directly (see stat_utils.averageScalar_mp).
    expected_err = mp.sqrt(
        sum((v - expected_avg) ** 2 for v in values) / (params.num_boot - 1)
    )
    assert abs(avg - expected_avg) < mpf("1e-25")
    assert abs(err - expected_err) < mpf("1e-25")


@pytest.mark.parametrize("sample_type", ["montecarlo", "bootstrap", "jackknife"])
def test_get_ssd_averaged_scalar_error_scales_with_sample_type(params, sample_type):
    # rho's error must be rescaled by the same sample_type-dependent factor as
    # the correlator's own error (common._variance_scale_factor), otherwise a
    # spectral density built from montecarlo/jackknife samples would silently
    # report a bootstrap-scaled (i.e. wrong) error.
    gt = mp.randmatrix(params.tmax, 1)
    samples = mp.randmatrix(params.num_boot, params.tmax)

    _, err = get_ssd_averaged_scalar(gt, samples, params, sample_type=sample_type)
    _, err_bootstrap = get_ssd_averaged_scalar(
        gt, samples, params, sample_type="bootstrap"
    )

    scale = _variance_scale_factor(sample_type, params.num_boot)
    scale_bootstrap = _variance_scale_factor("bootstrap", params.num_boot)
    expected_ratio = mp.sqrt(mpf(scale) / mpf(scale_bootstrap))

    assert abs(err / err_bootstrap - expected_ratio) < mpf("1e-25")


def test_combine_fMf_scalar_matches_direct_sum(params):
    gt = mp.randmatrix(params.tmax, 1)
    estar, alpha = mpf("0.5"), mpf("0.1")

    result = combine_fMf_scalar(gt, params, estar, alpha)
    expected = sum(
        (gt[i] * _ft(estar, i, params, alpha) for i in range(params.tmax)), mpf(0)
    )
    assert abs(result - expected) < mpf("1e-25")


def test_combine_base_scalar_matches_direct_sum(params):
    gt = mp.randmatrix(params.tmax, 1)
    estar = mpf("0.5")

    result = combine_base_scalar(gt, params, estar)
    expected = sum(
        (
            gt[i] * gte(T=params.time_extent, t=mpf(i + 1), e=estar, periodicity=params.periodicity)
            for i in range(params.tmax)
        ),
        mpf(0),
    )
    assert abs(result - expected) < mpf("1e-25")


def test_combine_likelihood_matches_quadratic_form(params):
    minv = mp.randmatrix(params.tmax, params.tmax)
    corr = mp.randmatrix(params.tmax, 1)

    result = combine_likelihood(minv, params, corr)
    expected = mpf(0)
    for i in range(params.tmax):
        row_dot = sum((minv[i, j] * corr[j] for j in range(params.tmax)), mpf(0))
        expected += row_dot * corr[i]
    assert abs(result - expected) < mpf("1e-25")
