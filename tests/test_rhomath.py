"""
Tests for src/lsdensities/utils/rhoMath.py.
"""

import numpy as np
import pytest
from mpmath import mp, mpf

from lsdensities.utils.rhoMath import (
    cauchy,
    gauss_fp,
    invert_matrix_ge,
    kronecker_fp,
    norm2_mp,
)


def test_kronecker_fp():
    assert kronecker_fp(1, 1) == 1
    assert kronecker_fp(1.0, 1.0) == 1
    assert kronecker_fp(1, 2) == 0


def test_gauss_fp_full_peak_value():
    sigma = 0.2
    # A normalised Gaussian peaks at 1/(sigma sqrt(2 pi))
    expected = 1.0 / (sigma * np.sqrt(2 * np.pi))
    assert gauss_fp(1.0, 1.0, sigma, norm="full") == pytest.approx(expected)


def test_gauss_fp_full_integrates_to_one():
    sigma = 0.3
    x0 = 0.7
    xs = np.linspace(x0 - 20 * sigma, x0 + 20 * sigma, 200000)
    ys = gauss_fp(xs, x0, sigma, norm="full")
    integral = np.trapezoid(ys, xs)
    assert integral == pytest.approx(1.0, abs=1e-6)


def test_gauss_fp_half_integrates_to_one_on_positive_axis():
    # norm="half" is normalised so that the integral from 0 to +inf is 1,
    # for a Gaussian centred at a positive x0 (as used for E >= 0 spectral densities).
    sigma = 0.3
    x0 = 0.7
    xs = np.linspace(0, x0 + 20 * sigma, 200000)
    ys = gauss_fp(xs, x0, sigma, norm="half")
    integral = np.trapezoid(ys, xs)
    assert integral == pytest.approx(1.0, abs=1e-6)


def test_gauss_fp_none_is_unnormalised_peak_one():
    assert gauss_fp(1.0, 1.0, 0.2, norm="none") == pytest.approx(1.0)


def test_gauss_fp_sigma_zero_is_kronecker():
    assert gauss_fp(1.0, 1.0, 0, norm="full") == 1
    assert gauss_fp(1.0, 2.0, 0, norm="full") == 0


def test_gauss_fp_invalid_norm_raises():
    # Regression test: gauss_fp used to silently return None for an unrecognised norm.
    with pytest.raises(ValueError):
        gauss_fp(1.0, 1.0, 0.2, norm="bogus")


def test_cauchy_peak_and_symmetry():
    sigma = 0.15
    omega = 0.6
    # Peak value at k == omega is 1/sigma
    assert cauchy(omega, sigma, omega) == pytest.approx(1.0 / sigma)
    # Symmetric around the peak
    assert cauchy(omega - 0.1, sigma, omega) == pytest.approx(cauchy(omega + 0.1, sigma, omega))


def test_norm2_mp_identity():
    for n in [1, 3, 8]:
        assert float(norm2_mp(mp.eye(n))) == pytest.approx(1.0)


def test_invert_matrix_ge_random_matrix():
    mp.dps = 30
    for n in [3, 10, 20]:
        M = mp.randmatrix(n, n)
        Minv = invert_matrix_ge(M)
        identity = M * Minv
        # Off from the true identity only by working-precision rounding noise.
        assert float(norm2_mp(identity) - 1) < 1e-25


def test_invert_matrix_ge_matches_builtin_inverse():
    mp.dps = 30
    M = mp.randmatrix(12, 12)
    Minv_ge = invert_matrix_ge(M)
    Minv_builtin = M**-1
    maxdiff = max(
        abs(Minv_ge[i, j] - Minv_builtin[i, j]) for i in range(12) for j in range(12)
    )
    assert maxdiff < mpf("1e-25")


def test_invert_matrix_ge_rejects_non_square():
    with pytest.raises(ValueError):
        invert_matrix_ge(mp.matrix(2, 3))
