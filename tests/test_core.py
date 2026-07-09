"""
Tests for src/lsdensities/core.py.

Where a closed-form formula exists (a0_scalar, ft_mp for FULLNORMGAUSS/
HALFNORMGAUSS), it is cross-checked against a direct mpmath numerical
integration of its definition, rather than against another copy of the same
closed-form algebra.
"""

import pytest
from mpmath import mp, mpf

from lsdensities.core import a0_array, a0_scalar, ft_mp, gte, cauchy_matrix

KERNEL_TYPES = ["FULLNORMGAUSS", "HALFNORMGAUSS", "CAUCHY"]


def _gauss_full(Ep, e, sigma):
    return mp.exp(-((Ep - e) ** 2) / (2 * sigma**2)) / (sigma * mp.sqrt(2 * mp.pi))


def _gauss_half(Ep, e, sigma):
    norm = sigma * mp.sqrt(mp.pi / 2) * (1 + mp.erf(e / (mp.sqrt(2) * sigma)))
    return mp.exp(-((Ep - e) ** 2) / (2 * sigma**2)) / norm


@pytest.fixture(autouse=True)
def _set_precision():
    mp.dps = 30


def test_cauchy_matrix_symmetric_exp():
    S = cauchy_matrix(9, alpha=mpf("0.3"), e0=mpf("0.1"), type="EXP")
    for i in range(9):
        for j in range(9):
            assert S[i, j] == S[j, i]


def test_cauchy_matrix_symmetric_cosh():
    S = cauchy_matrix(9, alpha=mpf("0.3"), e0=mpf("0.1"), type="COSH", T=20)
    for i in range(9):
        for j in range(9):
            assert S[i, j] == S[j, i]


def test_cauchy_matrix_diagonal_is_positive():
    S = cauchy_matrix(10, alpha=mpf("0.5"), e0=mpf("0"), type="EXP")
    for i in range(10):
        assert S[i, i] > 0


@pytest.mark.parametrize("ker_type", KERNEL_TYPES)
def test_a0_scalar_all_kernel_types_are_finite_and_positive(ker_type):
    # Regression test: a0_scalar used to crash with an UnboundLocalError for
    # ker_type="CAUCHY" (a misplaced `res = mp.quad(...)` after an unconditional raise).
    result = a0_scalar(
        e=mpf("0.8"), sigma=mpf("0.3"), alpha=mpf("0.2"), e0=mpf("0.1"), ker_type=ker_type
    )
    assert mp.isfinite(result)
    assert result > 0


def test_a0_scalar_invalid_kernel_raises():
    with pytest.raises(ValueError):
        a0_scalar(e=mpf("0.5"), sigma=mpf("0.2"), alpha=mpf("0"), ker_type="bogus")


def test_a0_scalar_fullnormgauss_matches_numerical_integral():
    e, sigma, alpha, e0 = mpf("0.8"), mpf("0.3"), mpf("0.4"), mpf("0.1")
    closed = a0_scalar(e=e, sigma=sigma, alpha=alpha, e0=e0, ker_type="FULLNORMGAUSS")
    numeric = mp.quad(
        lambda Ep: mp.exp(alpha * Ep) * _gauss_full(Ep, e, sigma) ** 2, [e0, mp.inf]
    )
    assert abs(closed - numeric) < mpf("1e-25")


def test_a0_scalar_halfnormgauss_matches_numerical_integral():
    e, sigma, alpha, e0 = mpf("0.8"), mpf("0.3"), mpf("0.4"), mpf("0.1")
    closed = a0_scalar(e=e, sigma=sigma, alpha=alpha, e0=e0, ker_type="HALFNORMGAUSS")
    numeric = mp.quad(
        lambda Ep: mp.exp(alpha * Ep) * _gauss_half(Ep, e, sigma) ** 2, [e0, mp.inf]
    )
    assert abs(closed - numeric) < mpf("1e-25")


def test_a0_array_matches_a0_scalar():
    from lsdensities.utils.common import Inputs

    par = Inputs()
    par.mpsigma = mpf("0.3")
    par.e0 = mpf("0.1")
    par.kerneltype = "FULLNORMGAUSS"
    par.Ne = 3
    espace_mp = mp.matrix([mpf("0.3"), mpf("0.6"), mpf("0.9")])

    result = a0_array(espace_mp, par, alpha=mpf("0.2"))
    for i in range(3):
        expected = a0_scalar(
            e=espace_mp[i], sigma=par.mpsigma, alpha=mpf("0.2"), e0=par.e0, ker_type="FULLNORMGAUSS"
        )
        assert result[i] == expected


@pytest.mark.parametrize("ker_type", ["FULLNORMGAUSS", "HALFNORMGAUSS", "CAUCHY"])
def test_ft_mp_all_kernel_types_are_finite(ker_type):
    result = ft_mp(
        e=mpf("0.8"), t=mpf("3"), sigma_=mpf("0.3"), alpha=mpf("0.2"), e0=mpf("0.1"), ker_type=ker_type
    )
    assert mp.isfinite(result)


def test_ft_mp_invalid_kernel_raises():
    with pytest.raises(ValueError):
        ft_mp(e=mpf("0.5"), t=mpf("2"), sigma_=mpf("0.2"), alpha=mpf("0"), ker_type="bogus")


def test_ft_mp_fullnormgauss_matches_numerical_integral():
    e, t, sigma, alpha, e0 = mpf("0.8"), mpf("3"), mpf("0.3"), mpf("0.4"), mpf("0.1")
    closed = ft_mp(e=e, t=t, sigma_=sigma, alpha=alpha, e0=e0, type="EXP", ker_type="FULLNORMGAUSS")
    numeric = mp.quad(
        lambda Ep: mp.exp(alpha * Ep) * _gauss_full(Ep, e, sigma) * mp.exp(-t * Ep), [e0, mp.inf]
    )
    assert abs(closed - numeric) < mpf("1e-25")


def test_ft_mp_halfnormgauss_matches_numerical_integral():
    e, t, sigma, alpha, e0 = mpf("0.8"), mpf("3"), mpf("0.3"), mpf("0.4"), mpf("0.1")
    closed = ft_mp(e=e, t=t, sigma_=sigma, alpha=alpha, e0=e0, type="EXP", ker_type="HALFNORMGAUSS")
    numeric = mp.quad(
        lambda Ep: mp.exp(alpha * Ep) * _gauss_half(Ep, e, sigma) * mp.exp(-t * Ep), [e0, mp.inf]
    )
    assert abs(closed - numeric) < mpf("1e-25")


@pytest.mark.parametrize("ker_type", ["FULLNORMGAUSS", "HALFNORMGAUSS"])
def test_ft_mp_cosh_equals_sum_of_periodic_images(ker_type):
    e, t, sigma, alpha, e0, T = mpf("0.8"), mpf("3"), mpf("0.3"), mpf("0.4"), mpf("0.1"), 20
    cosh_val = ft_mp(e=e, t=t, sigma_=sigma, alpha=alpha, e0=e0, type="COSH", T=T, ker_type=ker_type)
    exp_t = ft_mp(e=e, t=t, sigma_=sigma, alpha=alpha, e0=e0, type="EXP", ker_type=ker_type)
    exp_Tt = ft_mp(e=e, t=mpf(T) - t, sigma_=sigma, alpha=alpha, e0=e0, type="EXP", ker_type=ker_type)
    assert abs(cosh_val - (exp_t + exp_Tt)) < mpf("1e-25")


def test_gte_exp():
    assert gte(T=20, t=mpf(3), e=mpf("0.5"), periodicity="EXP") == mp.exp(-mpf("1.5"))


def test_gte_cosh_is_sum_of_periodic_images():
    T, t, e = 20, mpf(3), mpf("0.5")
    expected = mp.exp(-t * e) + mp.exp(-(T - t) * e)
    assert gte(T=T, t=t, e=e, periodicity="COSH") == expected
