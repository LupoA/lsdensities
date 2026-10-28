"""Tests of the arbitrary-precision SDP solver (lsdensities.utils.hp_sdp) through CausalBootstrap(solver="MPMATH")."""

import numpy as np
import pytest
from mpmath import mp, mpf

from lsdensities.causal_bootstrap import CausalBootstrap
from lsdensities.utils.hp_sdp import BlockSDP


def uniform_moments(n):
    # rho~ = 1 on [0, 1]: an interior (non-extremal) point of the moment cone
    return np.array([1.0 / (t + 1) for t in range(n)])


def test_tiny_sdp_with_known_optimum():
    # maximize -Tr[diag(1,2) Y] s.t. Tr[Y] = 1, Y >= 0 (1x1 rank-one blocks): optimum -1
    with mp.workdps(40):
        blocks = [{"type": "rank1", "V": [[mpf(1)]], "w": [mpf(1)]},
                  {"type": "rank1", "V": [[mpf(1)]], "w": [mpf(1)]}]
        D = [[[mpf(-1)]], [[mpf(-2)]]]
        result = BlockSDP(blocks, [1], D=D).solve(gap_threshold=mpf("1e-12"), feasibility_threshold=mpf("1e-12"))
        assert result["dual_objective"] == pytest.approx(-1, abs=1e-10)
        assert result["primal_objective"] == pytest.approx(-1, abs=1e-10)


def test_kernel_in_the_span_of_the_data_is_exact():
    # K = lambda^2 is itself a basis function: the bounds collapse to C(2).
    Ct = uniform_moments(5)
    lower, upper = CausalBootstrap(Ct, mp_digits=40).bounds([0.0, 0.0, 1.0], solver="MPMATH")
    assert lower == pytest.approx(Ct[2], abs=1e-12)
    assert upper == pytest.approx(Ct[2], abs=1e-12)


def test_next_moment_bounds_bracket_truth_and_are_certified():
    Ct = uniform_moments(5)
    bootstrap = CausalBootstrap(Ct, mp_digits=40)
    lower, upper = bootstrap.bounds([0.0] * 5 + [1.0], solver="MPMATH")
    assert lower < 1 / 6 < upper
    assert bootstrap.last_mpmath_result["gap"] < mpf("1e-13")


def test_matches_double_precision_where_that_is_reliable():
    Ct = uniform_moments(4)
    coeffs = [0.0, 1.0, -1.5, 0.7]
    hp = CausalBootstrap(Ct, mp_digits=40).bounds(coeffs, solver="MPMATH")
    dp = CausalBootstrap(Ct).bounds(coeffs, solver="CLARABEL")
    assert hp == pytest.approx(dp, abs=1e-5)


def test_noisy_rational_kernel_brackets_truth():
    lam_star, eps = 0.5, 0.1
    Ct = uniform_moments(5)
    covariance = np.diag((1e-3 * Ct) ** 2)
    bootstrap = CausalBootstrap(Ct, covariance=covariance, sigma0=2.0, mp_digits=40)
    lower, upper = bootstrap.bounds([eps], denominator=[lam_star**2 + eps**2, -2 * lam_star, 1.0], solver="MPMATH")
    truth = np.arctan((1 - lam_star) / eps) + np.arctan(lam_star / eps)  # int_0^1 eps/((l-l*)^2+eps^2)
    assert lower < truth < upper
    assert bootstrap.last_mpmath_result["gap"] < mpf("1e-13")


def test_inconsistent_data_is_reported_not_bounded():
    # Moments that no positive measure can produce (C1^2 > C0 C2): the primal
    # problem is infeasible and the solver must fail instead of returning a bound.
    with pytest.raises(RuntimeError):
        CausalBootstrap([1.0, 0.9, 0.5], mp_digits=30).bounds([0.0, 0.0, 0.0, 1.0], solver="MPMATH")
