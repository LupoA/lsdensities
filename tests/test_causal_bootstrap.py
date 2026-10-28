import numpy as np
import pytest

from lsdensities.causal_bootstrap import (
    SUPPORTED_KERNELS,
    CausalBootstrap,
    fit_kernel_polynomial,
    smearing_kernel,
)

TOL = 1e-4


@pytest.fixture(autouse=True)
def double_precision_solver(monkeypatch):
    # The problem construction is tested here with the fast double-precision
    # solver. Several tests use extremal data (single deltas), which the
    # high-precision solver rejects once rounded to double, since no positive
    # measure reproduces them exactly. The high-precision path is tested in
    # tests/test_hp_sdp.py.
    monkeypatch.setattr(CausalBootstrap, "default_solver", "SCS")


def gauss(E, estar, sigma, e0=0.0):
    return smearing_kernel(estar, sigma, "FULLNORMGAUSS", e0=e0)(np.atleast_1d(E))


def fit_gauss(estar, sigma, degree, e0=0.0):
    return fit_kernel_polynomial(smearing_kernel(estar, sigma, e0=e0), degree, e0=e0)


def delta_moments(lambda_peak, Nt):
    return np.array([lambda_peak**t for t in range(Nt)])


def two_delta_moments(lambda1, a1, lambda2, a2, Nt):
    return np.array(
        [a1 * lambda1**t + a2 * lambda2**t for t in range(Nt)]
    )


def multi_delta_moments(lambdas, weights, Nt, t_min=0):
    return np.array(
        [sum(w * lam ** (t_min + i) for w, lam in zip(weights, lambdas)) for i in range(Nt)]
    )


def test_constant_kernel_is_exact():
    # K(lambda) = c0: K[rho] = c0 C(0) for any positive rho, with Nt = 1.
    c0 = 2.5
    C0 = 0.73
    bootstrap = CausalBootstrap([C0])
    lower, upper = bootstrap.bounds([c0])
    true_value = c0 * C0
    assert lower == pytest.approx(true_value, abs=TOL)
    assert upper == pytest.approx(true_value, abs=TOL)


@pytest.mark.parametrize("j,Nt", [(0, 3), (1, 3), (2, 4), (3, 5)])
def test_monomial_kernel_is_exact(j, Nt):
    # K(lambda) = lambda^j: K[rho] = C(j) whenever j < Nt.
    Ct = np.array([0.9**t for t in range(Nt)])
    kernel_coeffs = [0.0] * j + [1.0]
    bootstrap = CausalBootstrap(Ct)
    lower, upper = bootstrap.bounds(kernel_coeffs)
    assert lower == pytest.approx(Ct[j], abs=TOL)
    assert upper == pytest.approx(Ct[j], abs=TOL)


@pytest.mark.parametrize("Nt", [2, 3, 5, 8])
def test_nontrivial_kernel_brackets_single_delta(Nt):
    lambda_peak = 0.4
    Ct = delta_moments(lambda_peak, Nt)
    # K(lambda) = (lambda - 0.5)^2
    kernel_coeffs = [0.25, -1.0, 1.0]
    true_value = (lambda_peak - 0.5) ** 2

    bootstrap = CausalBootstrap(Ct)
    lower, upper = bootstrap.bounds(kernel_coeffs)

    assert lower <= true_value + TOL
    assert upper >= true_value - TOL
    assert lower <= upper + TOL


def test_bracket_width_shrinks_with_more_moments():
    lambda_peak = 0.4
    kernel_coeffs = [0.25, -1.0, 1.0]

    widths = []
    for Nt in range(2, 9):
        Ct = delta_moments(lambda_peak, Nt)
        bootstrap = CausalBootstrap(Ct)
        lower, upper = bootstrap.bounds(kernel_coeffs)
        widths.append(upper - lower)

    for w_next, w_prev in zip(widths[1:], widths[:-1]):
        assert w_next <= w_prev + 1e-4


@pytest.mark.parametrize("degree", [8, 12, 16, 20])
def test_high_degree_kernel_stays_well_conditioned(degree):
    # A kernel of degree up to 20 against a multi-delta rho must solve
    # cleanly (in the monomial basis, the coefficients would span ~10 orders
    # of magnitude at this degree) and bracket the exact value.
    rng = np.random.default_rng(0)
    lambdas = [0.3, 0.55, 0.8]
    weights = [0.6, 0.4, 0.9]
    Ct = multi_delta_moments(lambdas, weights, degree + 6)
    kernel_coeffs = rng.normal(scale=0.3, size=degree + 1) * 0.3 ** np.arange(degree + 1)
    true_value = sum(
        w * np.polynomial.polynomial.polyval(lam, kernel_coeffs)
        for w, lam in zip(weights, lambdas)
    )

    bootstrap = CausalBootstrap(Ct)
    lower, upper = bootstrap.bounds(kernel_coeffs)

    assert lower <= true_value + 1e-3
    assert upper >= true_value - 1e-3
    assert lower <= upper + 1e-3


def test_gaussian_kernel_fit_epsilon_shrinks_with_degree():
    epsilons = [fit_gauss(1.0, 0.3, degree)[1] for degree in [4, 8, 12, 16]]
    for eps_next, eps_prev in zip(epsilons[1:], epsilons[:-1]):
        assert eps_next <= eps_prev


@pytest.mark.parametrize("degree", [8, 12, 16])
def test_gaussian_bounds_brackets_multi_delta_truth(degree):
    estar, sigma = 1.0, 0.35
    lambdas = [0.25, 0.5, 0.7]
    weights = [0.6, 1.1, 0.4]
    Ct = multi_delta_moments(lambdas, weights, 32)
    true_value = sum(
        w * gauss(-np.log(lam), estar, sigma)[0] for w, lam in zip(weights, lambdas)
    )

    bootstrap = CausalBootstrap(Ct)
    lower, upper, diagnostics = bootstrap.gaussian_bounds(estar, sigma, degree)

    assert lower <= true_value + 1e-6
    assert upper >= true_value - 1e-6
    assert diagnostics["epsilon"] > 0
    # For exact data the upper bound on C(0) equals Ct[0], up to the solver
    # tolerance.
    assert diagnostics["inflation"] == pytest.approx(
        diagnostics["epsilon"] * Ct[0], rel=1e-3
    )


def test_gaussian_kernel_fit_vanishes_at_origin():
    coeffs, _epsilon = fit_gauss(1.0, 0.3, 10)
    assert np.polynomial.polynomial.polyval(0.0, coeffs) == pytest.approx(0.0, abs=1e-12)


def test_bounds_stay_finite_without_c0_when_kernel_vanishes_at_origin():
    # With t_min = 1, bounds on a polynomial kernel are finite only if the
    # kernel vanishes at lambda = 0 (E -> infinity), as the fitted one does.
    estar, sigma = 0.8, 0.3
    lambdas = [0.3, 0.55, 0.75]
    weights = [0.5, 0.9, 0.3]
    t_min = 1
    Ct = multi_delta_moments(lambdas, weights, 20, t_min=t_min)
    true_value = sum(
        w * gauss(-np.log(lam), estar, sigma)[0] for w, lam in zip(weights, lambdas)
    )
    poly_coeffs, _epsilon = fit_gauss(estar, sigma, 12)

    bootstrap = CausalBootstrap(Ct, t_min=t_min)
    lower, upper = bootstrap.bounds(poly_coeffs)

    assert np.isfinite(lower) and np.isfinite(upper)
    assert lower <= true_value + 1e-3
    assert upper >= true_value - 1e-3


def test_smeared_bounds_requires_c0():
    bootstrap = CausalBootstrap([0.5, 0.3, 0.2], t_min=1)
    with pytest.raises(ValueError, match="t_min"):
        bootstrap.gaussian_bounds(1.0, 0.3, degree=4)


@pytest.mark.parametrize("Nt", [3, 5, 8])
def test_nontrivial_kernel_brackets_two_delta(Nt):
    lambda1, a1 = 0.3, 0.7
    lambda2, a2 = 0.8, 1.3
    Ct = two_delta_moments(lambda1, a1, lambda2, a2, Nt)
    kernel_coeffs = [0.25, -1.0, 1.0]
    true_value = a1 * (lambda1 - 0.5) ** 2 + a2 * (lambda2 - 0.5) ** 2

    bootstrap = CausalBootstrap(Ct)
    lower, upper = bootstrap.bounds(kernel_coeffs)

    assert lower <= true_value + TOL
    assert upper >= true_value - TOL
    assert lower <= upper + TOL


def test_noisy_data_rejects_bad_covariance_shape():
    with pytest.raises(ValueError):
        CausalBootstrap([1.0, 0.5], covariance=np.eye(3), sigma0=1.0)


def test_noisy_data_requires_sigma0():
    with pytest.raises(ValueError):
        CausalBootstrap([1.0, 0.5], covariance=np.eye(2))


def test_noisy_data_bracket_contains_exact_data_bracket_for_tiny_covariance():
    # As covariance, sigma0 -> 0, the noisy-data dual (Eq. 36) reduces to the
    # exact-data one (Eq. 49).
    lambdas = [0.3, 0.55, 0.8]
    weights = [0.6, 0.4, 0.9]
    Nt = 6
    Ct = multi_delta_moments(lambdas, weights, Nt)
    kernel_coeffs = [0.25, -1.0, 1.0]

    exact = CausalBootstrap(Ct)
    lower_exact, upper_exact = exact.bounds(kernel_coeffs)

    noisy = CausalBootstrap(Ct, covariance=np.eye(Nt) * 1e-10, sigma0=1e-4)
    lower_noisy, upper_noisy = noisy.bounds(kernel_coeffs)

    assert lower_noisy <= lower_exact + 1e-3
    assert upper_noisy >= upper_exact - 1e-3
    assert (upper_noisy - lower_noisy) < 1.0


def test_noisy_data_bracket_has_nominal_statistical_coverage():
    # Noisy realisations of Ct around a known truth: the bracket must contain
    # the true K[rho] at least about as often as the confidence level says.
    rng = np.random.default_rng(1)
    lambdas = [0.3, 0.55, 0.8]
    weights = [0.6, 0.4, 0.9]
    Nt = 6
    Ct_true = multi_delta_moments(lambdas, weights, Nt)
    kernel_coeffs = [0.25, -1.0, 1.0]
    true_value = sum(
        w * np.polynomial.polynomial.polyval(lam, kernel_coeffs)
        for w, lam in zip(weights, lambdas)
    )

    covariance = np.diag((0.05 * np.abs(Ct_true)) ** 2)
    confidence = 0.9
    sigma0 = CausalBootstrap.sigma0_from_confidence(confidence, Nt)

    n_trials = 60
    covered = 0
    for _ in range(n_trials):
        noisy_Ct = rng.multivariate_normal(Ct_true, covariance)
        bootstrap = CausalBootstrap(noisy_Ct, covariance=covariance, sigma0=sigma0)
        try:
            lower, upper = bootstrap.bounds(kernel_coeffs)
        except RuntimeError:
            # A draw outside its own sigma0 ellipsoid: no positive rho is
            # consistent with it (arXiv:2605.20509, Sec. II C), so there is
            # no bracket; counted as not covered.
            continue
        if lower - 1e-8 <= true_value <= upper + 1e-8:
            covered += 1

    # loose floor, given the binomial fluctuations with 60 trials
    assert covered / n_trials >= 0.7


@pytest.mark.parametrize("lam_max", [1.0, 0.6])
def test_restricted_interval_first_moment_bounds(lam_max):
    # Odd-degree branch of Lemma 5 on [0, b]: given C(0) = 1 only, C(1)
    # ranges over [0, b].
    bootstrap = CausalBootstrap([1.0], e0=-np.log(lam_max))
    lower, upper = bootstrap.bounds([0.0, 1.0])
    assert lower == pytest.approx(0.0, abs=TOL)
    assert upper == pytest.approx(lam_max, abs=TOL)


@pytest.mark.parametrize("lam_max", [1.0, 0.6])
def test_restricted_interval_second_moment_bounds(lam_max):
    # Even-degree branch of Lemma 5 on [0, b]: given C(0) = 1, C(1) = m,
    # C(2) ranges over [m^2, m b].
    m = 0.3
    bootstrap = CausalBootstrap([1.0, m], e0=-np.log(lam_max))
    lower, upper = bootstrap.bounds([0.0, 0.0, 1.0])
    assert lower == pytest.approx(m**2, abs=TOL)
    assert upper == pytest.approx(m * lam_max, abs=TOL)


def test_gaussian_bounds_with_e0_brackets_truth():
    e0 = 0.3
    estar, sigma = 0.9, 0.35
    lambdas = [0.2, 0.45, 0.7]  # E = 1.61, 0.80, 0.36 > e0
    weights = [0.6, 1.1, 0.4]
    Ct = multi_delta_moments(lambdas, weights, 24)
    true_value = sum(
        w * gauss(-np.log(lam), estar, sigma, e0=e0)[0] for w, lam in zip(weights, lambdas)
    )
    lower, upper, diagnostics = CausalBootstrap(Ct, e0=e0).gaussian_bounds(
        estar, sigma, 16
    )
    assert lower <= true_value + 1e-6
    assert upper >= true_value - 1e-6
    # On [0, 1] the fit would have to follow the jump of the kernel at
    # E = e0, so epsilon could not be smaller than half of it.
    jump = gauss(e0, estar, sigma)[0]
    assert diagnostics["epsilon"] < 0.1 * (0.5 * jump)


@pytest.mark.parametrize("kerneltype", SUPPORTED_KERNELS)
def test_every_supported_kernel_brackets_truth(kerneltype):
    estar, sigma = 0.9, 0.3
    lambdas = [0.25, 0.5, 0.7]
    weights = [0.6, 1.1, 0.4]
    Ct = multi_delta_moments(lambdas, weights, 24)
    kernel = smearing_kernel(estar, sigma, kerneltype)
    true_value = sum(w * kernel(np.array([-np.log(lam)]))[0] for w, lam in zip(weights, lambdas))
    lower, upper, diagnostics = CausalBootstrap(Ct).smeared_bounds(kernel, 16)
    assert lower <= true_value + 1e-6
    assert upper >= true_value - 1e-6
    poly_at_zero = np.polynomial.polynomial.polyval(0.0, diagnostics["poly_coeffs"])
    assert poly_at_zero == pytest.approx(0.0, abs=1e-10)


def test_unsupported_kernel_is_rejected():
    with pytest.raises(ValueError, match="LINEAR"):
        smearing_kernel(1.0, 0.3, "LINEAR")


def test_cosh_basis_matrix_matches_lambda_t_plus_lambda_T_minus_t():
    T, Nt = 16, 8
    bootstrap = CausalBootstrap(np.ones(Nt), periodicity="COSH", time_extent=T)
    degree = bootstrap._basis_degree()
    assert degree == T
    basis_leg = bootstrap._legendre_basis_matrix(degree)
    lam = np.linspace(0.0, 1.0, 101)
    for t in range(Nt):
        values = np.polynomial.Legendre(basis_leg[:, t], domain=[0, 1])(lam)
        np.testing.assert_allclose(values, lam**t + lam ** (T - t), atol=1e-12)


def test_cosh_requires_time_extent_and_half_lattice_data():
    with pytest.raises(ValueError, match="time_extent"):
        CausalBootstrap(np.ones(4), periodicity="COSH")
    with pytest.raises(ValueError, match="T/2"):
        CausalBootstrap(np.ones(10), periodicity="COSH", time_extent=16)


def test_cosh_gaussian_bounds_bracket_truth():
    # C(t) = sum_n w_n (e^{-E_n t} + e^{-E_n (T-t)}). The same data read as
    # EXP are not the moments of any positive measure (the backward term is
    # large at small E and T), so that case must fail.
    T, Nt = 16, 9
    estar, sigma = 0.6, 0.3
    energies = np.array([0.15, 0.5, 0.9])
    weights = np.array([0.3, 1.0, 0.6])
    Ct = np.array(
        [np.sum(weights * (np.exp(-energies * t) + np.exp(-energies * (T - t))))
         for t in range(Nt)]
    )
    true_value = np.sum(weights * gauss(energies, estar, sigma))
    lower, upper, _ = CausalBootstrap(
        Ct, periodicity="COSH", time_extent=T
    ).gaussian_bounds(estar, sigma, 12)
    assert lower <= true_value + 1e-6
    assert upper >= true_value - 1e-6
    with pytest.raises(RuntimeError):
        CausalBootstrap(Ct).gaussian_bounds(estar, sigma, 12)


def test_rational_kernel_with_unit_denominator_matches_polynomial():
    Ct = two_delta_moments(0.3, 0.7, 0.8, 0.5, 6)
    coeffs = [0.0, 1.0, -0.5, 0.2]
    bootstrap = CausalBootstrap(Ct)
    assert bootstrap.bounds(coeffs, denominator=[1.0]) == pytest.approx(
        bootstrap.bounds(coeffs), abs=TOL
    )


def test_rational_cauchy_kernel_is_exact_for_a_single_delta():
    # A single delta is fixed by its moments, so the bounds on a rational
    # kernel (Eq. 52, no approximation) collapse to its value there.
    lam0, lam_star, eps = 0.6, 0.55, 0.1
    Ct = delta_moments(lam0, 8)
    numerator = [eps]
    denominator = [lam_star**2 + eps**2, -2 * lam_star, 1.0]  # (lam - lam*)^2 + eps^2
    lower, upper = CausalBootstrap(Ct).bounds(numerator, denominator=denominator)
    exact = eps / ((lam0 - lam_star) ** 2 + eps**2)
    assert lower == pytest.approx(exact, rel=1e-3)
    assert upper == pytest.approx(exact, rel=1e-3)


def test_rational_cauchy_kernel_brackets_two_deltas():
    lam_star, eps = 0.5, 0.1
    Ct = two_delta_moments(0.3, 0.7, 0.8, 0.5, 8)
    denominator = [lam_star**2 + eps**2, -2 * lam_star, 1.0]
    lower, upper = CausalBootstrap(Ct).bounds([eps], denominator=denominator)
    exact = sum(w * eps / ((lam - lam_star) ** 2 + eps**2) for lam, w in [(0.3, 0.7), (0.8, 0.5)])
    assert lower <= exact + TOL
    assert upper >= exact - TOL


def test_rational_kernel_rejects_nonpositive_denominator():
    with pytest.raises(ValueError, match="denominator"):
        CausalBootstrap([1.0, 0.5]).bounds([1.0], denominator=[-0.25, 0.0, 1.0])


def test_extremal_measure_attains_bounds():
    # The extremal measures (arXiv:2605.20509, Sec. III B) are positive sums
    # of deltas reproducing the bounds, with moments on the boundary of the
    # chi^2 ellipsoid.
    E = np.array([0.3, 0.7, 1.2])
    a = np.array([1.0, 0.8, 0.5])
    t = np.arange(12)
    C = (a * np.exp(-np.outer(t, E))).sum(1)
    cov = np.diag((0.01 * C) ** 2)
    Chat = C + np.random.default_rng(1).multivariate_normal(np.zeros(len(C)), cov)
    sigma0 = CausalBootstrap.sigma0_from_confidence(0.9, len(C))
    cb = CausalBootstrap(Chat, covariance=cov, sigma0=sigma0, mp_digits=60)
    coeffs, _ = fit_gauss(0.7, 0.3, 10)
    lower = cb.extremal_measure(coeffs, "lower")
    upper = cb.extremal_measure(coeffs, "upper")
    assert lower["bound_value"] < upper["bound_value"]
    for m in (lower, upper):
        assert 0 < len(m["energies"]) <= len(C) // 2 + 1
        assert min(m["amplitudes"]) > 0
        assert m["measure_value"] == pytest.approx(m["bound_value"], rel=1e-12)
        assert m["relative_residual"] < 1e-12
        assert m["chi2"] == pytest.approx(sigma0**2, rel=1e-10)
