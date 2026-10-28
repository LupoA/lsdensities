"""
Rigorous bounds on smeared spectral densities from positivity of rho(E) and
Euclidean correlator data: the "causal bootstrap" of arXiv:2605.20509
("The Causal Bootstrap: Bounding Smeared Spectral Functions from
Non-Perturbative Euclidean Data").

The problem is formulated in the Hausdorff-moment variable lambda = exp(-E)
(arXiv:2605.20509, Sec. II B, Eqs. 9-10), in which C(t) = int lambda^t
rho~(lambda) dlambda. A smearing kernel K(E) which is not a polynomial in
lambda is approximated by one, and the approximation error is bounded
separately (end of Sec. IV A); see `fit_kernel_polynomial`.
"""

import contextlib
import functools

import numpy as np
import cvxpy as cp
from mpmath import mp, mpf
from numpy.polynomial import Legendre, Polynomial
from numpy.polynomial.legendre import legmul

from .utils.math_utils import cauchy, gauss_fp, theta_erf

__all__ = [
    "CausalBootstrap",
    "smearing_kernel",
    "fit_kernel_polynomial",
    "SUPPORTED_KERNELS",
]

_DOMAIN = [0.0, 1.0]

#: Smearing kernels (par.kerneltype) that can be bounded: they are bounded on
#: E >= e0 and vanish as E -> infinity (see `fit_kernel_polynomial`).
SUPPORTED_KERNELS = ("FULLNORMGAUSS", "HALFNORMGAUSS", "CAUCHY", "THETA-ERF")


def smearing_kernel(estar, sigma, kerneltype="FULLNORMGAUSS", e0=0.0):
    """
    The smearing kernel K(E) = Delta_sigma(estar, E) of core.ft_mp, as a
    numpy function of the energy E, set to zero for E < e0. Only the target
    observable rho_sigma(estar) = int K(E) rho(E) dE is defined here: the
    factor exp(alpha E) of the HLT norm plays no role in the bounds.
    """
    if kerneltype == "FULLNORMGAUSS":

        def kernel(E):
            return gauss_fp(E, estar, sigma, norm="full")
    elif kerneltype == "HALFNORMGAUSS":

        def kernel(E):
            return gauss_fp(E, estar, sigma, norm="half")
    elif kerneltype == "CAUCHY":

        def kernel(E):
            return cauchy(E, sigma, estar)
    elif kerneltype == "THETA-ERF":

        def kernel(E):
            return theta_erf(estar, E, sigma)
    else:
        raise ValueError(
            f"Causal-bootstrap bounds are not available for kerneltype '{kerneltype}' "
            f"(supported: {', '.join(SUPPORTED_KERNELS)})"
        )

    def restricted(E):
        E = np.asarray(E, dtype=float)
        out = np.zeros_like(E)
        mask = e0 <= E
        out[mask] = kernel(E[mask])
        return out

    return restricted


def fit_kernel_polynomial(kernel, degree, e0=0.0, n_samples=4000):
    """
    Approximates the kernel K(E) (a numpy function of E, e.g. from
    `smearing_kernel`) by a polynomial of degree `degree` in lambda = exp(-E)
    on [0, exp(-e0)], following the remark at the end of arXiv:2605.20509,
    Sec. IV A. The least-squares fit is done in the shifted Legendre basis on
    [0, 1], since a localised kernel has large, cancelling monomial
    coefficients already at moderate degree.

    The fit is constrained to vanish at lambda = 0 (E -> infinity), like the
    kernels it approximates. A polynomial kernel with K(0) != 0 has no finite
    bound unless C(0) is part of the data: a positive rho can then carry an
    arbitrary amount of weight at E -> infinity, which is invisible to every
    C(t >= 1) but not to K[rho].

    Returns (coeffs, epsilon): the monomial coefficients of the fit (lowest
    power first), and an estimate of
        epsilon = sup_{lambda in [0, exp(-e0)]} |K(-log lambda) - K_poly(lambda)|
    from a dense grid plus local refinement (with a 2% margin). Since rho >= 0,
        |K[rho] - K_poly[rho]| <= epsilon * C(0),
    which is how `CausalBootstrap.smeared_bounds` turns the bounds on
    K_poly[rho] into bounds on K[rho].
    """
    lam_max = np.exp(-e0)
    lam_fit = np.linspace(1e-10, lam_max, n_samples)
    target = kernel(-np.log(lam_fit))

    # Least squares in the shifted Legendre basis, with the linear constraint
    # sum_k a_k P_k(0) = 0, P_k(0) = (-1)^k, imposed by a Lagrange multiplier.
    V = np.polynomial.legendre.legvander(2.0 * lam_fit - 1.0, degree)
    zero_at_origin = (-1.0) ** np.arange(degree + 1)
    a_unconstrained, *_ = np.linalg.lstsq(V, target, rcond=None)
    VtV_inv_c = np.linalg.solve(V.T @ V, zero_at_origin)
    lagrange = (zero_at_origin @ a_unconstrained) / (zero_at_origin @ VtV_inv_c)
    legendre_coeffs = a_unconstrained - lagrange * VtV_inv_c

    poly = Legendre(legendre_coeffs, domain=_DOMAIN).convert(
        kind=Polynomial, domain=_DOMAIN, window=_DOMAIN
    )
    coeffs = np.asarray(poly.coef, dtype=float)

    def _residual(lam):
        lam = np.clip(lam, 1e-300, lam_max)
        return np.abs(kernel(-np.log(lam)) - poly(lam))

    lam_grid = np.linspace(0.0, lam_max, n_samples)
    residual_grid = _residual(lam_grid)
    epsilon = float(np.max(residual_grid))

    from scipy.optimize import minimize_scalar

    i_max = int(np.argmax(residual_grid))
    lo = lam_grid[max(i_max - 1, 0)]
    hi = lam_grid[min(i_max + 1, n_samples - 1)]
    refined = minimize_scalar(
        lambda lam: -_residual(np.array([lam]))[0], bounds=(lo, hi), method="bounded"
    )
    epsilon = max(epsilon, float(-refined.fun))
    epsilon *= 1.02

    return coeffs, epsilon


def _solve_cvxpy(problem, solver):
    """problem.solve, with solver failures reported as RuntimeError like
    every other failure to find a bound."""
    try:
        problem.solve(solver=solver)
    except cp.error.SolverError as error:
        raise RuntimeError(f"cvxpy solver {solver} failed: {error}") from error


class CausalBootstrap:
    """
    Bounds on a smeared spectral observable
        K[rho] = int_e0^inf K(E) rho(E) dE,
    valid for every rho(E) >= 0 compatible with the data, for a scalar
    correlator (arXiv:2605.20509).

    The basis functions are b_t(lambda) = lambda^t (EXP) or
    lambda^t + lambda^(T-t) (COSH), and the correlator values
    Ct[i] = C(t_min + i). The kernel is a polynomial (or a rational function
    with positive denominator, Problem IV.3, Eq. 52) in lambda; general
    smearing kernels are handled by `smeared_bounds`. Internally, every
    polynomial is expanded in the shifted Legendre basis on [0, 1], which
    keeps the coefficients O(1) up to the degrees needed here.

    Exact data are treated as in Problem IV.2 (Eq. 49). With `covariance`
    and `sigma0`, the data are only required to lie in the ellipsoid
    ||C[rho] - Ct||_covariance <= sigma0 (Problem III.2, SDP form Eq. 36),
    and the bounds hold at the confidence level that defines sigma0 (see
    `sigma0_from_confidence`).

    rho(E) = 0 for E < e0 is assumed, so the positivity certificate is
    imposed on lambda in [0, exp(-e0)] rather than [0, 1] (as done with
    m_gap in Sec. VI C).
    """

    #: Solver used when none is given. "MPMATH" is the arbitrary-precision
    #: interior-point method of utils/hp_sdp.py (arXiv:2605.20509, App. B),
    #: with a certified duality gap. Any other value is passed to cvxpy as a
    #: double-precision solver name (e.g. "CLARABEL", "SCS"): faster, but
    #: double precision can give loose or wrong bounds on these problems,
    #: so it is only meant for quick exploration.
    default_solver = "MPMATH"

    def __init__(
        self,
        Ct,
        covariance=None,
        sigma0=None,
        t_min=0,
        e0=0.0,
        periodicity="EXP",
        time_extent=None,
        mp_digits=100,
    ):
        """
        Ct[i] is the correlator at Euclidean time t = t_min + i. e0 is the
        lower edge of the spectral support. periodicity is "EXP" or "COSH",
        as in core.ft_mp; "COSH" requires time_extent. The COSH basis is
        still polynomial in lambda, so the reduction to a finite SDP stays
        exact (Sec. IV); only the degree of the certificate grows, to T.
        mp_digits is the working precision of the "MPMATH" solver.
        """
        if e0 < 0:
            raise ValueError(f"e0 must be >= 0, got {e0}")
        if periodicity not in ("EXP", "COSH"):
            raise ValueError(f"periodicity must be 'EXP' or 'COSH', got '{periodicity}'")
        if periodicity == "COSH" and time_extent is None:
            raise ValueError("time_extent is required for periodicity='COSH'")
        self.periodicity = periodicity
        self.mp_digits = int(mp_digits)
        self.time_extent = None if time_extent is None else int(time_extent)
        self.e0 = float(e0)
        self.lam_max = float(np.exp(-self.e0))
        self.Ct = np.asarray(Ct, dtype=float)
        if self.Ct.ndim != 1 or len(self.Ct) == 0:
            raise ValueError("Ct must be a non-empty 1D sequence of moments")
        self.Nt = len(self.Ct)
        self.t_min = int(t_min)
        if self.periodicity == "COSH" and 2 * (self.t_min + self.Nt - 1) > self.time_extent:
            raise ValueError(
                "COSH data must not extend past T/2 (the correlator is symmetric "
                f"there): got t up to {self.t_min + self.Nt - 1} with T={self.time_extent}"
            )

        if covariance is None:
            self.covariance = None
            self.sigma0 = None
        else:
            covariance = np.asarray(covariance, dtype=float)
            if covariance.shape != (self.Nt, self.Nt):
                raise ValueError(
                    f"covariance must have shape ({self.Nt}, {self.Nt}), "
                    f"got {covariance.shape}"
                )
            if sigma0 is None:
                raise ValueError("sigma0 (statistical consistency cutoff) is "
                                 "required when covariance is given")
            self.covariance = covariance
            self.sigma0 = float(sigma0)
        self._c0_upper = {}  # solver -> upper bound on C(0), see smeared_bounds

    @staticmethod
    def sigma0_from_confidence(confidence, Nt):
        """
        sigma0 such that the ellipsoid ||C[rho] - Ct||_covariance <= sigma0
        contains the true correlator with probability `confidence`, for Ct
        Gaussian-distributed around it (arXiv:2605.20509, Sec. II D):
        ||Delta C||^2 ~ chi2(Nt), so sigma0 = sqrt(chi2.ppf(confidence, Nt)).
        """
        from scipy.stats import chi2

        return float(np.sqrt(chi2.ppf(confidence, df=Nt)))

    @staticmethod
    def _to_legendre(monomial_coeffs, degree):
        """
        Shifted-Legendre coefficients (length degree+1) of the polynomial
        with the given monomial coefficients (lowest power first).
        """
        padded = np.zeros(degree + 1)
        padded[: len(monomial_coeffs)] = monomial_coeffs
        poly = Polynomial(padded, domain=_DOMAIN, window=_DOMAIN)
        leg_coef = Legendre.cast(poly, domain=_DOMAIN).coef
        # Legendre.cast trims trailing zeros
        out = np.zeros(degree + 1)
        out[: len(leg_coef)] = leg_coef[: degree + 1]
        return out

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _product_matrix(n, multiplier):
        """
        Constant (len, n*n) matrix M with M[k, i*n + j] = k-th Legendre
        coefficient of P_i P_j f, where P_n is the shifted Legendre
        polynomial of degree n on [0, 1], i, j < n, and f = `multiplier` is a
        small polynomial in monomial form (a tuple, lowest power first; e.g.
        (0, b, -1) for lambda (b - lambda)). len = 2n - 1 + deg f.
        M @ vec(Y) are then the Legendre coefficients of f q^T Y q, with
        q = (P_0, ..., P_(n-1)).
        """
        f_leg = CausalBootstrap._to_legendre(list(multiplier), len(multiplier) - 1)
        length = 2 * n - 1 + len(multiplier) - 1
        M = np.zeros((length, n * n))
        for i in range(n):
            for j in range(i, n):
                unit_i = np.zeros(i + 1)
                unit_i[i] = 1.0
                unit_j = np.zeros(j + 1)
                unit_j[j] = 1.0
                prod = legmul(legmul(unit_i, unit_j), f_leg)
                M[: len(prod), i * n + j] = prod
                M[: len(prod), j * n + i] = prod
        return M

    @classmethod
    def _sos_nonneg_on_interval(cls, degree, lam_max=1.0):
        """
        Representation of a polynomial of degree `degree` which is >= 0 on
        [0, b], b = `lam_max` (arXiv:2605.20509, App. A, Lemma 5, mapped from
        [0, 1] to [0, b] by a change of variables):

          degree = 2m:    p = q^T Y1 q + lambda (b - lambda) q'^T Y2 q'
                          q = (P_0,...,P_m), q' = (P_0,...,P_(m-1))
          degree = 2m+1:  p = lambda q^T Y1 q + (b - lambda) q^T Y2 q
                          q = (P_0,...,P_m)

        with Y1, Y2 positive semidefinite. Returns (coeffs, psd_vars):
        the Legendre coefficients of p as a cvxpy expression of length
        degree+1, and the PSD variables they are built from.
        """
        if degree % 2 == 0:
            m = degree // 2
            blocks = [(m + 1, (1.0,))]
            if m > 0:
                blocks.append((m, (0.0, lam_max, -1.0)))
        else:
            m = (degree - 1) // 2
            blocks = [(m + 1, (0.0, 1.0)), (m + 1, (lam_max, -1.0))]

        coeffs = 0
        psd_vars = []
        for n, multiplier in blocks:
            Y = cp.Variable((n, n), PSD=True)
            M = cls._product_matrix(n, multiplier)
            coeffs = coeffs + M @ cp.reshape(Y, (n * n,), order="C")
            psd_vars.append(Y)
        return coeffs, psd_vars

    def _basis_degree(self, denominator=(1.0,)):
        """Highest power of lambda among the basis functions, times the
        kernel denominator."""
        t_max = self.t_min + self.Nt - 1
        if self.periodicity == "COSH":
            t_max = max(t_max, self.time_extent - self.t_min)
        return t_max + len(denominator) - 1

    def _legendre_basis_matrix(self, degree, denominator=(1.0,)):
        """(degree+1, Nt) matrix whose column i holds the Legendre
        coefficients of d(lambda) b_t(lambda), t = t_min + i, with d the
        kernel denominator (monomial coefficients, lowest power first)."""
        basis_leg = np.zeros((degree + 1, self.Nt))
        for i in range(self.Nt):
            t = self.t_min + i
            powers = [t] if self.periodicity == "EXP" else [t, self.time_extent - t]
            for power in powers:
                unit = np.zeros(power + 1)
                unit[power] = 1.0
                product = np.polynomial.polynomial.polymul(unit, denominator)
                basis_leg[:, i] += self._to_legendre(product, degree)
        return basis_leg

    def _check_denominator(self, denominator):
        """Eq. 52 requires d > 0 on the whole support."""
        lam = np.linspace(0.0, self.lam_max, 10001)
        if np.polynomial.polynomial.polyval(lam, denominator).min() <= 0:
            raise ValueError("Kernel denominator must be > 0 on [0, exp(-e0)]")

    def _dual_bound(self, kernel_coeffs, solver=None, denominator=(1.0,)):
        if solver is None:
            solver = self.default_solver
        if solver == "MPMATH":
            return self._dual_bound_mpmath(kernel_coeffs, denominator)
        if self.covariance is None:
            return self._dual_bound_exact(kernel_coeffs, solver=solver, denominator=denominator)
        return self._dual_bound_noisy(kernel_coeffs, solver=solver, denominator=denominator)

    def _dual_bound_exact(self, kernel_coeffs, solver=None, denominator=(1.0,)):
        """
        Exact-data dual problem (arXiv:2605.20509, Problem IV.2, Eq. 49):

            maximize_{g_t}   sum_t g_t Ct
            subject to       K(lambda) - sum_t g_t b_t(lambda) >= 0  on [0, exp(-e0)]

        For a rational kernel K = n/d, d > 0, the constraint is multiplied
        by d (Problem IV.3, Eq. 52). Returns (bound, g, status), where bound
        is a lower bound on K[rho] for every rho >= 0 reproducing Ct.
        """
        kernel_coeffs = np.asarray(kernel_coeffs, dtype=float)
        degree = max(len(kernel_coeffs) - 1, self._basis_degree(denominator))

        kernel_leg = self._to_legendre(kernel_coeffs, degree)
        basis_leg = self._legendre_basis_matrix(degree, denominator)

        g = cp.Variable(self.Nt)
        sos_coeffs, _psd_vars = self._sos_nonneg_on_interval(degree, self.lam_max)

        constraints = [sos_coeffs == kernel_leg - basis_leg @ g]

        problem = cp.Problem(cp.Maximize(g @ self.Ct), constraints)
        _solve_cvxpy(problem, solver)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"Causal-bootstrap dual SDP did not solve to optimality: {problem.status}"
            )
        return float(problem.value), np.asarray(g.value, dtype=float), problem.status

    def _dual_bound_noisy(self, kernel_coeffs, solver=None, denominator=(1.0,)):
        """
        Noisy-data dual problem (arXiv:2605.20509, Problem III.2, SDP form
        Eq. 36):

            maximize_{Z}     -Z . Gamma0
            subject to       K(lambda) - sum_t g_t b_t(lambda) >= 0  on [0, exp(-e0)]
                             g_t = 2 Z[t, Nt]     (Eq. 35)
                             Z >= 0,  (Nt+1) x (Nt+1)

        with Gamma0 = [[covariance, -Ct], [-Ct^T, sigma0^2]] (Eq. 34).
        Rational kernels as in `_dual_bound_exact`. Returns (bound, g, status).
        """
        kernel_coeffs = np.asarray(kernel_coeffs, dtype=float)
        degree = max(len(kernel_coeffs) - 1, self._basis_degree(denominator))

        kernel_leg = self._to_legendre(kernel_coeffs, degree)
        basis_leg = self._legendre_basis_matrix(degree, denominator)

        # Each C(t) is rescaled by |Ct|, which leaves the optimum unchanged
        # (g_t -> g_t / scale_t, and the chi^2 norm is invariant under a
        # diagonal rescaling of the data) but avoids entries of Gamma0 that
        # differ by many orders of magnitude.
        scale = np.where(self.Ct != 0, np.abs(self.Ct), 1.0)
        Ct_scaled = self.Ct / scale
        covariance_scaled = self.covariance / np.outer(scale, scale)
        basis_leg_scaled = basis_leg / scale[np.newaxis, :]

        N = self.Nt
        Z = cp.Variable((N + 1, N + 1), PSD=True)
        g_scaled = 2.0 * Z[:N, N]

        sos_coeffs, _psd_vars = self._sos_nonneg_on_interval(degree, self.lam_max)
        constraints = [sos_coeffs == kernel_leg - basis_leg_scaled @ g_scaled]

        Gamma0 = np.zeros((N + 1, N + 1))
        Gamma0[:N, :N] = covariance_scaled
        Gamma0[:N, N] = -Ct_scaled
        Gamma0[N, :N] = -Ct_scaled
        Gamma0[N, N] = self.sigma0**2

        problem = cp.Problem(cp.Maximize(-cp.trace(Z @ Gamma0)), constraints)
        _solve_cvxpy(problem, solver)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"Causal-bootstrap noisy-data dual SDP did not solve to "
                f"optimality: {problem.status}"
            )
        g_value = np.asarray(g_scaled.value, dtype=float) / scale
        return float(problem.value), g_value, problem.status

    def _dual_bound_mpmath(self, kernel_coeffs, denominator=(1.0,)):
        """
        The dual problem of `_dual_bound_exact` / `_dual_bound_noisy`, solved
        by the arbitrary-precision primal-dual interior-point method of
        utils/hp_sdp.py (arXiv:2605.20509, App. B) at `self.mp_digits`
        digits, to a duality gap and infeasibility below 10^-(mp_digits/3).

        The polynomial identity "n - d sum_t g_t b_t = positive form of
        Lemma 5" is imposed at deg+1 Chebyshev points of [0, exp(-e0)], which
        makes every constraint matrix of the positive blocks rank one
        (Eqs. 54-55 and App. B). The double-precision inputs are converted
        exactly to mpf, so the result is the optimum of the SDP for exactly
        those inputs. Returns (bound, g, status).
        """
        from .utils.hp_sdp import BlockSDP, SDPError

        denominator = tuple(float(c) for c in denominator)
        degree = max(len(kernel_coeffs) - 1, self._basis_degree(denominator))
        if self.covariance is None:
            # At least one sample point more than free variables g_t, so that
            # a constraint is left once they are eliminated (hp_sdp); raising
            # the degree does not change the bound.
            degree = max(degree, self.Nt)
        threshold = mpf(10) ** (-(self.mp_digits // 3))
        with mp.workdps(self.mp_digits):
            b = mpf(self.lam_max)
            P = degree + 1
            points = [b / 2 * (1 - mp.cos(mp.pi * (2 * k + 1) / (2 * P))) for k in range(P)]

            def poly(coeffs, x):
                return mp.fsum(mpf(float(c)) * x**k for k, c in enumerate(coeffs))

            def chebyshev_basis(n, x):  # T_0..T_{n-1} at 2x/b - 1
                z = 2 * x / b - 1
                out = [mpf(1), z][:n]
                while len(out) < n:
                    out.append(2 * z * out[-1] - out[-2])
                return out

            # positive blocks of Lemma 5 on [0, b]: (size, weight function)
            if degree % 2 == 0:
                m = degree // 2
                specs = [(m + 1, lambda x: mpf(1))] + ([(m, lambda x: x * (b - x))] if m > 0 else [])
            else:
                m = (degree - 1) // 2
                specs = [(m + 1, lambda x: x), (m + 1, lambda x: b - x)]
            blocks = []
            for size, weight in specs:
                columns = [chebyshev_basis(size, x) for x in points]
                blocks.append({"type": "rank1", "V": [[col[i] for col in columns] for i in range(size)],
                               "w": [weight(x) for x in points]})

            c = [poly(kernel_coeffs, x) for x in points]
            scale = [mpf(abs(float(v))) if v != 0 else mpf(1) for v in self.Ct]

            def basis_value(t, x):
                value = x**t if self.periodicity == "EXP" else x**t + x ** (self.time_extent - t)
                return poly(denominator, x) * value

            F = [[basis_value(self.t_min + i, x) / scale[i] for i in range(self.Nt)] for x in points]

            if self.covariance is None:
                sdp = BlockSDP(blocks, c, B=F, b=[mpf(float(v)) / s for v, s in zip(self.Ct, scale)])
            else:
                N = self.Nt
                U = [[F[p][t] for p in range(P)] for t in range(N)] + [[mpf(0)] * P]
                Gamma0 = [[mpf(0)] * (N + 1) for _ in range(N + 1)]
                for i in range(N):
                    for j in range(N):
                        Gamma0[i][j] = mpf(float(self.covariance[i][j])) / (scale[i] * scale[j])
                    Gamma0[i][N] = Gamma0[N][i] = -mpf(float(self.Ct[i])) / scale[i]
                Gamma0[N][N] = mpf(self.sigma0) ** 2
                zeros = [[[mpf(0)] * n for _ in range(n)] for n, _ in specs]
                sdp = BlockSDP(blocks + [{"type": "rank2", "U": U}], c,
                               D=zeros + [[[-v for v in row] for row in Gamma0]])
            try:
                result = sdp.solve(gap_threshold=threshold, feasibility_threshold=threshold)
            except SDPError as error:
                raise RuntimeError(f"High-precision causal-bootstrap SDP failed: {error}") from error
            if self.covariance is None:
                g = [result["y"][i] / scale[i] for i in range(self.Nt)]
            else:
                Z = result["Y"][-1]
                g = [2 * Z[i][self.Nt] / scale[i] for i in range(self.Nt)]
            self.last_mpmath_result = result
            status = f"optimal (gap {mp.nstr(result['gap'], 3)}, {result['iterations']} iterations)"
            return float(result["dual_objective"]), np.array([float(v) for v in g]), status

    def bounds(self, kernel_coeffs, solver=None, denominator=(1.0,)):
        """
        (lower, upper) bounds on
            K[rho] = int_0^exp(-e0) K(lambda) rho~(lambda) dlambda
        for K = n/d, n = sum_k kernel_coeffs[k] lambda^k and
        d = sum_k denominator[k] lambda^k (default 1, a polynomial kernel).
        d must be > 0 on the support; no approximation is involved
        (arXiv:2605.20509, Problem IV.3, Eq. 52).
        """
        denominator = tuple(float(c) for c in denominator)
        self._check_denominator(denominator)
        lower, _, _ = self._dual_bound(kernel_coeffs, solver=solver, denominator=denominator)
        neg_upper, _, _ = self._dual_bound(
            [-c for c in kernel_coeffs], solver=solver, denominator=denominator
        )
        return lower, -neg_upper

    def extremal_measure(self, kernel_coeffs, bound="lower", e_max=20.0, n_grid=4000, rel_tol=1e-12):
        """
        The spectral measure attaining the lower (or upper) bound on a
        polynomial kernel (arXiv:2605.20509, Sec. III B, Eqs. 30-32), from
        the optimal dual certificate of the high-precision solver.

        By complementary slackness, the optimal rho~ is supported on the zeros
        of the certificate p(lambda) = +-K(lambda) - sum_t g_t b_t(lambda) >= 0
        on [0, exp(-e0)], i.e. it is a finite sum of delta functions. Its
        moments are
            C* = C_hat - sigma0 Sigma g / sqrt(g^T Sigma g)
        (C* = C_hat for exact data). The zeros are located as the local
        minima of p on a grid in E in [e0, e_max], plus lambda = 0
        (E = infinity), refined by a root search on p', and kept if
        p < rel_tol * max|K|; the amplitudes then solve
        sum_k a_k b_t(lambda_k) = C*_t by least squares.

        Returns a dict with the energies E_k (inf for lambda = 0) and
        amplitudes a_k of rho(E) = sum_k a_k delta(E - E_k), the bound from
        the solver, the value sum_k a_k K(lambda_k), the relative residual of
        the fit to C* and the chi^2 distance of C* from the data.
        Polynomial kernels only.
        """
        if bound not in ("lower", "upper"):
            raise ValueError(f"bound must be 'lower' or 'upper', got '{bound}'")
        sign = 1 if bound == "lower" else -1
        signed = [sign * float(c) for c in kernel_coeffs]
        dual_value, _, _ = self._dual_bound_mpmath(signed)
        result = self.last_mpmath_result
        N, T = self.Nt, self.time_extent
        times = [self.t_min + i for i in range(N)]
        with mp.workdps(self.mp_digits):
            scale = [mpf(abs(float(v))) if v != 0 else mpf(1) for v in self.Ct]
            if self.covariance is None:
                g = [result["y"][i] / scale[i] for i in range(N)]
            else:
                Z = result["Y"][-1]
                g = [2 * Z[i][N] / scale[i] for i in range(N)]
            kpoly = [mpf(c) for c in signed]

            def b(t, x):
                return x**t if self.periodicity == "EXP" else x**t + x ** (T - t)

            def db(t, x):
                d = t * x ** (t - 1) if t > 0 else mpf(0)
                return d if self.periodicity == "EXP" else d + (T - t) * x ** (T - t - 1)

            def K(x):
                return mp.fsum(c * x**k for k, c in enumerate(kpoly))

            def p(x):
                return K(x) - mp.fsum(gt * b(t, x) for gt, t in zip(g, times))

            def dp(x):
                dk = mp.fsum(k * c * x ** (k - 1) for k, c in enumerate(kpoly) if k > 0)
                return dk - mp.fsum(gt * db(t, x) for gt, t in zip(g, times))

            lam_max = mpf(self.lam_max)
            grid = [lam_max * mp.exp(-mpf(e)) for e in np.linspace(0.0, e_max - self.e0, n_grid)] + [mpf(0)]
            values = [p(x) for x in grid]
            k_scale = max(abs(float(K(x))) for x in grid[::10])
            support = []
            for i, v in enumerate(values):
                left = values[i - 1] if i > 0 else mp.inf
                right = values[i + 1] if i + 1 < len(values) else mp.inf
                if not (v <= left and v <= right):
                    continue
                x = grid[i]
                if 0 < i < len(grid) - 1:  # interior minimum: refine on p' = 0
                    with contextlib.suppress(ValueError, ZeroDivisionError):
                        x = mp.findroot(dp, (grid[i + 1], grid[i - 1]), solver="anderson")
                if p(x) < rel_tol * k_scale:
                    support.append(x)

            C_star = [mpf(float(v)) for v in self.Ct]
            chi2 = mpf(0)
            if self.covariance is not None:
                Sigma = mp.matrix([[mpf(float(v)) for v in row] for row in self.covariance])
                Sg = Sigma * mp.matrix(g)
                norm = mp.sqrt(mp.fdot(g, Sg))
                C_star = [c - mpf(self.sigma0) * Sg[i] / norm for i, c in enumerate(C_star)]
                delta = mp.matrix([cs - mpf(float(c)) for cs, c in zip(C_star, self.Ct)])
                chi2 = mp.fdot(delta, mp.lu_solve(Sigma, delta))
            A = mp.matrix([[b(t, x) / scale[i] for x in support] for i, t in enumerate(times)])
            rhs = mp.matrix([c / s for c, s in zip(C_star, scale)])
            amplitudes, residual = mp.qr_solve(A, rhs)
            value = sign * mp.fsum(a * K(x) for a, x in zip(amplitudes, support))
            return {
                "bound": bound,
                "bound_value": sign * dual_value,
                "measure_value": float(value),
                "energies": [float(-mp.log(x)) if x > 0 else float("inf") for x in support],
                "amplitudes": [float(a) for a in amplitudes],
                "relative_residual": float(residual / mp.norm(rhs)),
                "chi2": float(chi2),
                "sigma0_squared": None if self.sigma0 is None else self.sigma0**2,
                "certificate_minima": [float(p(x)) for x in support],
            }

    def c0_upper_bound(self, solver=None):
        """
        Upper bound on C(0) = int rho, consistent with the data. Equal to
        Ct[0] for exact data; for noisy data, the largest C(0) within the
        consistency ellipsoid. Cached per solver.
        """
        if solver is None:
            solver = self.default_solver
        if solver not in self._c0_upper:
            self._c0_upper[solver] = -self._dual_bound([-1.0], solver=solver)[0]
        return self._c0_upper[solver]

    def smeared_bounds(self, kernel, degree, solver=None):
        """
        (lower, upper, diagnostics) bounds on K[rho] = int_e0^inf K(E) rho(E) dE
        for a general kernel K(E) (a numpy function of E which vanishes as
        E -> infinity, e.g. from `smearing_kernel`). K is replaced by its
        polynomial approximation of degree `degree` (`fit_kernel_polynomial`),
        whose bounds are then widened by epsilon * C0_upper, with C0_upper the
        upper bound on C(0) from `c0_upper_bound` rather than the measured
        Ct[0]: for noisy data, the latter is itself only an estimate.

        diagnostics holds the polynomial coefficients, epsilon, the widening
        and the bounds on the polynomial kernel (lower_poly, upper_poly).

        Requires t_min == 0: without C(0), a positive rho can hide any amount
        of weight at E -> infinity, so C(0) has no upper bound.
        """
        if self.t_min != 0:
            raise ValueError(
                "smeared_bounds requires t_min=0 (Ct must include C(0)), since "
                "C(0) has no upper bound otherwise. Without C(0), bounds() can "
                "still be used directly on the output of fit_kernel_polynomial."
            )
        poly_coeffs, epsilon = fit_kernel_polynomial(kernel, degree, e0=self.e0)
        lower_poly, upper_poly = self.bounds(poly_coeffs, solver=solver)
        inflation = epsilon * self.c0_upper_bound(solver=solver)
        diagnostics = {
            "poly_coeffs": poly_coeffs,
            "epsilon": epsilon,
            "inflation": inflation,
            "lower_poly": lower_poly,
            "upper_poly": upper_poly,
        }
        return lower_poly - inflation, upper_poly + inflation, diagnostics

    def gaussian_bounds(self, estar, sigma, degree, solver=None):
        """`smeared_bounds` for the FULLNORMGAUSS kernel centred at estar."""
        kernel = smearing_kernel(estar, sigma, "FULLNORMGAUSS", e0=self.e0)
        return self.smeared_bounds(kernel, degree, solver=solver)
