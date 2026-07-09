"""
Synthetic vector-vector-like lattice correlator, with a realistic noise model,
shared by run_hlt_wnoise.py and run_gp_wnoise.py so that both methods are
compared against exactly the same mock dataset.

The mock spectrum follows the same recipe as arXiv:2605.14652 (Eqs. 25-26):
a handful of delta-function states between roughly 2*m_pi and a few GeV. Here
it is shaped like a vector-current (rho-meson / "vector-vector") correlator:
one dominant low-lying resonance around the rho mass, plus a multi-hadron
continuum above the two-pion threshold.

The covariance is the qualitatively realistic one for hadronic
correlators: the relative statistical error grows with Euclidean time and nearby time slices are
correlated.

"""

import numpy as np

from lsdensities.utils.math_utils import gauss_fp
from lsdensities.utils.common import Obs

PION_MASS = 0.140  # GeV
RHO_MASS = 0.775  # GeV, dominant vector-meson resonance


def generate_states(seed, n_continuum=24, continuum_emax=3.6):
    """
    One resonance near the rho mass plus a continuum of states above 2*m_pi,
    mimicking the spectral content of a vector-current correlator.
    """
    rng = np.random.default_rng(seed)
    resonance = np.array([RHO_MASS])
    resonance_weight = np.array([rng.uniform(0.35, 0.45)])
    continuum = np.sort(rng.uniform(2 * PION_MASS, continuum_emax, n_continuum))
    continuum_weight = rng.uniform(0.005, 0.02, n_continuum)
    peaks = np.concatenate([resonance, continuum])
    weights = np.concatenate([resonance_weight, continuum_weight])
    return peaks, weights


def true_spectral_density(espace, sigma, peaks, weights):
    """Exact smeared spectral density (Gaussian-smeared sum of delta functions)."""
    rho = np.zeros(len(espace))
    for e_i, e in enumerate(espace):
        rho[e_i] = np.sum(
            [gauss_fp(peak, e, sigma, norm="full") * w for peak, w in zip(peaks, weights)]
        )
    return rho


def build_covariance(exact_correlator, base_rel_error, target_rel_error, corr_length):
    """
    Diagonal relative error grows exponentially from base_rel_error (at t=0)
    to target_rel_error (at the last time slice) -- the standard lattice-QCD
    signal-to-noise problem for hadronic correlators. Nearby time slices are
    correlated with a correlation length of corr_length time slices, as is
    typical of real bootstrap/jackknife covariances.
    """
    T = len(exact_correlator)
    growth_rate = np.log(target_rel_error / base_rel_error) / (T - 1)
    rel_error = base_rel_error * np.exp(growth_rate * np.arange(T))
    sigma_t = np.abs(exact_correlator) * rel_error

    idx = np.arange(T)
    correlation = np.exp(-np.abs(idx[:, None] - idx[None, :]) / corr_length)
    cov = np.outer(sigma_t, sigma_t) * correlation
    return 0.5 * (cov + cov.T)


def generate_correlator(
    par,
    espace,
    seed=1,
    n_continuum=24,
    base_rel_error=0.002,
    target_rel_error=0.15,
    corr_length=3.0,
    n_samples=800,
    with_cholesky=False,
):
    """
    Builds a synthetic vector-vector-like correlator (EXP periodicity, i.e.
    already folded) together with the exact smeared spectral density it was
    generated from, for use as the reconstruction target.

    Returns (correlator, rho_true) where correlator is an lsdensities.Obs
    instance ready to be passed to HLTWithBackusGilbert/GaussianProcessWrapper
    (mp sample, covariance and central value already filled in). Pass
    with_cholesky=True for HLTWithSVD, which additionally needs
    the Cholesky factor of the covariance (mpcholesky).
    """
    peaks, weights = generate_states(
        seed, n_continuum=n_continuum, continuum_emax=3 * par.emax
    )

    T = par.time_extent
    exact_correlator = np.array(
        [np.sum(weights * np.exp(-peaks * t)) for t in range(T)]
    )
    cov = build_covariance(exact_correlator, base_rel_error, target_rel_error, corr_length)

    rng = np.random.default_rng(seed + 1)
    sample = rng.multivariate_normal(exact_correlator, cov, size=n_samples)

    corr = Obs(T=T, tmax=par.tmax, nms=n_samples, sample_type="bootstrap")
    corr.sample = sample
    corr.evaluate()
    corr.evaluate_covmatrix()
    if with_cholesky:
        corr.evaluate_cholesky()
    corr.fill_mp_sample(w_cholesky=with_cholesky)

    rho_true = true_spectral_density(espace, par.sigma, peaks, weights)

    return corr, rho_true
