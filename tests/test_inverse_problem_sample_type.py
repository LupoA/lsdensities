"""
Regression test: HLTWithBackusGilbert must propagate the correlator's own
Obs.sample_type into the frequentist ("Bootstrap") error on the smeared
spectral density, exactly as it already does for the Bayesian error (which
goes through Obs.evaluate_covmatrix -> B). Without this, drho_estar_Bootstrap
would silently be computed as if sample_type="bootstrap" regardless of what
was passed in (see lsdensities.transform.get_ssd_averaged_scalar).
"""

import numpy as np
from mpmath import mp, mpf

from lsdensities.inverse_problem_solvers.hlt_stability import AlgorithmParameters, HLTWithBackusGilbert
from lsdensities.utils.common import Inputs, Obs, _variance_scale_factor, init_precision


def _make_params(nms):
    par = Inputs()
    par.time_extent = 6
    par.periodicity = "EXP"
    par.kerneltype = "FULLNORMGAUSS"
    par.sigma = 0.3
    par.e0 = 0
    par.num_boot = nms  # get_ssd_averaged_scalar loops range(params.num_boot)
    par.assign_values()
    return par


def _make_wrapper(par, sample_type, raw_sample, nms, fixed_B):
    """
    `fixed_B` is passed in (rather than each correlator's own mpcov) so that
    every wrapper here inverts the exact same matrix and gets the exact same
    coefficients _g_t_estar. That isolates the one thing sample_type should
    still affect: the scale factor applied on top of the per-replica spread of
    rho in get_ssd_averaged_scalar. (correlator.mpcov's own sample_type scaling
    is already covered by test_sample_type.py.)
    """
    corr = Obs(T=par.time_extent, tmax=par.tmax, nms=nms, sample_type=sample_type)
    corr.sample = raw_sample
    corr.evaluate()
    corr.evaluate_covmatrix()
    corr.fill_mp_sample()

    algorithmPar = AlgorithmParameters(lambdaMax=1.0)
    wrapper = HLTWithBackusGilbert(
        par=par,
        algorithmPar=algorithmPar,
        B=fixed_B,
        bnorm=mpf(1),
        correlator=corr,
        energies=[1.0],
    )
    wrapper.prepareHLT()
    return wrapper


def test_lambda_to_rho_bootstrap_error_scales_with_correlator_sample_type():
    init_precision(30)
    nms = 12
    par = _make_params(nms)

    rng = np.random.default_rng(0)
    raw_sample = rng.normal(loc=1.0, scale=0.1, size=(nms, par.time_extent))

    estar = 1.0
    lambda_ = mpf("0.5")

    # Fixed regulator matrix shared by every wrapper below, so that the
    # inverted matrix (and hence the coefficients _g_t_estar) is identical
    # regardless of sample_type -- see _make_wrapper's docstring.
    reference_corr = Obs(T=par.time_extent, tmax=par.tmax, nms=nms, sample_type="bootstrap")
    reference_corr.sample = raw_sample
    reference_corr.evaluate()
    reference_corr.evaluate_covmatrix()
    reference_corr.fill_mp_sample()
    fixed_B = reference_corr.mpcov

    errors = {}
    for sample_type in ("montecarlo", "bootstrap", "jackknife"):
        wrapper = _make_wrapper(par, sample_type, raw_sample, nms, fixed_B)
        _, _, drho_boot, _, _, _ = wrapper.lambdaToRho(
            lambda_, estar, wrapper.channelA.alpha_mp
        )
        errors[sample_type] = drho_boot

    scale_bootstrap = _variance_scale_factor("bootstrap", nms)
    for sample_type in ("montecarlo", "jackknife"):
        scale = _variance_scale_factor(sample_type, nms)
        expected_ratio = mp.sqrt(mpf(scale) / mpf(scale_bootstrap))
        actual_ratio = errors[sample_type] / errors["bootstrap"]
        # Loose-ish tolerance: the ratio itself is exact, but matrix inversion
        # (invert_matrix_ge) sheds a few digits relative to the mp.dps=30 setting.
        assert abs(actual_ratio - expected_ratio) < mpf("1e-15")
