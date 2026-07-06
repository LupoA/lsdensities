"""
Tests for Obs's sample_type ("montecarlo"/"bootstrap"/"jackknife"), the
matching mp-precision estimators in rhoStat.py, and JackknifeLoop.

Formulas (n = number of samples, S = sum((x_i - mean)**2), ddof=1 std sigma =
sqrt(S/(n-1))):
    montecarlo : err = sigma / sqrt(n)          (standard error of the mean)
    bootstrap  : err = sigma
    jackknife  : err = sigma * (n - 1) / sqrt(n)
"""

import random

import numpy as np
import pytest
from mpmath import mp

from lsdensities.utils.rhoParallelUtils import JackknifeLoop, ParallelBootstrapLoop
from lsdensities.utils.rhoStat import averageScalar_mp, averageVector_mp
from lsdensities.utils.rhoUtils import Inputs, Obs, _variance_scale_factor


def _textbook_jackknife_se(replicates):
    """(n-1)/n * sum((theta_i - theta_bar)**2), sqrt'd."""
    n = len(replicates)
    theta_bar = np.mean(replicates)
    S = np.sum((replicates - theta_bar) ** 2)
    return np.sqrt((n - 1) / n * S)


@pytest.fixture
def raw_sample():
    rng = np.random.default_rng(42)
    return rng.normal(loc=5.0, scale=2.0, size=41)


def test_variance_scale_factor_invalid_raises():
    with pytest.raises(ValueError):
        _variance_scale_factor("bogus", 10)


@pytest.mark.parametrize("sample_type", ["montecarlo", "bootstrap", "jackknife"])
def test_obs_err_matches_manual_formula(raw_sample, sample_type):
    n = len(raw_sample)
    o = Obs(T=1, tmax=1, nms=n, sample_type=sample_type)
    o.sample[:, 0] = raw_sample
    o.evaluate()

    sigma = np.std(raw_sample, ddof=1)
    if sample_type == "montecarlo":
        expected_err = sigma / np.sqrt(n)
    elif sample_type == "bootstrap":
        expected_err = sigma
    else:
        expected_err = sigma * (n - 1) / np.sqrt(n)

    assert o.err[0] == pytest.approx(expected_err)


def test_obs_jackknife_err_matches_textbook_formula(raw_sample):
    # Independent of _variance_scale_factor: compares against a from-scratch
    # implementation of the standard jackknife variance formula.
    n = len(raw_sample)
    o = Obs(T=1, tmax=1, nms=n, sample_type="jackknife")
    o.sample[:, 0] = raw_sample
    o.evaluate()
    assert o.err[0] == pytest.approx(_textbook_jackknife_se(raw_sample))


@pytest.mark.parametrize("sample_type", ["montecarlo", "bootstrap", "jackknife"])
def test_obs_cov_diagonal_matches_err_squared(sample_type):
    rng = np.random.default_rng(1)
    n, T = 30, 4
    o = Obs(T=T, tmax=T, nms=n, sample_type=sample_type)
    o.sample = rng.normal(size=(n, T))
    o.evaluate()
    cov = o.evaluate_covmatrix()
    assert np.allclose(np.diagonal(cov), o.err**2)


def test_obs_invalid_sample_type_raises():
    with pytest.raises(ValueError):
        Obs(T=2, tmax=2, nms=5, sample_type="bogus")


def test_obs_corrmat_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(2)
    n, T = 20, 5
    o = Obs(T=T, tmax=T, nms=n, sample_type="bootstrap")
    o.sample = rng.normal(size=(n, T))
    o.evaluate()
    o.evaluate_covmatrix()
    o.corrmat_from_covmat()
    assert np.allclose(np.diagonal(o.corrmat), 1.0)
    assert np.allclose(o.corrmat, o.corrmat.T)


@pytest.mark.parametrize("sample_type", ["montecarlo", "bootstrap", "jackknife"])
def test_float_and_mp_paths_agree(sample_type):
    # Obs.evaluate() (numpy) and averageVector_mp/averageScalar_mp (mpmath) must
    # report the same error for the same data and sample_type.
    mp.dps = 30
    rng = np.random.default_rng(3)
    n = 23
    raw = rng.normal(3.0, 1.5, size=n)

    o = Obs(T=1, tmax=1, nms=n, sample_type=sample_type)
    o.sample[:, 0] = raw
    o.evaluate()

    mp_row = mp.matrix([raw.tolist()])
    vec_avg, vec_err = averageVector_mp(mp_row, sample_type=sample_type)[0, :]
    scalar_avg, scalar_err = averageScalar_mp(mp.matrix(raw.tolist()), sample_type=sample_type)

    assert float(vec_err) == pytest.approx(o.err[0])
    assert float(scalar_err) == pytest.approx(o.err[0])
    assert float(vec_avg) == pytest.approx(o.central[0])
    assert float(scalar_avg) == pytest.approx(o.central[0])


def test_jackknife_loop_leave_one_out_arithmetic():
    par = Inputs()
    par.num_samples = 6
    par.time_extent = 4
    raw = np.arange(24, dtype=float).reshape(6, 4)

    out = JackknifeLoop(par, raw, is_folded=False).run()

    assert out.shape == (6, 4)
    for i in range(6):
        expected = np.delete(raw, i, axis=0).mean(axis=0)
        assert np.allclose(out[i], expected)


def test_jackknife_loop_folded_only_fills_half_the_columns():
    par = Inputs()
    par.num_samples = 5
    par.time_extent = 8
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(5, 8))

    out = JackknifeLoop(par, raw, is_folded=True).run()

    vlen = int(par.time_extent / 2) + 1
    assert out.shape == (5, par.time_extent)
    assert np.all(out[:, vlen:] == 0)
    for i in range(5):
        expected = np.delete(raw[:, :vlen], i, axis=0).mean(axis=0)
        assert np.allclose(out[i, :vlen], expected)


def test_jackknife_reproduces_standard_error_of_the_mean():
    """
    delete-1 jackknife of the sample mean reproduces the
    plain standard error of the mean almost exactly (unlike, say, jackknife of
    a ratio estimator, where the two would differ).
    """
    par = Inputs()
    par.num_samples = 200
    par.time_extent = 1

    rng = np.random.default_rng(5)
    raw = rng.normal(loc=10.0, scale=3.0, size=(par.num_samples, 1))

    replicates = JackknifeLoop(par, raw, is_folded=False).run()

    o = Obs(T=1, tmax=1, nms=par.num_samples, sample_type="jackknife")
    o.sample = replicates
    o.evaluate()

    naive_se_of_raw_mean = np.std(raw[:, 0], ddof=1) / np.sqrt(par.num_samples)
    assert o.err[0] == pytest.approx(naive_se_of_raw_mean, rel=1e-6)


def test_montecarlo_bootstrap_jackknife_agree_at_large_n():
    """
    cross-check, through the actual pipeline classes (not just the
    formulas in isolation): draw one raw i.i.d. sample, resample it with both
    ParallelBootstrapLoop and JackknifeLoop, and build an Obs of each of the
    three sample_types from it. At a large enough sample size, all three must
    agree with each other and with the analytic standard error of the mean
    (sigma_true / sqrt(n)) for a population of known variance.
    """
    n, num_boot = 2000, 500
    true_sigma = 2.0
    rng = np.random.default_rng(7)
    raw = rng.normal(loc=5.0, scale=true_sigma, size=(n, 1))
    analytic_se = true_sigma / np.sqrt(n)

    par = Inputs()
    par.num_samples = n
    par.time_extent = 1
    par.num_boot = num_boot

    o_mc = Obs(T=1, tmax=1, nms=n, sample_type="montecarlo")
    o_mc.sample = raw.copy()
    o_mc.evaluate()

    random.seed(123)  # ParallelBootstrapLoop reads Python's global `random` state
    boot_replicates = ParallelBootstrapLoop(par, raw, is_folded=False).run()
    o_boot = Obs(T=1, tmax=1, nms=num_boot, sample_type="bootstrap")
    o_boot.sample = boot_replicates.copy()
    o_boot.evaluate()

    jack_replicates = JackknifeLoop(par, raw, is_folded=False).run()
    o_jack = Obs(T=1, tmax=1, nms=n, sample_type="jackknife")
    o_jack.sample = jack_replicates.copy()
    o_jack.evaluate()

    # Jackknife of the mean reproduces the raw standard error almost exactly
    assert o_jack.err[0] == pytest.approx(o_mc.err[0], rel=1e-6)

    # Bootstrap carries extra Monte Carlo noise from resampling with only
    # num_boot replicates, so it gets a looser (but still meaningful) tolerance.
    assert o_mc.err[0] == pytest.approx(analytic_se, rel=0.15)
    assert o_jack.err[0] == pytest.approx(analytic_se, rel=0.15)
    assert o_boot.err[0] == pytest.approx(analytic_se, rel=0.15)
