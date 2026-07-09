"""
Tests for src/lsdensities/correlator/correlator_utils.py.
"""

import numpy as np
import pytest

from lsdensities.correlator.correlator_utils import (
    InputsCorrelatorAnalysis,
    effective_mass,
    foldPeriodicCorrelator,
    symmetrisePeriodicCorrelator,
)
from lsdensities.utils.common import Inputs, Obs


def _make_cosh_correlator(T, nms, mass, symmetric=True):
    par = Inputs()
    par.time_extent = T
    par.periodicity = "COSH"
    par.num_samples = nms
    par.num_boot = nms
    par.tmax = T // 2

    corr = Obs(T=T, tmax=T - 1, nms=nms, sample_type="montecarlo")
    for n in range(nms):
        for t in range(T):
            if symmetric:
                corr.sample[n, t] = np.exp(-mass * t) + np.exp(-mass * (T - t))
            else:
                corr.sample[n, t] = np.exp(-mass * t)
    return par, corr


def test_fold_periodic_correlator_reproduces_symmetric_input():
    T, nms, mass = 16, 3, 0.35
    par, corr = _make_cosh_correlator(T, nms, mass, symmetric=True)

    folded = foldPeriodicCorrelator(corr, par)

    assert folded.sample.shape == (nms, T // 2 + 1)
    # The input is already symmetric under t <-> T-t, so folding must exactly
    # reproduce the first half (up to and including the midpoint).
    for n in range(nms):
        assert np.allclose(folded.sample[n], corr.sample[n, : T // 2 + 1])


def test_fold_periodic_correlator_averages_asymmetric_input():
    T, nms, mass = 16, 2, 0.3
    par, corr = _make_cosh_correlator(T, nms, mass, symmetric=False)

    folded = foldPeriodicCorrelator(corr, par)

    for n in range(nms):
        assert folded.sample[n, 0] == pytest.approx(corr.sample[n, 0])
        for t in range(1, T // 2 + 1):
            expected = (corr.sample[n, t] + corr.sample[n, T - t]) / 2
            assert folded.sample[n, t] == pytest.approx(expected)


def test_fold_periodic_correlator_requires_cosh():
    par, corr = _make_cosh_correlator(16, 2, 0.3)
    par.periodicity = "EXP"
    with pytest.raises(AssertionError):
        foldPeriodicCorrelator(corr, par)


@pytest.mark.parametrize("sample_type", ["montecarlo", "bootstrap", "jackknife"])
def test_fold_periodic_correlator_propagates_sample_type(sample_type):
    # Folding is just an average over existing samples: it must never change
    # (or silently assume) the sample_type, unlike a resampling step would.
    par, corr = _make_cosh_correlator(16, 3, 0.35)
    corr.sample_type = sample_type

    folded = foldPeriodicCorrelator(corr, par)

    assert folded.sample_type == sample_type


def test_symmetrise_periodic_correlator_reproduces_symmetric_input():
    T, nms, mass = 16, 3, 0.35
    par, corr = _make_cosh_correlator(T, nms, mass, symmetric=True)

    symm = symmetrisePeriodicCorrelator(corr, par)

    assert np.allclose(symm.sample, corr.sample)


def test_symmetrise_periodic_correlator_averages_asymmetric_input():
    T, nms, mass = 16, 2, 0.3
    par, corr = _make_cosh_correlator(T, nms, mass, symmetric=False)

    symm = symmetrisePeriodicCorrelator(corr, par)

    for n in range(nms):
        assert symm.sample[n, 0] == pytest.approx(corr.sample[n, 0])
        mid = T // 2
        assert symm.sample[n, mid] == pytest.approx(corr.sample[n, mid])
        for t in range(1, mid):
            expected = (corr.sample[n, t] + corr.sample[n, T - t]) / 2
            assert symm.sample[n, t] == pytest.approx(expected)
            assert symm.sample[n, T - t] == pytest.approx(expected)


def test_effective_mass_cosh_recovers_known_mass():
    T, nms, mass = 16, 3, 0.35
    par, corr = _make_cosh_correlator(T, nms, mass, symmetric=True)
    folded = foldPeriodicCorrelator(corr, par)

    emass = effective_mass(folded, par, type="COSH")

    assert emass.central == pytest.approx(mass, abs=1e-8)


def test_effective_mass_exp_recovers_known_mass():
    T, nms, mass = 16, 3, 0.4
    par = Inputs()
    par.time_extent = T
    par.periodicity = "EXP"
    par.num_boot = nms

    corr = Obs(T=T, tmax=T - 1, nms=nms, sample_type="montecarlo")
    for n in range(nms):
        for t in range(T):
            corr.sample[n, t] = np.exp(-mass * t)

    emass = effective_mass(corr, par, type="EXP")

    assert emass.central == pytest.approx(mass, abs=1e-8)


@pytest.mark.parametrize("sample_type", ["montecarlo", "bootstrap", "jackknife"])
def test_effective_mass_propagates_sample_type(sample_type):
    # effective_mass used to hardcode sample_type="bootstrap" regardless of
    # corr's actual sample_type, silently mis-scaling the mass's error
    # whenever corr wasn't a bootstrap correlator.
    T, nms, mass_ = 16, 3, 0.35
    par, corr = _make_cosh_correlator(T, nms, mass_, symmetric=True)
    corr.sample_type = sample_type

    emass = effective_mass(corr, par, type="COSH")

    assert emass.sample_type == sample_type
    assert emass.nms == corr.nms


def test_effective_mass_invalid_type_raises():
    T, nms, mass = 16, 2, 0.3
    par, corr = _make_cosh_correlator(T, nms, mass, symmetric=True)
    with pytest.raises(ValueError):
        effective_mass(corr, par, type="bogus")


def test_inputs_correlator_analysis_type_validation():
    InputsCorrelatorAnalysis(time_extent=16, num_boot=100, num_samples=100)
    with pytest.raises(TypeError):
        InputsCorrelatorAnalysis(time_extent="16")
    with pytest.raises(TypeError):
        InputsCorrelatorAnalysis(num_boot="100")
    with pytest.raises(TypeError):
        InputsCorrelatorAnalysis(num_samples="100")
