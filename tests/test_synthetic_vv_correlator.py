"""Tests of the synthetic vector-vector-like correlator used by the examples."""

import numpy as np

from lsdensities.correlator.synthetic_vv_correlator import generate_correlator, generate_states
from lsdensities.utils.common import Inputs, init_precision


def _make_par(emax=1.2):
    init_precision(30)
    par = Inputs()
    par.time_extent = 12
    par.periodicity = "EXP"
    par.sigma = 0.4
    par.emin = 0.3
    par.emax = emax
    par.Ne = 2
    par.num_boot = 100
    par.assign_values()
    return par


def test_mean_is_close_to_the_exact_correlator():
    par = _make_par()
    corr, rho_true = generate_correlator(par, [0.3, 0.6], n_samples=par.num_boot)
    peaks, weights = generate_states(1, continuum_emax=3 * par.emax)
    exact = np.array([np.sum(weights * np.exp(-peaks * t)) for t in range(par.time_extent)])
    assert corr.sample.shape == (par.num_boot, par.time_extent)
    assert np.allclose(corr.central, exact, rtol=0.05)
    assert len(rho_true) == 2 and np.all(rho_true > 0)


def test_spectrum_depends_on_continuum_emax_only():
    a, rho_a = generate_correlator(_make_par(emax=1.2), [0.6], n_samples=50)
    b, rho_b = generate_correlator(_make_par(emax=2.0), [0.6], n_samples=50, continuum_emax=3 * 1.2)
    assert np.array_equal(a.sample, b.sample)
    assert np.array_equal(rho_a, rho_b)
