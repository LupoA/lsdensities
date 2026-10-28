"""
Tests of CausalBootstrapAnalysis (bounds on rho_sigma(E) from the causal
bootstrap alone, arXiv:2605.20509) and of the bounds stored next to the
stability analysis and the eigen-space analysis of arXiv:2605.14652.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from mpmath import mpf

from lsdensities.inverse_problem_solvers.causal_bootstrap_bounds import CausalBootstrapAnalysis
from lsdensities.inverse_problem_solvers.hlt_eigenspace import HETpar, HLTWithSVD
from lsdensities.inverse_problem_solvers.hlt_stability import AlgorithmParameters, HLTWithBackusGilbert
from lsdensities.io_utils import load_json
from lsdensities.utils.common import Inputs, Obs, init_precision
from lsdensities.utils.math_utils import gauss_fp

PEAKS = np.array([0.5, 0.9, 1.6])
WEIGHTS = np.array([0.6, 1.0, 0.8])
ENERGIES = [0.6, 1.0]
CB_SETTINGS = dict(degrees=[8, 10], confidence=0.9, mp_digits=40)


def _make_par(tmp_path, time_extent=10, Na=1):
    init_precision(40)
    par = Inputs()
    par.time_extent = time_extent
    par.periodicity = "EXP"
    par.kerneltype = "FULLNORMGAUSS"
    par.sigma = 0.4
    par.emin = ENERGIES[0]
    par.emax = ENERGIES[-1]
    par.Ne = len(ENERGIES)
    par.Na = Na
    par.e0 = 0
    par.prec = 40
    par.num_boot = 200
    par.logpath = str(tmp_path)
    par.assign_values()
    return par


def _make_correlator(par, rel_error=1e-3, seed=0):
    t = np.arange(par.time_extent)
    exact = (WEIGHTS * np.exp(-np.outer(t, PEAKS))).sum(axis=1)
    cov = np.diag((rel_error * exact) ** 2)
    rng = np.random.default_rng(seed)
    measured = rng.multivariate_normal(exact, cov)
    corr = Obs(T=par.time_extent, tmax=par.tmax, nms=par.num_boot, sample_type="bootstrap")
    corr.sample = rng.multivariate_normal(measured, cov, size=par.num_boot)
    corr.evaluate()
    corr.evaluate_covmatrix()
    corr.evaluate_cholesky()
    corr.fill_mp_sample(w_cholesky=True)
    return corr


def _rho_true(energy, sigma):
    return float(np.sum(WEIGHTS * gauss_fp(PEAKS, energy, sigma, norm="full")))


def test_bounds_bracket_the_exact_smeared_density(tmp_path):
    par = _make_par(tmp_path)
    analysis = CausalBootstrapAnalysis(par, _make_correlator(par), ENERGIES, **CB_SETTINGS)
    analysis.run()
    for e_i, energy in enumerate(ENERGIES):
        assert analysis.lower[e_i] <= _rho_true(energy, par.sigma) <= analysis.upper[e_i]


def test_result_is_the_intersection_over_degrees(tmp_path):
    par = _make_par(tmp_path)
    analysis = CausalBootstrapAnalysis(par, _make_correlator(par), ENERGIES, **CB_SETTINGS)
    analysis.run()
    for e_i in range(len(ENERGIES)):
        scan = analysis.scan[e_i]
        assert scan["degree"] == CB_SETTINGS["degrees"]
        assert analysis.lower[e_i] == max(scan["lower"])
        assert analysis.upper[e_i] == min(scan["upper"])


def test_save_writes_bounds_and_scan(tmp_path):
    par = _make_par(tmp_path)
    analysis = CausalBootstrapAnalysis(par, _make_correlator(par), ENERGIES, **CB_SETTINGS)
    analysis.run()
    data = load_json(analysis.save(str(tmp_path / "cb.json")))
    assert data["metadata"]["method"] == "CausalBootstrap"
    assert data["metadata"]["algorithm"]["degrees"] == CB_SETTINGS["degrees"]
    for e_i, entry in enumerate(data["energies"]):
        result = entry["result"]["CausalBootstrap"]
        assert result["lower"] == pytest.approx(analysis.lower[e_i])
        assert result["upper"] == pytest.approx(analysis.upper[e_i])
        assert result["rho"] == pytest.approx((result["lower"] + result["upper"]) / 2)
        assert entry["scan"]["degree"] == CB_SETTINGS["degrees"]


def test_inconsistent_data_give_no_bounds(tmp_path):
    # A correlator growing with t is not the Laplace transform of any
    # positive spectral density.
    par = _make_par(tmp_path)
    corr = _make_correlator(par)
    corr.central = corr.central[::-1].copy()
    analysis = CausalBootstrapAnalysis(par, corr, ENERGIES, **CB_SETTINGS)
    analysis.run()
    assert analysis.lower == [None] * len(ENERGIES)
    assert analysis.upper == [None] * len(ENERGIES)


def test_unsupported_kernel_is_rejected(tmp_path):
    par = _make_par(tmp_path)
    par.kerneltype = "LINEAR"
    with pytest.raises(ValueError, match="LINEAR"):
        CausalBootstrapAnalysis(par, _make_correlator(par), ENERGIES)


def test_unsupported_periodicity_is_rejected(tmp_path):
    par = _make_par(tmp_path)
    par.periodicity = "SINH"
    with pytest.raises(ValueError, match="periodicity"):
        CausalBootstrapAnalysis(par, _make_correlator(par), ENERGIES)


def test_stability_analysis_output_carries_the_bounds(tmp_path):
    par = _make_par(tmp_path)
    corr = _make_correlator(par)
    hlt = HLTWithBackusGilbert(
        par=par,
        algorithmPar=AlgorithmParameters(lambdaMax=10, lambdaMin=1e-2, lambdaStep=2, lambdaScanCap=2),
        B=corr.mpcov,
        bnorm=mpf(str(corr.central[1] ** 2)),
        correlator=corr,
        energies=ENERGIES,
    )
    hlt.prepareHLT()
    hlt.run()
    hlt.compute_causal_bootstrap_bounds(**CB_SETTINGS)
    data = load_json(hlt.save(str(tmp_path / "hlt.json")))
    assert data["metadata"]["causal_bootstrap"]["confidence"] == CB_SETTINGS["confidence"]
    for e_i, entry in enumerate(data["energies"]):
        bounds = entry["causal_bootstrap"]
        assert bounds["lower"] == pytest.approx(hlt.causal_bootstrap.lower[e_i])
        assert bounds["lower"] <= _rho_true(entry["energy"], par.sigma) <= bounds["upper"]
        assert "HLT" in entry["result"]


def test_eigenspace_analysis_output_carries_the_bounds(tmp_path):
    par = _make_par(tmp_path)
    corr = _make_correlator(par)
    ea = HLTWithSVD(
        par=par,
        algorithmPar=HETpar(),
        correlator=corr,
        energies=ENERGIES,
        l_reg=mpf("1e-9"),
    )
    ea.prepare()
    ea.run()
    ea.compute_causal_bootstrap_bounds(**CB_SETTINGS)
    data = load_json(ea.save(str(tmp_path / "ea.json")))
    assert "causal_bootstrap" in data["metadata"]
    for e_i, entry in enumerate(data["energies"]):
        assert entry["causal_bootstrap"]["upper"] == pytest.approx(ea.causal_bootstrap.upper[e_i])
        # the channel results are left as they are
        assert sorted(entry["result"].keys()) == ["A"]


def test_output_without_bounds_is_unchanged(tmp_path):
    from lsdensities.inverse_problem_solvers.causal_bootstrap_bounds import add_causal_bootstrap_output

    metadata, energies = {}, [{"energy": 1.0}]
    add_causal_bootstrap_output(SimpleNamespace(causal_bootstrap=None), metadata, energies)
    assert metadata == {}
    assert energies == [{"energy": 1.0}]
