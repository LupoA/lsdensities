"""
Smeared spectral density from the causal bootstrap alone (arXiv:2605.20509):
at every energy, the result is the interval [lower, upper] containing
rho_sigma(E) = int K(E, E') rho(E') dE' for every rho(E') >= 0 which is
statistically consistent with the measured correlator (central value and
covariance), at a given confidence level. No regulator is involved.

Smearing kernels that are not polynomials in exp(-E) are replaced by
polynomial approximations, whose error is added to the bounds
(arXiv:2605.20509, end of Sec. IV A; see causal_bootstrap.smeared_bounds).
Every degree of the approximation gives valid bounds, which hold
simultaneously whenever the true correlator lies in the consistency
ellipsoid, so the result at each energy is the intersection of the
intervals obtained at all the requested degrees.

The same bounds can be overlaid on the stability analysis
(HLTWithBackusGilbert, GaussianProcessWrapper) and on the eigen-space
analysis (HLTWithSVD) of arXiv:2605.14652, via their
compute_causal_bootstrap_bounds() method.
"""

import logging

import numpy as np

from .. import io_utils
from ..causal_bootstrap import CausalBootstrap, smearing_kernel
from ..utils.common import Inputs, Obs, log

__all__ = ["CausalBootstrapAnalysis"]


class CausalBootstrapAnalysis:
    """
    par : Inputs. par.tmax+1 correlator values C(0), ..., C(tmax) are used:
        C(0) is needed for the bounds to be finite (see
        CausalBootstrap.smeared_bounds), unlike in the HLT analysis.
    correlator : Obs, with central value and covariance (Obs.central,
        Obs.cov) evaluated.
    energies : energies at which rho_sigma(E) is bounded.
    degrees : degree (or list of degrees) of the polynomial approximation
        of the smearing kernel.
    confidence : confidence level of the statistical-consistency ellipsoid
        (CausalBootstrap.sigma0_from_confidence).
    mp_digits : working precision of the high-precision SDP solver.
    solver : "MPMATH" (default, see CausalBootstrap.default_solver) or the
        name of a double-precision cvxpy solver.
    """

    def __init__(
        self,
        par: Inputs,
        correlator: Obs,
        energies,
        degrees=16,
        confidence=0.9,
        mp_digits=100,
        solver=None,
    ):
        self.par = par
        self.correlator = correlator
        self.espace = np.asarray(energies, dtype=float)
        self.par.Ne = len(self.espace)
        self.degrees = [int(degrees)] if np.isscalar(degrees) else [int(d) for d in degrees]
        self.confidence = float(confidence)
        self.mp_digits = int(mp_digits)
        self.solver = solver
        smearing_kernel(self.espace[0], par.sigma, par.kerneltype)  # fails early if unsupported

        self.Nt = par.tmax + 1
        self.Ct = np.array([float(c) for c in correlator.central[: self.Nt]])
        self.covariance = np.array(
            [[float(correlator.cov[i][j]) for j in range(self.Nt)] for i in range(self.Nt)]
        )
        self.sigma0 = CausalBootstrap.sigma0_from_confidence(self.confidence, self.Nt)
        self.bootstrap()  # fails early on unsupported periodicity or e0

        #   Results, one entry per energy (None where no bound exists)
        self.lower = [None] * self.par.Ne
        self.upper = [None] * self.par.Ne
        self.scan = [None] * self.par.Ne
        self.c0_upper = None

    def bootstrap(self):
        return CausalBootstrap(
            self.Ct,
            covariance=self.covariance,
            sigma0=self.sigma0,
            t_min=0,
            e0=float(self.par.e0),
            periodicity=self.par.periodicity,
            time_extent=self.par.time_extent,
            mp_digits=self.mp_digits,
        )

    def _bounds_at(self, bootstrap, energy):
        """Bounds at one energy for every degree; returns the scan over degrees."""
        kernel = smearing_kernel(
            energy, float(self.par.sigma), self.par.kerneltype, e0=float(self.par.e0)
        )
        scan = {key: [] for key in ("degree", "lower", "upper", "epsilon", "lower_poly", "upper_poly")}
        for degree in self.degrees:
            try:
                lower, upper, diagnostics = bootstrap.smeared_bounds(kernel, degree, solver=self.solver)
            except RuntimeError as error:
                log(
                    f"Causal bootstrap: no bound at E={energy:.4g}, degree {degree} ({error})",
                    level=logging.WARNING,
                )
                lower = upper = None
                diagnostics = {"epsilon": None, "lower_poly": None, "upper_poly": None}
            scan["degree"].append(degree)
            scan["lower"].append(lower)
            scan["upper"].append(upper)
            for key in ("epsilon", "lower_poly", "upper_poly"):
                scan[key].append(diagnostics[key])
            if lower is not None:
                log(
                    f"Causal bootstrap ::: E = {energy:.4g}, degree {degree}: "
                    f"[{lower:.6g}, {upper:.6g}], epsilon = {diagnostics['epsilon']:.2e}"
                )
        return scan

    def run(self):
        bootstrap = self.bootstrap()
        try:
            self.c0_upper = bootstrap.c0_upper_bound(solver=self.solver)
        except RuntimeError as error:
            # The C(0) bound enters every energy: if it does not exist, the
            # data are outside their own consistency ellipsoid.
            log(
                f"Causal bootstrap: the data are not consistent with any positive "
                f"spectral density at {self.confidence:.0%} confidence ({error})",
                level=logging.WARNING,
            )
            return 0
        for e_i, energy in enumerate(self.espace):
            scan = self._bounds_at(bootstrap, energy)
            self.scan[e_i] = scan
            lowers = [v for v in scan["lower"] if v is not None]
            uppers = [v for v in scan["upper"] if v is not None]
            if lowers:
                self.lower[e_i] = max(lowers)
                self.upper[e_i] = min(uppers)
        return 0

    def result(self, e_i):
        """Bounds at the e_i-th energy, as a dict (None if there are none)."""
        if self.lower[e_i] is None:
            return None
        lower, upper = float(self.lower[e_i]), float(self.upper[e_i])
        return {
            "lower": lower,
            "upper": upper,
            "rho": (lower + upper) / 2,
            "err": (upper - lower) / 2,
        }

    def settings(self):
        return {
            "confidence": self.confidence,
            "sigma0": self.sigma0,
            "degrees": self.degrees,
            "mp_digits": self.mp_digits,
            "solver": self.solver or CausalBootstrap.default_solver,
            "c0_upper": None if self.c0_upper is None else float(self.c0_upper),
        }

    def save(self, path=None):
        """
        Writes the bounds at every energy, together with their dependence on
        the degree of the kernel approximation, to a single JSON file (see
        io_utils.py for the shared schema). Use examples/plot_output.py to
        plot from it.
        """
        metadata = io_utils.base_metadata(self.par, "CausalBootstrap")
        metadata["algorithm"] = self.settings()
        energies = []
        for e_i, energy in enumerate(self.espace):
            scan = self.scan[e_i] or {}
            energies.append(
                {
                    "energy": float(energy),
                    "scan": {
                        key: [None if v is None else (int(v) if key == "degree" else float(v)) for v in values]
                        for key, values in scan.items()
                    },
                    "result": {"CausalBootstrap": self.result(e_i)},
                }
            )
        path = path or io_utils.default_output_path(self.par, metadata["method"])
        return io_utils.write_json(path, metadata, energies)


def compute_causal_bootstrap_bounds(wrapper, **kwargs):
    """
    Runs a CausalBootstrapAnalysis on the data and energies of `wrapper`
    (HLTWithBackusGilbert, GaussianProcessWrapper or HLTWithSVD), for the
    same smeared spectral density, and stores it as wrapper.causal_bootstrap
    so that save() writes the bounds next to the wrapper's own results.
    kwargs are those of CausalBootstrapAnalysis.
    """
    analysis = CausalBootstrapAnalysis(wrapper.par, wrapper.correlator, wrapper.espace, **kwargs)
    analysis.run()
    wrapper.causal_bootstrap = analysis
    return analysis


def add_causal_bootstrap_output(wrapper, metadata, energies):
    """
    Adds the bounds stored by compute_causal_bootstrap_bounds, if any, to
    the metadata and per-energy entries of the wrapper's JSON output:
    metadata["causal_bootstrap"] and entry["causal_bootstrap"].
    """
    analysis = getattr(wrapper, "causal_bootstrap", None)
    if analysis is None:
        return
    metadata["causal_bootstrap"] = analysis.settings()
    for e_i, entry in enumerate(energies):
        entry["causal_bootstrap"] = analysis.result(e_i)
