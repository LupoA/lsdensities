"""
Bounds on the smeared spectral density from the causal bootstrap alone
(arXiv:2605.20509), on the same synthetic vector-vector-like correlator used
by run_hlt_wnoise.py and run_ea_wnoise.py.

At every energy, the result is the interval containing rho_sigma(E) for every
positive spectral density statistically consistent with the data, at the
confidence level --confidence. The smearing kernel is approximated by
polynomials in exp(-E) of the degrees given by --degrees, and the result is
the intersection of the bounds obtained at each degree.

Saves the bounds (and their dependence on the degree) to a JSON file under
--outdir (see src/lsdensities/io_utils.py for the schema) and a plot comparing
them with the known exact smeared spectral density. Use
examples/plot_output.py (modes causal-bootstrap and spectrum) to plot from the
saved JSON.
"""

import argparse
import os

import numpy as np

from lsdensities.inverse_problem_solvers.causal_bootstrap_bounds import CausalBootstrapAnalysis
from lsdensities.utils.common import Inputs, init_precision, log
from lsdensities.correlator.synthetic_vv_correlator import generate_correlator

DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_bbounds_wnoise")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emin", type=float, default=0.3, help="Lowest energy, in GeV. Default=0.3")
    parser.add_argument("--emax", type=float, default=1.2, help="Highest energy, in GeV. Default=1.2")
    parser.add_argument("--ne", type=int, default=4, help="Number of energies. Default=4")
    parser.add_argument("--sigma", type=float, default=0.4, help="Smearing kernel width, in GeV. Default=0.4")
    parser.add_argument("--time-extent", type=int, default=24, help="Time extent of the mock correlator. Default=24")
    parser.add_argument("--nsamples", type=int, default=800, help="Number of mock bootstrap samples. Default=800")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the synthetic correlator. Default=1")
    parser.add_argument("--degrees", type=int, nargs="+", default=[12, 16, 20], help="Degrees of the polynomial approximation of the kernel. Default=12 16 20")
    parser.add_argument("--confidence", type=float, default=0.9, help="Confidence level of the bounds. Default=0.9")
    parser.add_argument("--mp-digits", type=int, default=100, help="Working precision of the SDP solver, in decimal digits. Default=100")
    parser.add_argument("--solver", type=str, default=None, help="SDP solver: MPMATH (high precision, default) or a cvxpy double-precision solver such as CLARABEL (faster, less reliable)")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory. Default=examples/test_bbounds_wnoise")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Accepted strings are 'WARNING', 'INFO' or 'DEBUG'. Default=INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    log("Initialising causal-bootstrap test with synthetic vector-vector-like data")

    par = Inputs()
    par.time_extent = args.time_extent
    par.periodicity = "EXP"
    par.kerneltype = "FULLNORMGAUSS"
    par.sigma = args.sigma
    par.emin = args.emin
    par.emax = args.emax
    par.Ne = args.ne
    par.Na = 1
    par.e0 = 0
    par.prec = 50
    par.num_boot = args.nsamples
    par.outdir = args.outdir
    par.directoryName = "run"
    par.loglevel = args.loglevel
    par.assign_values()
    par.apply_loglevel()
    init_precision(par.prec)
    par.plotpath = os.path.join(par.outdir, "Plots")
    par.logpath = os.path.join(par.outdir, "Logs")
    os.makedirs(par.plotpath, exist_ok=True)
    os.makedirs(par.logpath, exist_ok=True)
    par.report()

    espace = np.linspace(par.emin, par.emax, par.Ne)

    log("Generating synthetic vector-vector-like correlator with realistic noise")
    corr, rho_true = generate_correlator(
        par, espace, seed=args.seed, n_samples=args.nsamples
    )

    CB = CausalBootstrapAnalysis(
        par=par,
        correlator=corr,
        energies=espace,
        degrees=args.degrees,
        confidence=args.confidence,
        mp_digits=args.mp_digits,
        solver=args.solver,
    )
    CB.run()
    output_file = CB.save()
    log("Wrote causal-bootstrap data to", output_file)
    log(
        "Plot it with: python3 plot_output.py causal-bootstrap --file",
        output_file,
        "--energy <E>",
    )

    log("Energy \t True \t Lower \t Upper")
    for e_i in range(par.Ne):
        lower, upper = CB.lower[e_i], CB.upper[e_i]
        log(
            "{:2.3f} \t {:2.4f} \t {} \t {}".format(
                espace[e_i],
                rho_true[e_i],
                "-" if lower is None else "{:2.4f}".format(lower),
                "-" if upper is None else "{:2.4f}".format(upper),
            )
        )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(espace, rho_true, marker="o", ls="--", color="black", label="Exact")
    bounds = [(e, lo, hi) for e, lo, hi in zip(espace, CB.lower, CB.upper) if lo is not None]
    if bounds:
        e_b, lo_b, hi_b = map(np.array, zip(*bounds))
        plt.errorbar(
            e_b,
            (lo_b + hi_b) / 2,
            yerr=(hi_b - lo_b) / 2,
            marker="none",
            ls="",
            capsize=6,
            elinewidth=2,
            label=f"Causal bootstrap ({args.confidence:.0%} CL)",
        )
    plt.xlabel(r"$E$ [GeV]")
    plt.ylabel(r"$\rho_\sigma(E)$")
    plt.title("Smeared spectral density: bounds vs. exact")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(par.outdir, "BoundsVsTrue.png"), dpi=300)
    plt.close()

    log("Done. Output written to", par.outdir)


if __name__ == "__main__":
    main()
