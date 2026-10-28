"""
Eigen-space analysis (arXiv:2605.14652, Sec. III.B), HLT stability analysis
(arXiv:2605.14652, Sec. III.A) and causal-bootstrap bounds (arXiv:2605.20509)
on the same smeared spectral density, from the same synthetic
vector-vector-like correlator used by run_ea_wnoise.py, run_hlt_wnoise.py and
run_bbounds_wnoise.py (see lsdensities.correlator.synthetic_vv_correlator).

The bounds are computed once and stored next to the results of both
analyses. Saves, under --outdir/Logs, the eigen-space and stability-analysis
JSON files; under --outdir/Plots, the stability plot and the eigen-space plots
with the bounds overlaid, at every energy; and --outdir/ReconstructionVsTrue.png,
comparing the three results with the exact smeared spectral density. The HLT
result is shown with its frequentist (bootstrap) error; with --bayes, the
result at the minimum of the negative log-likelihood, with its Bayesian error,
is shown as well.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpmath import mp, mpf

import plot_output
from lsdensities.correlator.synthetic_vv_correlator import generate_correlator
from lsdensities.inverse_problem_solvers.hlt_eigenspace import HETpar, HLTWithSVD
from lsdensities.inverse_problem_solvers.hlt_stability import AlgorithmParameters, HLTWithBackusGilbert
from lsdensities.io_utils import load_json
from lsdensities.utils.common import Inputs, init_precision, log

DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_all_wnoise")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emin", type=float, default=0.3, help="Lowest reconstructed energy, in GeV. Default=0.3")
    parser.add_argument("--emax", type=float, default=1.2, help="Highest reconstructed energy, in GeV. Default=1.2")
    parser.add_argument("--ne", type=int, default=4, help="Number of reconstructed energies. Default=4")
    parser.add_argument("--sigma", type=float, default=0.4, help="Smearing kernel width, in GeV. Default=0.4")
    parser.add_argument("--time-extent", type=int, default=24, help="Time extent of the mock correlator. Default=24")
    parser.add_argument("--prec", type=int, default=50, help="Numerical precision, in decimal digits. Default=50")
    parser.add_argument("--nsamples", type=int, default=800, help="Number of mock bootstrap samples. Default=800")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the synthetic correlator. Default=1")
    parser.add_argument("--Na", type=int, default=3, choices=[1, 2, 3], help="Number of alpha values used to cross-check the stability analysis and the eigen-space truncation. Default=3")
    parser.add_argument("--A0cut", type=float, default=0.2, help="Stability analysis: maximum accepted A/A0. Default=0.2")
    parser.add_argument("--bayes", action="store_true", help="Also show the HLT result at the minimum of the negative log-likelihood, with its Bayesian error")
    parser.add_argument("--l-reg", type=float, default=1e-9, help="Eigen-space analysis: small regulator added to the eigenvalues to avoid dividing by (numerically) zero. Default=1e-9")
    parser.add_argument("--n-consecutive", type=int, default=2, help="Eigen-space analysis: number of consecutive eigenmodes compatible with zero required before truncating. Default=2")
    parser.add_argument("--cb-degrees", type=int, nargs="+", default=[16], help="Degrees of the polynomial approximation of the kernel for the causal-bootstrap bounds. Default=16")
    parser.add_argument("--cb-confidence", type=float, default=0.9, help="Confidence level of the causal-bootstrap bounds. Default=0.9")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory. Default=examples/test_all_wnoise")
    parser.add_argument("--loglevel", type=str, default="WARNING", help="Accepted strings are 'WARNING', 'INFO' or 'DEBUG'. Default=WARNING")
    return parser.parse_args()


def main():
    args = parse_args()
    log("Initialising eigen-space, HLT and causal-bootstrap test with synthetic vector-vector-like data")

    par = Inputs()
    par.time_extent = args.time_extent
    par.periodicity = "EXP"
    par.kerneltype = "FULLNORMGAUSS"
    par.sigma = args.sigma
    par.emin = args.emin
    par.emax = args.emax
    par.Ne = args.ne
    par.Na = args.Na
    par.e0 = 0
    par.prec = args.prec
    par.num_boot = args.nsamples
    par.A0cut = args.A0cut
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
    #   The Cholesky factor is needed by the eigen-space analysis only
    corr, rho_true = generate_correlator(
        par, espace, seed=args.seed, n_samples=args.nsamples, with_cholesky=True
    )
    log("Condition number of the correlator covariance: {:2.2e}".format(float(mp.cond(corr.mpcov))))

    #   Eigen-space analysis
    EA = HLTWithSVD(
        par=par,
        algorithmPar=HETpar(alphaA=0, alphaB=1.99, alphaC=0.5, n_consecutive=args.n_consecutive),
        correlator=corr,
        energies=espace,
        l_reg=mpf(str(args.l_reg)),
        useCOV=True,
    )
    EA.prepare()
    log("Running the eigen-space analysis")
    EA.run()

    #   Stability analysis
    hltParams = AlgorithmParameters(
        alphaA=0,
        alphaB=1.99,
        alphaC=0.5,
        lambdaMax=1e4,
        lambdaStep=2,
        lambdaScanCap=6,
        kfactor=0.1,
        lambdaMin=1e-5,
        comparisonRatio=0.3,
    )
    HLT = HLTWithBackusGilbert(
        par=par,
        algorithmPar=hltParams,
        B=corr.mpcov,
        bnorm=mpf(str(corr.central[1] ** 2)),
        correlator=corr,
        energies=espace,
    )
    HLT.prepareHLT()
    log("Running the stability analysis")
    HLT.run()

    #   Causal-bootstrap bounds: they depend on the data and on the smeared
    #   spectral density only, so they are computed once for both analyses
    log("Computing the causal-bootstrap bounds at every energy")
    bounds = HLT.compute_causal_bootstrap_bounds(degrees=args.cb_degrees, confidence=args.cb_confidence)
    EA.causal_bootstrap = bounds

    ea_file = EA.save()
    hlt_file = HLT.save()
    log("Wrote eigen-space data, with the bounds, to", ea_file)
    log("Wrote stability-analysis data, with the bounds, to", hlt_file)

    header = "Energy \t True \t EA (res +/- err) \t HLT (stat+sys)"
    if args.bayes:
        header += " \t Bayes (stat+sys)"
    print(header + " \t Causal bootstrap")
    for e_i in range(par.Ne):
        lower, upper = bounds.lower[e_i], bounds.upper[e_i]
        line = "{:2.3f} \t {:2.4f} \t {:2.4f} +/- {:2.4f} \t {:2.4f} +/- {:2.4f}".format(
            espace[e_i],
            rho_true[e_i],
            EA.channelA.res[e_i],
            EA.channelA.err[e_i],
            HLT.rhoResultHLT[e_i],
            HLT.rho_quadrature_err_HLT[e_i],
        )
        if args.bayes:
            line += " \t {:2.4f} +/- {:2.4f}".format(
                HLT.rhoResultBayes[e_i],
                HLT.rho_quadrature_err_Bayes[e_i],
            )
        line += " \t " + ("-" if lower is None else "[{:2.4f}, {:2.4f}]".format(lower, upper))
        print(line)

    plt.style.use(plot_output.DEFAULT_MPLSTYLE)

    #   Stability and eigen-space analyses with the bounds overlaid, at every energy
    hlt_data = load_json(hlt_file)
    ea_data = load_json(ea_file)
    for energy in espace:
        plot_output.plot_stability(hlt_data, energy, par.plotpath, show=False, show_causal_bootstrap=True)
        plot_output.plot_eigenspace(ea_data, energy, None, None, par.plotpath, show=False, show_causal_bootstrap=True)

    #   Small horizontal offsets, so that the error bars do not overlap
    de = 0.01 * (espace[-1] - espace[0]) if par.Ne > 1 else 0.01
    plt.figure(figsize=(8, 6))
    plt.plot(espace, rho_true, marker="o", ls="--", color="black", label="Exact")
    plt.errorbar(
        espace - de,
        EA.channelA.res,
        yerr=EA.channelA.err,
        marker="^",
        ls="",
        capsize=3,
        label="Eigen-space analysis",
    )
    plt.errorbar(
        espace + de,
        HLT.rhoResultHLT,
        yerr=HLT.rho_quadrature_err_HLT,
        marker="s",
        ls="",
        capsize=3,
        label="HLT",
    )
    if args.bayes:
        plt.errorbar(
            espace + 2 * de,
            HLT.rhoResultBayes,
            yerr=HLT.rho_quadrature_err_Bayes,
            marker="d",
            ls="",
            capsize=3,
            label="HLT (Bayesian NLL)",
        )
    bounded = [(e, lo, hi) for e, lo, hi in zip(espace, bounds.lower, bounds.upper) if lo is not None]
    if bounded:
        e_b, lo_b, hi_b = map(np.array, zip(*bounded))
        plt.errorbar(
            e_b,
            (lo_b + hi_b) / 2,
            yerr=(hi_b - lo_b) / 2,
            marker="none",
            ls="",
            capsize=6,
            elinewidth=2,
            color="gray",
            alpha=0.6,
            label=f"Causal bootstrap ({args.cb_confidence:.0%} CL)",
        )
    plt.xlabel(r"$E$ [GeV]")
    plt.ylabel(r"$\rho_\sigma(E)$")
    plt.title("Smeared spectral density: reconstructed vs. exact")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(par.outdir, "ReconstructionVsTrue.png"), dpi=300)
    plt.close()

    log("Done. Output written to", par.outdir)


if __name__ == "__main__":
    main()
