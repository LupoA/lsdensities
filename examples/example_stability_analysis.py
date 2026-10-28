"""
Stability analysis of the HLT method (arXiv:2605.14652, Sec. III.A and
Fig. 6) at a single energy, with the causal-bootstrap bounds
(arXiv:2605.20509) on the same smeared spectral density overlaid.

The data are the synthetic vector-vector-like correlator of run_hlt_wnoise.py
(see lsdensities.correlator.synthetic_vv_correlator), and the default energy
is the second one of the energies reconstructed there, E = 0.6 GeV.

rho_sigma(E) is shown as a function of the regulator lambda, for every value
of alpha, together with the HLT result and the band allowed by the causal
bootstrap. Saves the stability-analysis JSON under --outdir/Logs and the plot
under --outdir/Plots.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpmath import mpf

import plot_output
from lsdensities.inverse_problem_solvers.hlt_stability import AlgorithmParameters, HLTWithBackusGilbert
from lsdensities.io_utils import load_json
from lsdensities.utils.common import Inputs, init_precision, log
from lsdensities.correlator.synthetic_vv_correlator import generate_correlator

DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_stability_analysis")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy", type=float, default=0.6, help="Energy at which rho_sigma is reconstructed, in GeV. Default=0.6")
    parser.add_argument("--sigma", type=float, default=0.4, help="Smearing kernel width, in GeV. Default=0.4")
    parser.add_argument("--time-extent", type=int, default=24, help="Time extent of the mock correlator. Default=24")
    parser.add_argument("--prec", type=int, default=50, help="Numerical precision, in decimal digits. Default=50")
    parser.add_argument("--nsamples", type=int, default=800, help="Number of mock bootstrap samples. Default=800")
    parser.add_argument("--seed", type=int, default=1, help="Seed for the synthetic correlator. Default=1")
    parser.add_argument("--Na", type=int, default=3, choices=[1, 2, 3], help="Number of alpha values used to cross-check the stability analysis. Default=3")
    parser.add_argument("--A0cut", type=float, default=0.2, help="Maximum accepted A/A0. Default=0.2")
    parser.add_argument("--cb-degrees", type=int, nargs="+", default=[16], help="Degrees of the polynomial approximation of the kernel for the causal-bootstrap bounds. Default=16")
    parser.add_argument("--cb-confidence", type=float, default=0.9, help="Confidence level of the causal-bootstrap bounds. Default=0.9")
    parser.add_argument("--show", action="store_true", help="Display the plot instead of saving it")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory. Default=examples/example_stability_analysis")
    parser.add_argument("--loglevel", type=str, default="WARNING", help="Accepted strings are 'WARNING', 'INFO' or 'DEBUG'. Setting 'INFO' shows the details of the scan over lambda and alpha. Default=WARNING")
    return parser.parse_args()


def main():
    args = parse_args()

    par = Inputs()
    par.time_extent = args.time_extent
    par.periodicity = "EXP"
    par.kerneltype = "FULLNORMGAUSS"
    par.sigma = args.sigma
    par.emin = args.energy
    par.emax = args.energy
    par.Ne = 1
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

    espace = np.array([args.energy])
    #   Same mock spectrum as run_hlt_wnoise.py, whose default emax is 1.2 GeV
    corr, rho_true = generate_correlator(
        par, espace, seed=args.seed, n_samples=args.nsamples, continuum_emax=3 * 1.2
    )

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
    log("Running the stability analysis at E =", args.energy)
    HLT.run()
    log("Computing the causal-bootstrap bounds")
    HLT.compute_causal_bootstrap_bounds(degrees=args.cb_degrees, confidence=args.cb_confidence)
    output_file = HLT.save()
    log("Wrote stability-analysis data, with the bounds, to", output_file)

    lower, upper = HLT.causal_bootstrap.lower[0], HLT.causal_bootstrap.upper[0]
    print("E = {:2.3f} GeV, sigma = {:2.3f} GeV".format(args.energy, args.sigma))
    print("  exact            {:2.4f}".format(rho_true[0]))
    print("  HLT (stat+sys)   {:2.4f} +/- {:2.4f}".format(HLT.rhoResultHLT[0], HLT.rho_quadrature_err_HLT[0]))
    print("  causal bootstrap {}".format("-" if lower is None else "[{:2.4f}, {:2.4f}]".format(lower, upper)))

    plt.style.use(plot_output.DEFAULT_MPLSTYLE)
    plot_output.plot_stability(
        load_json(output_file), args.energy, par.plotpath, args.show, show_causal_bootstrap=True
    )


if __name__ == "__main__":
    main()
