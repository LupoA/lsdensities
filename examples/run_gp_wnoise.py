"""
Gaussian-process method (GaussianProcessWrapper) with the stability analysis.
Still needs a lot of work, not a state-of-art gaussian process implementation!

Saves the full stability-analysis scan to a JSON file under --outdir (see
src/lsdensities/io_utils.py for the schema) and a plot comparing the
reconstructed smeared spectral density against the known exact one. Use
examples/plot_output.py to plot the stability analysis (arXiv:2605.14652
Fig. 6) at a given energy from the saved JSON.
"""

import argparse
import os

import numpy as np
from mpmath import mp, mpf

from lsdensities.inverse_problem_solvers.gaussian_process import AlgorithmParameters, GaussianProcessWrapper
from lsdensities.utils.common import Inputs, init_precision, log
from syntheticVVCorrelator import generate_correlator

DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_gp_wnoise")


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
    parser.add_argument("--Na", type=int, default=3, choices=[1, 2, 3], help="Number of alpha values used to cross-check the stability analysis. Default=3")
    parser.add_argument("--A0cut", type=float, default=0.2, help="Maximum accepted A/A0. Default=0.2")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory. Default=examples/test_gp_wnoise")
    parser.add_argument("--loglevel", type=str, default="WARNING", help="Accepted strings are 'WARNING', 'INFO' or 'DEBUG'. Setting 'INFO' shows the details of the scan over lambda and alpha. Default=WARNING")
    return parser.parse_args()


def main():
    args = parse_args()
    log("Initialising Gaussian-process stability-analysis test with synthetic vector-vector-like data")

    par = Inputs()
    par.time_extent = args.time_extent
    par.periodicity = "EXP"
    par.kerneltype = "FULLNORMGAUSS"  # required by GaussianProcessWrapper's prior
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
    corr, rho_true = generate_correlator(
        par, espace, seed=args.seed, n_samples=args.nsamples
    )
    log("Condition number of the correlator covariance: {:2.2e}".format(float(mp.cond(corr.mpcov))))

    cNorm = mpf(str(corr.central[1] ** 2))
    gpParams = AlgorithmParameters(
        alphaA=0,
        alphaB=1.99,
        alphaC=0.5,
        lambdaMax=1e6,
        lambdaStep=5e5,
        lambdaScanCap=6,
        kfactor=0.1,
        lambdaMin=1e-4,
        comparisonRatio=0.3,
    )
    GP = GaussianProcessWrapper(
        par=par,
        algorithmPar=gpParams,
        B=corr.mpcov,
        bnorm=cNorm,
        correlator=corr,
        energies=espace,
    )
    GP.prepareGP()
    GP.run()
    output_file = GP.save()
    log("Wrote stability-analysis data to", output_file)
    log(
        "Plot it with: python3 plot_output.py stability --file",
        output_file,
        "--energy <E>",
    )

    log("Energy \t True \t GP (stat+sys) \t GP-Bayes (stat+sys)")
    for e_i in range(par.Ne):
        log(
            "{:2.3f} \t {:2.4f} \t {:2.4f} +/- {:2.4f} \t {:2.4f} +/- {:2.4f}".format(
                espace[e_i],
                rho_true[e_i],
                GP.rhoResultHLT[e_i],
                GP.rho_quadrature_err_HLT[e_i],
                GP.rhoResultBayes[e_i],
                GP.rho_quadrature_err_Bayes[e_i],
            )
        )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(espace, rho_true, marker="o", ls="--", color="black", label="Exact")
    plt.errorbar(
        espace,
        GP.rhoResultHLT,
        yerr=GP.rho_quadrature_err_HLT,
        marker="s",
        ls="",
        capsize=3,
        label="GP (plateau)",
    )
    plt.errorbar(
        espace,
        GP.rhoResultBayes,
        yerr=GP.rho_quadrature_err_Bayes,
        marker="d",
        ls="",
        capsize=3,
        label="GP (Bayesian NLL)",
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
