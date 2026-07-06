"""
Test of the HLT method (InverseProblemWrapper) with the stability
analysis of arXiv:2605.14652 (Sec. III.A), on a synthetic vector-vector-like
correlator with a realistic, growing-with-time statistical error (see
syntheticVVCorrelator.py).

Saves the full stability-analysis scan to a JSON file under --outdir (see
src/lsdensities/ioutils.py for the schema) and a plot comparing the
reconstructed smeared spectral density against the known exact one. Use
examples/plot_output.py to plot the stability analysis (arXiv:2605.14652
Fig. 6) at a given energy from the saved JSON.

example usage
python plot_output.py stability --file test_hlt_wnoise/Logs/HLT_....json --energy 0.9 --outdir path/to/plot
"""

import argparse
import os

import numpy as np
from mpmath import mp, mpf

from lsdensities.InverseProblemWrapper import AlgorithmParameters, InverseProblemWrapper
from lsdensities.utils.rhoUtils import Inputs, init_precision, log
from syntheticVVCorrelator import generate_correlator

DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_hlt_wnoise")


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
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory. Default=examples/test_hlt_wnoise")
    return parser.parse_args()


def main():
    args = parse_args()
    log("Initialising HLT stability-analysis test with synthetic vector-vector-like data")

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
    par.assign_values()
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
    HLT = InverseProblemWrapper(
        par=par,
        algorithmPar=hltParams,
        B=corr.mpcov,
        bnorm=cNorm,
        correlator=corr,
        energies=espace,
    )
    HLT.prepareHLT()
    HLT.run()
    output_file = HLT.save()
    log("Wrote stability-analysis data to", output_file)
    log(
        "Plot it with: python3 plot_output.py stability --file",
        output_file,
        "--energy <E>",
    )

    log("Energy \t True \t HLT (stat+sys) \t Bayes (stat+sys)")
    for e_i in range(par.Ne):
        log(
            "{:2.3f} \t {:2.4f} \t {:2.4f} +/- {:2.4f} \t {:2.4f} +/- {:2.4f}".format(
                espace[e_i],
                rho_true[e_i],
                HLT.rhoResultHLT[e_i],
                HLT.rho_quadrature_err_HLT[e_i],
                HLT.rhoResultBayes[e_i],
                HLT.rho_quadrature_err_Bayes[e_i],
            )
        )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(espace, rho_true, marker="o", ls="--", color="black", label="Exact")
    plt.errorbar(
        espace,
        HLT.rhoResultHLT,
        yerr=HLT.rho_quadrature_err_HLT,
        marker="s",
        ls="",
        capsize=3,
        label="HLT (Backus-Gilbert)",
    )
    plt.errorbar(
        espace,
        HLT.rhoResultBayes,
        yerr=HLT.rho_quadrature_err_Bayes,
        marker="d",
        ls="",
        capsize=3,
        label="HLT (Bayesian NLL)",
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
