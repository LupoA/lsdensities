import lsdensities.utils.rhoUtils as u
from lsdensities.utils.rhoUtils import (
    init_precision,
    LogMessage,
    end,
    generate_seed,
)
from lsdensities.utils.rhoUtils import create_out_paths
from lsdensities.correlator.correlatorUtils import symmetrisePeriodicCorrelator
from lsdensities.utils.rhoParallelUtils import ParallelBootstrapLoop
from lsdensities.utils.rhoParser import parse_inputs
import os
from mpmath import mp, mpf
import numpy as np
from lsdensities.InverseProblemWrapper import AlgorithmParameters, InverseProblemWrapper
from lsdensities.utils.rhoUtils import MatrixBundle
import random


def main():
    print(LogMessage(), "Initialising")
    par = parse_inputs()
    par.init()
    init_precision(par.prec)
    par.report()

    seed = generate_seed(par)
    random.seed(seed)
    np.random.seed(random.randint(0, 2 ** (32) - 1))

    #   Reading datafile, storing correlator
    rawcorr, par.time_extent, par.num_samples = u.read_datafile(par.datapath)
    par.assign_values()
    par.report()
    par.plotpath, par.logpath = create_out_paths(par)

    #   Folding the correlator (if applicable)
    rawcorr.evaluate()
    rawcorr.tmax = par.tmax
    if par.periodicity == "COSH":
        print(LogMessage(), "Folding correlator")
        symCorr = symmetrisePeriodicCorrelator(corr=rawcorr, par=par)
        symCorr.evaluate()

    #   #   #   Resampling
    if par.periodicity == "EXP":
        corr = u.Obs(
            T=par.time_extent, tmax=par.tmax, nms=par.num_boot, is_resampled=True
        )
        resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)
    if par.periodicity == "COSH":
        corr = u.Obs(
            T=symCorr.T,
            tmax=symCorr.tmax,
            nms=par.num_boot,
            is_resampled=True,
        )
        resample = ParallelBootstrapLoop(par, symCorr.sample, is_folded=False)

    corr.sample = resample.run()
    corr.evaluate()
    #   -   -   -   -   -   -   -   -   -   -   -

    #   Covariance
    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)
    with open(os.path.join(par.logpath, "covarianceMatrix.txt"), "w") as output:
        for i in range(par.time_extent):
            for j in range(par.time_extent):
                print(i, j, corr.cov[i, j], file=output)
    #   -   -   -   -   -   -   -   -   -   -   -

    #   Turn correlator into mpmath variable
    print(LogMessage(), "Converting correlator into mpmath type")
    corr.fill_mp_sample()
    print(LogMessage(), "Cond[Cov C] = {:3.3e}".format(float(mp.cond(corr.mpcov))))

    #   Prepare
    cNorm = mpf(str(corr.central[1] ** 2))
    lambdaMax = 1e4
    energies = np.linspace(par.emin, par.emax, par.Ne)

    hltParams = AlgorithmParameters(
        alphaA=0,
        alphaB=1.99,
        alphaC=0.5,
        lambdaMax=lambdaMax,
        lambdaStep=lambdaMax / 2,
        lambdaScanCap=8,
        kfactor=0.1,
        lambdaMin=5e-2,
        comparisonRatio=0.3,
    )
    matrix_bundle = MatrixBundle(Bmatrix=corr.mpcov, bnorm=cNorm)

    HLT = InverseProblemWrapper(
        par=par,
        algorithmPar=hltParams,
        matrix_bundle=matrix_bundle,
        correlator=corr,
        energies=energies,
    )
    HLT.prepareHLT()
    HLT.run()
    HLT.stabilityPlot(
        generateHLTscan=True,
        generateLikelihoodShared=True,
        generateLikelihoodPlot=True,
        generateKernelsPlot=True,
    )  # Lots of plots as it is
    HLT.plotResult()
    end()


if __name__ == "__main__":
    main()
