import lsdensities.utils.rhoUtils as u
from lsdensities.utils.rhoUtils import (
    init_precision,
    LogMessage,
    end,
    generate_seed,
)

from lsdensities.correlator.correlatorUtils import symmetrisePeriodicCorrelator
from lsdensities.utils.rhoParser import parse_inputs
from lsdensities.utils.rhoStat import resample
import os
from mpmath import mp, mpf
import numpy as np
from lsdensities.InverseProblemWrapper import AlgorithmParameters, InverseProblemWrapper
from lsdensities.utils.rhoUtils import MatrixBundle
import random


def main():
    '''
    Solves the inverse problem with the HLT method and provides, at the same time, the corresponding Bayesian solution,
    which is based on a white-noise Gaussian Process, see [hep-lat/2311.18125].
    HLT solution: parameter lambda is chosen from a stability analysis.
    Bayesian solution: parameter lambda is chosen from the minimum of the negative log likelihood.
    Smearing kernel: fixed by par.ker_type with smearing radius par.sigma
    '''
    print(LogMessage(), "Initialising")
    par = parse_inputs()
    init_precision(par.prec)

    #   When a datafile is read, we don't assign values of par until the file is passed
    #   Reading datafile, storing correlator. Initialisation of par is inside read_datafile
    rawcorr = u.read_datafile(par, resampled=False) # specify if the correlator is resampled, since it affects
                                                    # the computation of its statistical error
    par.report()
    rawcorr.evaluate() # computes average correlator

    seed = generate_seed(par)
    random.seed(seed)
    np.random.seed(random.randint(0, 2 ** (32) - 1))

    #   Folding the correlator (if applicable)
    if par.periodicity == "COSH":
        print(LogMessage(), "Folding correlator")
        symCorr = symmetrisePeriodicCorrelator(corr=rawcorr, par=par)
        symCorr.evaluate()

    #   #   #   Resampling
    if par.periodicity == "EXP":
        corr = resample(rawcorr, par)
    if par.periodicity == "COSH":
        corr = resample(symCorr, par)
    #   -   -   -   -   -   -   -   -   -   -   -

    #   Covariance
    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
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
        generateKernelsPlot=False,
    )  # Lots of plots as it is
    HLT.plotResult()
    end()


if __name__ == "__main__":
    main()
