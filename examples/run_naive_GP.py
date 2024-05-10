import lsdensities.utils.rhoUtils as u
from lsdensities.utils.rhoUtils import init_precision, LogMessage, end, generate_seed
from lsdensities.utils.rhoParser import parse_inputs
from lsdensities.utils.rhoStat import resample
from lsdensities.correlator.correlatorUtils import symmetrisePeriodicCorrelator
from mpmath import mp, mpf
import random
import numpy as np
from lsdensities.GP_class import (
    AlgorithmParameters,
    MatrixBundle,
    GaussianProcessWrapper,
)

#   use True if you have the ill conditioned matrix already stored in a file
#   use False if you need to compute it. Can take some time.
read_SIGMA_ = True


def main():
    '''
    This function solve the inverse problem using the traditional Bayesian approach based on Gaussian Processes.
    The output is smeared with an unconstrained kernel. If this is a problem, switch to InverseProblemWrapper rather than GaussianProcessWrapper.
    Hyperparameters: we use the negative log likelihood to determine lambda. At the same time we perform the HLT-type stability analysis.
    The smearing radius of the Gaussian model prior is not treated as a hyperparameter. It is an input (par.sigma).
    For completeness, the function returns both Bayesian and Frequentist error.
    These are unconventional choice, but this package is mainly interested in the fixed-smearing kernel solution,
    provided by InverseProblemWrapper, see [hep-lat/2311.18125]
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
    corr.corrmat_from_covmat(plot=False)

    #   Make it into a mp sample
    print(LogMessage(), "Converting correlator into mpmath type")
    corr.fill_mp_sample()
    print(LogMessage(), "Cond[Cov C] = {:3.3e}".format(float(mp.cond(corr.mpcov))))

    cNorm = mpf(str(corr.central[1] ** 2))
    lambdaMax = 1e3
    energies = np.linspace(par.emin, par.emax, par.Ne)

    hltParams = AlgorithmParameters(
        alphaA=0,
        alphaB=1.99,
        alphaC=0.5,
        lambdaMax=lambdaMax,
        lambdaStep=lambdaMax / 2,
        lambdaScanCap=8,
        kfactor=0.1,
        lambdaMin=5e-4,
        comparisonRatio=0.3,
    )
    matrix_bundle = MatrixBundle(Bmatrix=corr.mpcov, bnorm=cNorm)

    #   Wrapper for the Inverse Problem
    GP = GaussianProcessWrapper(
        par=par,
        algorithmPar=hltParams,
        matrix_bundle=matrix_bundle,
        correlator=corr,
        energies=energies,
        read_SIGMA=read_SIGMA_,
    )
    GP.prepareGP()

    #   Run
    GP.run()
    GP.stabilityPlot(
        generateHLTscan=True,
        generateLikelihoodShared=True,
        generateLikelihoodPlot=True,
        generateKernelsPlot=False,
    )
    GP.plotResult()
    end()

    end()


if __name__ == "__main__":
    main()
