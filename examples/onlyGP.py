import lsdensities.utils.rhoUtils as u
from lsdensities.utils.rhoUtils import init_precision, LogMessage, end
from lsdensities.utils.rhoParser import parse_inputs
from lsdensities.utils.rhoUtils import create_out_paths
from lsdensities.correlator.correlatorUtils import symmetrisePeriodicCorrelator
from lsdensities.utils.rhoParallelUtils import ParallelBootstrapLoop
from mpmath import mp, mpf
from lsdensities.GP_class import (
    AlgorithmParameters,
    MatrixBundle,
    GaussianProcessWrapper,
)

read_SIGMA_ = True


def main():
    print(LogMessage(), "Initialising")
    par = parse_inputs()
    par.init()
    init_precision(par.prec)
    par.report()

    #   Reading datafile, storing correlator
    rawcorr, par.time_extent, par.num_samples = u.read_datafile(par.datapath)
    par.assign_values()
    par.report()
    par.plotpath, par.logpath = create_out_paths(par)

    #   Here is the correlator
    rawcorr.evaluate()
    rawcorr.tmax = par.tmax

    #   Symmetrise
    if par.periodicity == "COSH":
        print(LogMessage(), "Folding correlator")
        symCorr = symmetrisePeriodicCorrelator(corr=rawcorr, par=par)
        symCorr.evaluate()

    #   Here is the resampling
    if par.periodicity == "EXP":
        corr = u.Obs(
            T=par.time_extent, tmax=par.tmax, nms=par.num_boot, is_resampled=True
        )
    if par.periodicity == "COSH":
        corr = u.Obs(
            T=symCorr.T,
            tmax=symCorr.tmax,
            nms=par.num_boot,
            is_resampled=True,
        )

    if par.periodicity == "COSH":
        # resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)
        resample = ParallelBootstrapLoop(par, symCorr.sample, is_folded=False)
    if par.periodicity == "EXP":
        resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)

    corr.sample = resample.run()
    corr.evaluate()

    #   Covariance
    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)

    #   Make it into a mp sample
    print(LogMessage(), "Converting correlator into mpmath type")
    corr.fill_mp_sample()
    print(LogMessage(), "Cond[Cov C] = {:3.3e}".format(float(mp.cond(corr.mpcov))))

    cNorm = mpf(str(corr.central[1] ** 2))

    lambdaMax = 1e6

    #   Prepare
    hltParams = AlgorithmParameters(
        alphaA=0,
        alphaB=1 / 2,
        alphaC=1.99,
        lambdaMax=lambdaMax,
        lambdaStep=lambdaMax / 2,
        lambdaScanPrec=1,
        lambdaScanCap=24,
        kfactor=0.1,
        lambdaMin=1e-4,
    )
    matrix_bundle = MatrixBundle(Bmatrix=corr.mpcov, bnorm=cNorm)

    #   Wrapper for the Inverse Problem
    GP = GaussianProcessWrapper(
        par=par,
        algorithmPar=hltParams,
        matrix_bundle=matrix_bundle,
        correlator=corr,
        read_SIGMA=read_SIGMA_,
    )
    GP.prepareGP()

    #   Run
    GP.run(how_many_alphas=par.Na)
    GP.plotParameterScan(how_many_alphas=par.Na, save_plots=True, plot_live=True)
    GP.plothltrho(savePlot=True)

    end()


if __name__ == "__main__":
    main()
