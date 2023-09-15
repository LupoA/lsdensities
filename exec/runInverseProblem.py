import sys
import random
sys.path.append("..")
from importall import *


def init_variables(args_):
    in_ = Inputs()
    in_.tmax = args_.tmax
    in_.periodicity = args_.periodicity
    in_.prec = args_.prec
    in_.datapath = args_.datapath
    in_.outdir = args_.outdir
    in_.massNorm = args_.mpi
    in_.num_boot = args_.nboot
    in_.sigma = args_.sigma
    in_.emax = (
        args_.emax * args_.mpi
    )  #   we pass it in unit of Mpi, here to turn it into lattice (working) units
    if args_.emin == 0:
        in_.emin = (
            args_.mpi / 20
        ) * args_.mpi
    else:
        in_.emin = args_.emin * args_.mpi
    in_.e0 = args_.e0
    in_.Ne = args_.ne
    in_.Na = args_.Na
    in_.A0cut = args_.A0cut
    return in_


def main():
    print(LogMessage(), "Initialising")
    #random.seed(1994)
    args = parseArgumentRhoFromData()
    init_precision(args.prec)
    par = init_variables(args)

    #   Reading datafile, storing correlator
    rawcorr, par.time_extent, par.num_samples = u.read_datafile(par.datapath)
    par.assign_values()
    par.report()
    par.plotpath, par.logpath = create_out_paths(par)

    #   Correlator
    rawcorr.evaluate()
    rawcorr.tmax = par.tmax
    if par.periodicity == "COSH":
        print(LogMessage(), "Folding correlator")
        symCorr = symmetrisePeriodicCorrelator(corr=rawcorr, par=par)
        symCorr.evaluate()

    #   Resampling
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
    with open(os.path.join(par.logpath,'covarianceMatrix.txt'), "w") as output:
        for i in range(par.time_extent):
            for j in range(par.time_extent):
                print(i, j, corr.cov[i,j], file=output)
    #   -   -   -   -   -   -   -   -   -   -   -

    #   Turn correlator into mpmath variable
    print(LogMessage(), "Converting correlator into mpmath type")
    corr.fill_mp_sample()
    print(LogMessage(), "Cond[Cov C] = {:3.3e}".format(float(mp.cond(corr.mpcov))))

    #   Prepare
    cNorm = mpf(str(corr.central[1] ** 2))
    lambdaMax = 1e+4

    hltParams = AlgorithmParameters(
        alphaA=0,
        alphaB=1/2,
        alphaC=+1.99,
        lambdaMax=lambdaMax,
        lambdaStep=lambdaMax/2,
        lambdaScanCap=8,
        kfactor=0.1,
        lambdaMin=1e-1,
        comparisonRatio=0.15,    #   brutal! but we are comparing highly correlated measurements
    )
    matrix_bundle = MatrixBundle(Bmatrix=corr.mpcov, bnorm=cNorm)

    HLT = InverseProblemWrapper(par=par, algorithmPar=hltParams, matrix_bundle=matrix_bundle, correlator=corr, read_energies=0)
    HLT.prepareHLT()
    HLT.run()
    HLT.stabilityPlot(generateHLTscan=True, generateLikelihoodShared=True, generateLikelihoodPlot=True, generateKernelsPlot=True)

    end()

if __name__ == "__main__":
    main()
