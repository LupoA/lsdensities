import sys

sys.path.append("..")

sys.path.append("..")
sys.path.append("rhos")
sys.path.append("../rhos")


import rhoUtils as u
from .rhoUtils import init_precision
from .rhoUtils import LogMessage
from .rhoUtils import end
from .rhoUtils import Obs
from .rhoUtils import adjust_precision
from .rhoUtils import Inputs
from .rhoUtils import *
from .rhoStat import *
from .rhoMath import *
from .core import *
from .rhoParser import *
from .transform import *
from .abw import *
from .rhoParallelUtils import *
from .HLT_class import *
from .GPHLT_class import *
from .GP_class import *
from .correlatorUtils import foldPeriodicCorrelator
from .correlatorUtils import symmetrisePeriodicCorrelator
from .mpmath import mp, mpf
from .InverseProblemWrapper import *
from .plotutils import *


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
    args = parseArgumentRhoFromData()
    init_precision(args.prec)
    par = init_variables(args)

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
        #resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)
        resample = ParallelBootstrapLoop(par, symCorr.sample, is_folded=False)
    if par.periodicity == "EXP":
        resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)

    corr.sample = resample.run()
    corr.evaluate()

    #   Covariance
    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)

    with open(os.path.join(par.outdir,'corrmatrix.txt'), "w") as output:
        for i in range(par.time_extent):
            for j in range(par.time_extent):
                print(i, j, corr.corrmat[i,j], file=output)

    #   Make it into a mp sample
    print(LogMessage(), "Converting correlator into mpmath type")
    # mpcorr_sample = mp.matrix(par.num_boot, tmax)
    corr.fill_mp_sample()
    print(LogMessage(), "Cond[Cov C] = {:3.3e}".format(float(mp.cond(corr.mpcov))))

    cNorm = mpf(str(corr.central[1] ** 2))

    lambdaMax = 1e+8

    #   Prepare
    hltParams = AlgorithmParameters(
        alphaA=0,
        alphaB=1/2,
        alphaC=+1.99,
        lambdaMax=lambdaMax,
        lambdaStep=lambdaMax/2,
        lambdaScanPrec=1,
        lambdaScanCap=16,
        kfactor=0.1,
        lambdaMin=1e-6
    )
    matrix_bundle = MatrixBundle(Bmatrix=corr.mpcov, bnorm=cNorm)

    #   Wrapper for the Inverse Problem
    HLT = HLTWrapper(
        par=par, algorithmPar=hltParams, matrix_bundle=matrix_bundle, correlator=corr
    )
    HLT.prepareHLT()

    #   Run
    HLT.run(how_many_alphas=par.Na)
    HLT.plotParameterScan(how_many_alphas=par.Na, save_plots=True, plot_live=True)
    HLT.plotRhos(savePlot=True)

    end()


if __name__ == "__main__":
    main()
