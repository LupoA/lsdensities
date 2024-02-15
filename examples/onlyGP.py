import sys


import lsdensities.utils.rhoUtils as u
from lsdensities.utils.rhoUtils import init_precision
from lsdensities.utils.rhoUtils import LogMessage
from lsdensities.utils.rhoUtils import end
from lsdensities.utils.rhoUtils import Obs
from lsdensities.utils.rhoUtils import adjust_precision
from lsdensities.utils.rhoUtils import Inputs
from lsdensities.utils.rhoUtils import *
from lsdensities.utils.rhoStat import *
from lsdensities.utils.rhoMath import *
from lsdensities.core import *
from lsdensities.utils.rhoParser import *
from lsdensities.transform import *
from lsdensities.abw import *
from lsdensities.utils.rhoParallelUtils import *
from lsdensities.HLT_class import *
from lsdensities.GPHLT_class import *
from lsdensities.GP_class import *
from lsdensities.correlator.correlatorUtils import foldPeriodicCorrelator
from lsdensities.correlator.correlatorUtils import symmetrisePeriodicCorrelator
from mpmath import mp, mpf
from lsdensities.InverseProblemWrapper import *
from lsdensities.plotutils import *
import lsdensities

read_SIGMA_ = True

def init_variables(args_):
    in_ = Inputs()
    in_.tmax = args_.tmax
    in_.periodicity = args_.periodicity
    in_.prec = args_.prec
    in_.datapath = args_.datapath
    in_.kerneltype = args_.kerneltype
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

    #   Make it into a mp sample
    print(LogMessage(), "Converting correlator into mpmath type")
    corr.fill_mp_sample()
    print(LogMessage(), "Cond[Cov C] = {:3.3e}".format(float(mp.cond(corr.mpcov))))

    cNorm = mpf(str(corr.central[1] ** 2))

    lambdaMax = 1e+6

    #   Prepare
    hltParams = lsdensities.GP_class.AlgorithmParameters(
        alphaA=0,
        alphaB=1/2,
        alphaC=1.99,
        lambdaMax=lambdaMax,
        lambdaStep=lambdaMax/2,
        lambdaScanPrec=1,
        lambdaScanCap=24,
        kfactor=0.1,
        lambdaMin=1e-4
    )
    matrix_bundle = lsdensities.GP_class.MatrixBundle(Bmatrix=corr.mpcov, bnorm=cNorm)

    #   Wrapper for the Inverse Problem
    GP = lsdensities.GP_class.GaussianProcessWrapper(
        par=par, algorithmPar=hltParams, matrix_bundle=matrix_bundle, correlator=corr, read_SIGMA=read_SIGMA_
    )
    GP.prepareGP()

    #   Run
    GP.run(how_many_alphas=par.Na)
    GP.plotParameterScan(how_many_alphas=par.Na, save_plots=True, plot_live=True)
    GP.plothltrho(savePlot=True)

    end()


if __name__ == "__main__":
    main()
