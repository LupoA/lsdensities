import sys

sys.path.append("..")
from importall import *


def init_variables(args_):
    in_ = Inputs()
    in_.tmax = args_.tmax
    in_.periodicity = args_.periodicity
    in_.prec = args_.prec
    in_.datapath = args_.datapath
    in_.outdir = args_.outdir
    in_.plotpath, in_.logpath = create_out_paths(in_.outdir)
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
    adjust_precision(par.tmax)
    #   Here is the correlator
    rawcorr.evaluate()
    rawcorr.tmax = par.tmax

    #   Symmetrise
    if par.periodicity == "COSH":
        print(LogMessage(), "Folding correlator")
        #foldedCorr = foldPeriodicCorrelator(corr=rawcorr, par=par, is_resampled=False)
        #foldedCorr.evaluate()
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
        #resample = ParallelBootstrapLoop(par, foldedCorr.sample, is_folded=True)    # it should be but doesnt work
        #resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)
        resample = ParallelBootstrapLoop(par, symCorr.sample, is_folded=False)      # it should be but doesnt work
    if par.periodicity == "EXP":
        resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)

    corr.sample = resample.run()
    corr.evaluate()

    #   Plot all correlators
    plt.tight_layout()
    plt.grid(alpha=0.1)
    plt.yscale("log")
    plt.errorbar(
        x=list(range(0, rawcorr.T)),
        y=rawcorr.central,
        yerr=rawcorr.err,
        marker=plot_markers[0],
        markersize=1.5,
        elinewidth=1,
        ls="",
        label='Input Data',
        color=CB_color_cycle[0],
    )
    plt.errorbar(
        x=list(range(0, symCorr.T)),
        y=symCorr.central,
        yerr=symCorr.err,
        marker=plot_markers[1],
        markersize=3.5,
        elinewidth=3,
        ls="",
        label='Symmetrised Data',
        color=CB_color_cycle[1],
    )
    plt.errorbar(
        x=list(range(0, corr.T)),
        y=corr.central,
        yerr=corr.err,
        marker=plot_markers[2],
        markersize=2.5,
        elinewidth=1,
        ls="",
        label='Resampled Data (symmetrised)',
        color=CB_color_cycle[2],
    )
    plt.legend(prop={"size": 12, "family": "Helvetica"})
    plt.tight_layout()
    #plt.show()
    plt.clf()

    #   Covariance
    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)

    #   Make it into a mp sample
    print(LogMessage(), "Converting correlator into mpmath type")
    # mpcorr_sample = mp.matrix(par.num_boot, tmax)
    corr.fill_mp_sample()
    cNorm = mpf(str(corr.central[1] ** 2))

    #   Prepare
    hltParams = AlgorithmParameters(
        alphaA=0,
        alphaB=-1,
        alphaC=-1.99,
        lambdaMax=100,
        lambdaStep=5,
        lambdaScanPrec=1,
        lambdaScanCap=5,
        kfactor=10,
        lambdaMin=10
    )
    matrix_bundle = MatrixBundle(Bmatrix=corr.mpcov, bnorm=cNorm)

    #   Wrapper for the Inverse Problem
    HLT = HLTWrapper(
        par=par, algorithmPar=hltParams, matrix_bundle=matrix_bundle, correlator=corr
    )
    HLT.prepareHLT()

    #   Run
    HLT.run(how_many_alphas=par.Na)
    HLT.plotParameterScan(how_many_alphas=par.Na, save_plots=True)
    HLT.plotRhos(savePlot=True)

    end()


if __name__ == "__main__":
    main()
