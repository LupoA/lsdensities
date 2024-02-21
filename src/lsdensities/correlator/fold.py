from ..utils.rhoUtils import Obs, LogMessage, read_datafile
from ..utils.rhoParallelUtils import ParallelBootstrapLoop
from .correlatorUtils import foldPeriodicCorrelator, InputsCorrelatorAnalysis
from ..utils.rhoParser import parseArgumentCorrelatorAnalysis


def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentCorrelatorAnalysis()
    par = InputsCorrelatorAnalysis(
        datapath=args.datapath, outdir=args.outdir, num_boot=args.nboot
    )

    #   Reading datafile, storing correlator
    rawcorr, par.time_extent, par.num_samples = read_datafile(par.datapath)
    par.periodicity = args.periodicity
    if args.periodicity == "EXP":
        par.tmax = par.time_extent - 1
    elif args.periodicity == "COSH":
        par.tmax = int(par.time_extent / 2) + 1
    else:
        raise ValueError("Invalid type specified. Only COSH and EXP are allowed.")

    par.report()

    #   Here is the correlator
    rawcorr.evaluate()
    #    rawcorr.plot(label="raw data")

    #   Here is the folding
    foldedCorr = foldPeriodicCorrelator(corr=rawcorr, par=par, is_resampled=False)
    foldedCorr.evaluate()
    #    foldedCorr.plot(label="folded")

    #   Here is the resampling
    corr = Obs(int(par.time_extent / 2) + 1, par.num_boot, is_resampled=True)
    resample = ParallelBootstrapLoop(par, foldedCorr.sample, is_folded=True)
    corr.sample = resample.run()
    corr.evaluate()
    corr.plot(show=True, label="Correlator (bootstrap)")

    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)

    from correlatorUtils import effective_mass

    effmass = effective_mass(corr, par, type="COSH")
    effmass.plot(logscale=False)
    print(effmass.central, "±", effmass.err)

    exit(1)


if __name__ == "__main__":
    main()
