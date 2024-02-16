import rhoUtils as u
from ..rhoUtils import LogMessage
from ..utils.rhoParser import parseArgumentCorrelatorAnalysis
from ..utils.rhoParallelUtils import ParallelBootstrapLoop
from .correlatorUtils import InputsCorrelatorAnalysis


def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentCorrelatorAnalysis()
    par = InputsCorrelatorAnalysis(
        datapath=args.datapath, outdir=args.outdir, num_boot=args.nboot
    )

    #   Reading datafile, storing correlator
    rawcorr, par.time_extent, par.num_samples = u.read_datafile(par.datapath)
    par.report()

    #   Here is the correlator
    rawcorr.evaluate()

    #   Here is the resampling
    corr = u.Obs(par.time_extent, par.num_boot, is_resampled=True)
    resample = ParallelBootstrapLoop(par, rawcorr.sample)
    corr.sample = resample.run()
    corr.evaluate()
    corr.plot(show=True, label="Correlator (bootstrap)")

    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)


if __name__ == "__main__":
    main()
