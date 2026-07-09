"""
Demo for lsdensities.correlator.correlator_utils: read a raw correlator, fold
it if periodic, bootstrap-resample it, and look at its covariance and
effective mass. Merges what used to be two near-identical scripts
(fold.py and resample.py) into one, since folding is just the COSH-only step
that comes before the (otherwise identical) resample/covariance/mass pipeline.
"""

from lsdensities.utils.common import Obs, LogMessage, read_datafile
from lsdensities.utils.parallel_utils import ParallelBootstrapLoop
from lsdensities.correlator.correlator_utils import (
    effective_mass,
    foldPeriodicCorrelator,
    InputsCorrelatorAnalysis,
    parseArgumentCorrelatorAnalysis,
)


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

    #   Folding (COSH only) before resampling
    if par.periodicity == "COSH":
        print(LogMessage(), "Folding correlator")
        foldedCorr = foldPeriodicCorrelator(corr=rawcorr, par=par)
        foldedCorr.evaluate()
        sample_to_resample = foldedCorr.sample
        corr_T = int(par.time_extent / 2) + 1
    else:
        sample_to_resample = rawcorr.sample
        corr_T = par.time_extent

    #   Here is the resampling
    corr = Obs(T=corr_T, tmax=par.tmax, nms=par.num_boot, sample_type="bootstrap")
    resample = ParallelBootstrapLoop(
        par, sample_to_resample, is_folded=(par.periodicity == "COSH")
    )
    corr.sample = resample.run()
    corr.evaluate()
    corr.plot(show=True, label="Correlator (bootstrap)")

    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)

    effmass = effective_mass(corr, par, type=par.periodicity)
    effmass.plot(logscale=False)
    print(LogMessage(), "Effective mass", effmass.central, "±", effmass.err)


if __name__ == "__main__":
    main()
