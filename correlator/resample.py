import sys
sys.path.append("..")
from importall import *
from correlatorUtils import *

def init_variables(args_):
    in_ = InputsCorrelatorAnalysis()
    in_.prec = args_.prec
    in_.datapath = args_.datapath
    in_.outdir = args_.outdir
    in_.massNorm = args_.mpi
    in_.num_boot = args_.nboot
    in_.sigma = args_.sigma
    in_.emax = args_.emax * args_.mpi
    in_.Ne = args_.ne
    in_.alpha = args_.alpha
    in_.emin = args_.emin
    in_.prec = -1
    in_.plots = args_.plots
    return in_

def resample():
    return 0

def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentCorrelatorAnalysis()
    par = InputsCorrelatorAnalysis(datapath=args.datapath, outdir=args.outdir, num_boot=args.nboot)

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
    corr.plot(show=True, label='Correlator (bootstrap)')

    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)

if __name__ == "__main__":
    main()