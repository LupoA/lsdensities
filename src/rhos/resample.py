import sys

import rhoUtils as u
from rhos.rhoUtils import init_precision
from rhos.rhoUtils import LogMessage
from rhos.rhoUtils import end
from rhos.rhoUtils import Obs
from rhos.rhoUtils import adjust_precision
from rhos.rhoUtils import Inputs
from rhos.rhoUtils import *
from rhos.rhoStat import *
from rhos.rhoMath import *
from rhos.core import *
from rhos.rhoParser import *
from rhos.transform import *
from rhos.abw import *
from rhos.rhoParallelUtils import *
from rhos.HLT_class import *
from rhos.GPHLT_class import *
from rhos.GP_class import *
from rhos.correlatorUtils import foldPeriodicCorrelator
from rhos.correlatorUtils import symmetrisePeriodicCorrelator
from mpmath import mp, mpf
from rhos.InverseProblemWrapper import *
from rhos.plotutils import *


from .correlatorUtils import *


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
