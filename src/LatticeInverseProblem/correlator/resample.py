import sys

import rhoUtils as u
from ..rhoUtils import init_precision
from ..rhoUtils import LogMessage
from ..rhoUtils import end
from ..rhoUtils import Obs
from ..rhoUtils import adjust_precision
from ..rhoUtils import Inputs
from ..rhoUtils import *
from ..rhoStat import *
from ..rhoMath import *
from ..core import *
from ..rhoParser import *
from ..transform import *
from ..abw import *
from ..rhoParallelUtils import *
from ..HLT_class import *
from ..GPHLT_class import *
from ..GP_class import *
from ..correlatorUtils import foldPeriodicCorrelator
from ..correlatorUtils import symmetrisePeriodicCorrelator
from mpmath import mp, mpf
from ..InverseProblemWrapper import *
from ..plotutils import *


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
