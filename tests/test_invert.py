import sys


import lattice-inverse-problem.rhoUtils as u
from lattice-inverse-problem.rhoUtils import init_precision
from lattice-inverse-problem.rhoUtils import LogMessage
from lattice-inverse-problem.rhoUtils import end
from lattice-inverse-problem.rhoUtils import Obs
from lattice-inverse-problem.rhoUtils import adjust_precision
from lattice-inverse-problem.rhoUtils import Inputs
from lattice-inverse-problem.rhoUtils import *
from lattice-inverse-problem.rhoStat import *
from lattice-inverse-problem.rhoMath import *
from lattice-inverse-problem.core import *
from lattice-inverse-problem.rhoParser import *
from lattice-inverse-problem.transform import *
from lattice-inverse-problem.abw import *
from lattice-inverse-problem.rhoParallelUtils import *
from lattice-inverse-problem.HLT_class import *
from lattice-inverse-problem.GPHLT_class import *
from lattice-inverse-problem.GP_class import *
from lattice-inverse-problem.correlatorUtils import foldPeriodicCorrelator
from lattice-inverse-problem.correlatorUtils import symmetrisePeriodicCorrelator
from mpmath import mp, mpf
from lattice-inverse-problem.InverseProblemWrapper import *
from lattice-inverse-problem.plotutils import *


def init_precision():
    print(LogMessage(), " Initialising...")
    mp.dps = 64
    print(LogMessage(), " Binary precision in bit: ", mp.prec)
    print(LogMessage(), " Approximate decimal precision: ", mp.dps)


if __name__ == "__main__":
    init_precision()
    tmax = 15

    S = Smatrix_mp(tmax, alpha_=0)

    invS = S ** (-1)

    identity = S * invS

    shouldbeone = norm2_mp(identity)

    print(LogMessage(), "norm( S Sinv ) - 1 = ", shouldbeone - 1)
    print(LogMessage(), "Target decimal precision was", mp.dps)

    assert shouldbeone - 1 < mp.dps / 2
    exit(1)
