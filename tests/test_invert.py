import sys


import lsdensities.rhoUtils as u
from lsdensities.rhoUtils import init_precision
from lsdensities.rhoUtils import LogMessage
from lsdensities.rhoUtils import end
from lsdensities.rhoUtils import Obs
from lsdensities.rhoUtils import adjust_precision
from lsdensities.rhoUtils import Inputs
from lsdensities.rhoUtils import *
from lsdensities.rhoStat import *
from lsdensities.rhoMath import *
from lsdensities.core import *
from lsdensities.rhoParser import *
from lsdensities.transform import *
from lsdensities.abw import *
from lsdensities.rhoParallelUtils import *
from lsdensities.HLT_class import *
from lsdensities.GPHLT_class import *
from lsdensities.GP_class import *
from lsdensities.correlatorUtils import foldPeriodicCorrelator
from lsdensities.correlatorUtils import symmetrisePeriodicCorrelator
from mpmath import mp, mpf
from lsdensities.InverseProblemWrapper import *
from lsdensities.plotutils import *


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
