import sys

sys.path.append("..")
sys.path.append("rhos")
sys.path.append("../rhos")


import rhos.rhoUtils as u
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
