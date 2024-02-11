import sys

import hltrho.rhoUtils as u
from hltrho.rhoUtils import init_precision
from hltrho.rhoUtils import LogMessage
from hltrho.rhoUtils import end
from hltrho.rhoUtils import Obs
from hltrho.rhoUtils import adjust_precision
from hltrho.rhoUtils import Inputs
from hltrho.rhoUtils import *
from hltrho.rhoStat import *
from hltrho.rhoMath import *
from hltrho.core import *
from hltrho.rhoParser import *
from hltrho.transform import *
from hltrho.abw import *
from hltrho.rhoParallelUtils import *
from hltrho.HLT_class import *
from hltrho.GPHLT_class import *
from hltrho.GP_class import *
from hltrho.correlatorUtils import foldPeriodicCorrelator
from hltrho.correlatorUtils import symmetrisePeriodicCorrelator
from mpmath import mp, mpf
from hltrho.InverseProblemWrapper import *
from hltrho.plotutils import *


def init_precision():
    print(LogMessage(), " Initialising...")
    mp.dps = 64
    print(LogMessage(), " Binary precision in bit: ", mp.prec)
    print(LogMessage(), " pproximate decimal precision: ", mp.dps)


if __name__ == "__main__":
    init_precision()
