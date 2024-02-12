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
    print(LogMessage(), " pproximate decimal precision: ", mp.dps)


if __name__ == "__main__":
    init_precision()
