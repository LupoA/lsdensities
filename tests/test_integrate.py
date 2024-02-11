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


import time

def integrate_exponential(alpha, s, t1,t2, E0, periodicity, T, precision):
    delta_x = 1e-3
    integral = 0.0

    x = E0
    while True:
        integral = mp.fadd(integral, integrandSigmaMat_DEBUG(x, alpha, s, t1, t2, E0, periodicity, T) * delta_x)
        x += delta_x

        if integrandSigmaMat_DEBUG(x, alpha, s, t1, t2, E0, periodicity, T) < precision:
            break

    return integral


def main():
    mp.dps = 120

    start = time.time()
    integral = mp.quad(lambda x: integrandSigmaMat_DEBUG(x, 0, s=0.1, t1=3, t2=3, E0=0, periodicity='COSH', T=16),
                       [0, mp.inf], error=True, method='tanh-sinh')
    end=time.time()
    print(LogMessage(), float(integral[0]), "in ", end-start, "s")

    start = time.time()
    integral = integrate_exponential(alpha=0, s=0.1, t1=3,t2=3, E0=0, periodicity='COSH', T=16, precision=1e-20)
    end=time.time()
    print(LogMessage(), float(integral), "in ", end-start, "s")

    exit(1)


if __name__ == "__main__":
    main()
