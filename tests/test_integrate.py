import sys


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
