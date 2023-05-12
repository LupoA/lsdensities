import numpy as np
import sys
import math
sys.path.append("../utils")
from rhoUtils import *


#   Usage:
#       from correlatorUtils import effective_mass
#       effmass = effective_mass(corr, par, type='EXP')
#       effmass.plot(logscale=False)
#       print(effmass.avg, '±', effmass.err)

def effective_mass(corr, par, type='COSH'):
    th=int(par.time_extent/2)
    thm=th-1
    mass = Obs(T_=thm, nms_=par.num_boot, is_resampled = True)
    for i in range(0, thm):
        for b in range(0,par.num_boot):
            mass.sample[b,i] = math.acosh( (corr.sample[b,i+2] + corr.sample[b,i])/(2*corr.sample[b,i+1]) )
    mass.evaluate()
    return mass

def effective_mass(corr, par, type='EXP'):
    th=int(par.time_extent/2)
    thm=th-1
    mass = Obs(T_=thm, nms_=par.num_boot, is_resampled = True)
    for i in range(0, thm):
        for b in range(0,par.num_boot):
            mass.sample[b,i] = - math.log( corr.sample[b,i+1] /(corr.sample[b,i]) )
    mass.evaluate()
    return mass