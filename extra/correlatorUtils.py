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

def effective_mass_beta(corr, par, type='COSH'):
    th = int(par.time_extent / 2)
    thm = th - 1
    mass = Obs(T_=thm, nms_=par.num_boot, is_resampled=True)
    if type == 'COSH':
        mass.sample[:, :] = np.arccosh((corr.sample[:, 2:th+1] + corr.sample[:, 0:th-1]) / (2 * corr.sample[:, 1:th]))
    elif type == 'EXP':
        mass.sample[:, :] = -np.log(corr.sample[:, 1:th] / corr.sample[:, 0:th-1])
    else:
        raise ValueError('Invalid type specified. Only COSH and EXP are allowed.')

    mass.evaluate()
    return mass