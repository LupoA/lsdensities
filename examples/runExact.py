from numpy.random import normal
from os import path
import numpy as np
from scipy.stats import norm
from numpy import *
import matplotlib.pyplot as plt
import os
import sys

from mpmath import mpf


import numpy as np
import os
import matplotlib.pyplot as plt


import LatticeInverseProblem.utils.rhoUtils as u
from LatticeInverseProblem.utils.rhoUtils import init_precision
from LatticeInverseProblem.utils.rhoUtils import LogMessage
from LatticeInverseProblem.utils.rhoUtils import end
from LatticeInverseProblem.utils.rhoUtils import Obs
from LatticeInverseProblem.utils.rhoUtils import adjust_precision
from LatticeInverseProblem.utils.rhoUtils import Inputs
from LatticeInverseProblem.utils.rhoUtils import *
from LatticeInverseProblem.utils.rhoStat import *
from LatticeInverseProblem.utils.rhoMath import *
from LatticeInverseProblem.core import *
from LatticeInverseProblem.utils.rhoParser import *
from LatticeInverseProblem.transform import *
from LatticeInverseProblem.abw import *
from LatticeInverseProblem.utils.rhoParallelUtils import *
from LatticeInverseProblem.HLT_class import *
from LatticeInverseProblem.GPHLT_class import *
from LatticeInverseProblem.GP_class import *
from LatticeInverseProblem.correlator.correlatorUtils import foldPeriodicCorrelator
from LatticeInverseProblem.correlator.correlatorUtils import symmetrisePeriodicCorrelator
from mpmath import mp, mpf
from LatticeInverseProblem.InverseProblemWrapper import *
from LatticeInverseProblem.plotutils import *
import LatticeInverseProblem


pion_mass = 0.140       # in Gev
a = 0.4                 # in Gev ^-1 ( 1 fm = 5.068 GeV^-1 )
a_fm = a / 5.068        # lattice spacing in fm
aMpi = pion_mass * a    # pion mass in lattice units
STATES = 666

#   We only pass lattice units variables. GeV or mass scales are used in plots only

def init_variables(args_):
    in_ = Inputs()
    in_.time_extent = args_.T
    in_.num_samples = args_.nms
    in_.tmax = args_.tmax
    in_.periodicity = args_.periodicity
    in_.prec = args_.prec
    in_.outdir = args_.outdir
    in_.kerneltype = args_.kerneltype
    in_.massNorm = args_.mpi
    in_.num_boot = args_.nboot
    in_.sigma = args_.sigma
    in_.emax = (
        args_.emax
    )
    if args_.emin == 0:
        in_.emin = (
            args_.mpi / 20
        ) * args_.mpi
    else:
        in_.emin = args_.emin
    in_.e0 = args_.e0
    in_.Ne = args_.ne
    in_.Na = args_.Na
    in_.A0cut = args_.A0cut
    return in_

def kernel_correlator(E,t,T, par):
    if par.periodicity == "COSH":
        return mp.exp(-mpf(E)*mpf(t)) + mp.exp(-mpf(E)*mpf(T-t))
    if par.periodicity == "EXP":
        return mp.exp(-mpf(E) * mpf(t))

def generate(par, seed, espace):

    this_seed = seed
    random.seed(this_seed)
    np.random.seed(this_seed)

    peaks_location = np.random.uniform(np.random.uniform(0.5*aMpi, 2*aMpi), (2.1*par.emax)*aMpi, STATES)   #   generates STATES numbers between a random value in (0.25*aMpi, 3*aMpi) and 1.5*emax*aMpi
    weights = np.random.uniform(0, 0.004, len(peaks_location))


    exact_correlator = mp.matrix(par.tmax,1)                    #   Exact correlator

    for _t in range(par.tmax):
        for _n in range(STATES):
            exact_correlator[_t] += kernel_correlator(mpf(peaks_location[_n]), mpf(_t+1), par.time_extent, par) * mpf(weights[_n])

    rhoStrue = np.zeros(par.Ne)

    for e_i in range(par.Ne):
        for _n in range(STATES):
            rhoStrue[e_i] += gauss_fp(peaks_location[_n], espace[e_i], par.sigma, norm="full") * weights[_n]

    if par.periodicity == "EXP":
        return exact_correlator, espace, rhoStrue
    if par.periodicity == "COSH":
        return exact_correlator, espace, rhoStrue

def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentSynthData()
    init_precision(args.prec)
    par = init_variables(args)
    par.massNorm = pion_mass
    par.assign_values()
    par.plotpath, par.logpath = create_out_paths(par)
    par.report()
    print(par.emin * par.massNorm)
    print(par.emax * par.massNorm)
    espace = np.linspace(par.emin * par.massNorm, par.emax * par.massNorm, par.Ne)  # TODO:   move this outside when you are done

    print(LogMessage(), "Energies [lattice units] : ", espace)
    print(LogMessage(), "Energies [Gev] : ", espace / a)
    print(LogMessage(), "Energies [1/M_pi] : ", espace / par.massNorm)
    print(LogMessage(), " Sigma [lattice units] : ", par.sigma)
    print(LogMessage(), " Sigma [GeV] : ", par.sigma / a)
    print(LogMessage(), " Sigma [1/M_pi] : ", par.sigma / par.massNorm)

    #   Init done

    nseed = 1995
    random.seed(nseed)
    np.random.seed(nseed)
    ITERATIONS = 1
    seeds = [random.randint(1, 1e+6) for _ in range(ITERATIONS)]

    exact_correlator, espace, rhoStrue = generate(par, seeds[-1], espace)

    S = Smatrix_mp(tmax_ = par.tmax, alpha_=0, e0_ = mpf(0), type = par.periodicity, T = par.time_extent)

    Sinv = invert_matrix_ge(S)

    print("S Sinv - 1 = ", float(norm2_mp(S*Sinv) - 1))

    rhos = np.zeros(par.Ne)

    for e_i in range(len(espace)):
        print(LogMessage(), "Energy [a^-1]", espace[e_i])
        _g_t_estar = h_Et_mp_Eslice(Sinv, par, espace[e_i], alpha_=0)
        rhos[e_i] = y_combine_central_Eslice_mp(_g_t_estar, exact_correlator, par)

    plt.plot(
        espace/a,
        np.array(rhoStrue, dtype=float),
        marker="o",
        markersize=3.5,
        ls="-",
        label="Exact",
        color='k',
    )

    plt.plot(
        espace/a,
        np.array(rhos, dtype=float),
        marker="o",
        markersize=3.5,
        ls="-",
        label="Reconstructed",
        color='r',
    )
    plt.xlabel("GeV")
    plt.title("# States : {:2d}".format(STATES))
    plt.legend(prop={"size": 12, "family": "Helvetica"}, frameon=False)
    plt.show()
    end()

if __name__ == "__main__":
    main()

