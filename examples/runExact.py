import numpy as np
import matplotlib.pyplot as plt
from mpmath import mpf, mp
from lsdensities.utils.rhoUtils import (
    LogMessage,
    init_precision,
    Inputs,
    create_out_paths,
    end,
    generate_seed,
)
from lsdensities.utils.rhoParser import parseArgumentSynthData
from lsdensities.utils.rhoMath import gauss_fp, invert_matrix_ge, norm2_mp, cauchy
from lsdensities.core import Smatrix_mp
from lsdensities.transform import h_Et_mp_Eslice, y_combine_central_Eslice_mp
import random

pion_mass = 0.140  # in Gev
a = 0.4  # in Gev ^-1 ( 1 fm = 5.068 GeV^-1 )
a_fm = a / 5.068  # lattice spacing in fm
aMpi = pion_mass * a  # pion mass in lattice units
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
    in_.emax = args_.emax
    if args_.emin == 0:
        in_.emin = (args_.mpi / 20) * args_.mpi
    else:
        in_.emin = args_.emin
    in_.e0 = args_.e0
    in_.Ne = args_.ne
    in_.Na = args_.Na
    in_.A0cut = args_.A0cut
    return in_


def kernel_correlator(E, t, T, par):
    if par.periodicity == "COSH":
        return mp.exp(-mpf(E) * mpf(t)) + mp.exp(-mpf(E) * mpf(T - t))
    if par.periodicity == "EXP":
        return mp.exp(-mpf(E) * mpf(t))


def generate(par, espace):
    peaks_location = np.random.uniform(
        np.random.uniform(0.5 * aMpi, 2 * aMpi), (2.1 * par.emax) * aMpi, STATES
    )  #   generates STATES numbers between a random value in (0.25*aMpi, 3*aMpi) and 1.5*emax*aMpi
    weights = np.random.uniform(0, 0.004, len(peaks_location))

    exact_correlator = mp.matrix(par.tmax, 1)  #   Exact correlator

    for _t in range(par.tmax):
        for _n in range(STATES):
            exact_correlator[_t] += kernel_correlator(
                mpf(peaks_location[_n]), mpf(_t + 1), par.time_extent, par
            ) * mpf(weights[_n])

    rhoStrue = np.zeros(par.Ne)

    for e_i in range(par.Ne):
        for _n in range(STATES):
            if par.kerneltype == "FULLNORMGAUSS":
                rhoStrue[e_i] += (
                    gauss_fp(peaks_location[_n], espace[e_i], par.sigma, norm="full")
                    * weights[_n]
                )
            elif par.kerneltype == "HALFNORMGAUSS":
                rhoStrue[e_i] += (
                    gauss_fp(peaks_location[_n], espace[e_i], par.sigma, norm="half")
                    * weights[_n]
                )
            elif par.kerneltype == "CAUCHY":
                rhoStrue[e_i] += (
                    cauchy(peaks_location[_n], par.sigma, espace[e_i]) * weights[_n]
                )

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
    espace = np.linspace(
        par.emin * par.massNorm, par.emax * par.massNorm, par.Ne
    )  # TODO:   move this outside when you are done

    print(LogMessage(), "Energies [lattice units] : ", espace)
    print(LogMessage(), "Energies [Gev] : ", espace / a)
    print(LogMessage(), "Energies [1/M_pi] : ", espace / par.massNorm)
    print(LogMessage(), " Sigma [lattice units] : ", par.sigma)
    print(LogMessage(), " Sigma [GeV] : ", par.sigma / a)
    print(LogMessage(), " Sigma [1/M_pi] : ", par.sigma / par.massNorm)

    #   Init done

    seed = generate_seed(par)

    print(LogMessage(), " Random Seed : ", seed)

    random.seed(seed)
    np.random.seed(random.randint(0, 2 ** (32) - 1))

    exact_correlator, espace, rhoStrue = generate(par, espace)

    S = Smatrix_mp(
        tmax_=par.tmax, alpha_=0, e0_=mpf(0), type=par.periodicity, T=par.time_extent
    )

    Sinv = invert_matrix_ge(S)

    print("S Sinv - 1 = ", float(norm2_mp(S * Sinv) - 1))

    rhos = np.zeros(par.Ne)

    for e_i in range(len(espace)):
        print(LogMessage(), "Energy [a^-1]", espace[e_i])
        _g_t_estar = h_Et_mp_Eslice(Sinv, par, espace[e_i], alpha_=0)
        rhos[e_i] = y_combine_central_Eslice_mp(_g_t_estar, exact_correlator, par)
        print("Exact Rho: ", rhoStrue[e_i])
        print("Reconstructed Rho: ", rhos[e_i])

    plt.plot(
        espace / a,
        np.array(rhoStrue, dtype=float),
        marker="o",
        markersize=3.5,
        ls="-",
        label="Exact",
        color="k",
    )

    plt.plot(
        espace / a,
        np.array(rhos, dtype=float),
        marker="o",
        markersize=3.5,
        ls="-",
        label="Reconstructed",
        color="r",
    )
    plt.xlabel("GeV")
    plt.title("# States : {:2d}".format(STATES))
    plt.legend(prop={"size": 12, "family": "Helvetica"}, frameon=False)
    plt.show()
    end()


if __name__ == "__main__":
    main()
