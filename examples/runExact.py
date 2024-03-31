import numpy as np
import matplotlib.pyplot as plt
from mpmath import mpf, mp
from lsdensities.utils.rhoUtils import (
    LogMessage,
    init_precision,
    Inputs,
    create_out_paths,
    end,
    generate_seed
)
from lsdensities.utils.rhoParser import parse_synthetic_inputs
from lsdensities.utils.rhoMath import gauss_fp, invert_matrix_ge, norm2_mp, cauchy
from lsdensities.core import Smatrix_mp
from lsdensities.transform import h_Et_mp_Eslice, y_combine_central_Eslice_mp
import random

pion_mass = 0.140  # Gev
a = 0.4  # in Gev ^-1 ( 1 fm = 5.068 GeV^-1 )
a_fm = a / 5.068  # lattice spacing in fm
aMpi = pion_mass * a  # pion mass in lattice units
STATES = 666

def kernel_correlator(E, t, T, par):
    if par.periodicity == "COSH":
        return mp.exp(-mpf(E) * mpf(t)) + mp.exp(-mpf(E) * mpf(T - t))
    if par.periodicity == "EXP":
        return mp.exp(-mpf(E) * mpf(t))


def generate(par, espace):
    '''
    generates a correlator of dimension [1/a] with n=STATES states
    '''
    peaks_location = np.random.uniform(
        np.random.uniform(2*pion_mass, 3 * pion_mass), (3 * par.emax), STATES
    )
    first_peak = np.random.uniform(0.8 * pion_mass, 1.2 * pion_mass, 1)
    peaks_location = np.concatenate([first_peak, peaks_location])
    peaks_location *= a
    weights = np.random.uniform(0, 0.004, len(peaks_location))
    weights /= a #  gives dimension

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
    par = parse_synthetic_inputs()
    par.init()
    par.report()
    espace = np.linspace(
        par.emin, par.emax, par.Ne
    )

    print(LogMessage(), "Energies [Gev] : ", espace)
    print(LogMessage(), " Sigma [GeV] : ", par.sigma)

    seed = generate_seed(par)
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

    plt.plot(
        espace,
        np.array(rhoStrue, dtype=float),
        marker="o",
        markersize=3.5,
        ls="-",
        label="Exact",
        color="k",
    )

    plt.plot(
        espace,
        np.array(rhos, dtype=float),
        marker="o",
        markersize=3.5,
        ls="-",
        label="Reconstructed",
        color="r",
    )
    plt.xlabel("GeV")
    plt.title("# States : {:2d}".format(STATES))
    plt.legend()
    plt.show()
    end()


if __name__ == "__main__":
    main()
