import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from lsdensities.utils.rhoUtils import (
    LogMessage,
    generate_seed,
    Obs,
    CB_colors,
)
from lsdensities.utils.rhoParser import parse_synthetic_inputs
from lsdensities.utils.rhoMath import gauss_fp, cauchy
import random
from lsdensities.InverseProblemWrapper import AlgorithmParameters, InverseProblemWrapper
from lsdensities.utils.rhoUtils import MatrixBundle
import json

pion_mass = 0.140  # Gev
a = 0.41  # in Gev ^-1 ( 1 fm = 5.068 GeV^-1 )
a_fm = a / 5.068  # lattice spacing in fm ~ 0.08
aMpi = pion_mass * a  # pion mass in lattice units
nms = 1000


def kernel_correlator(E, t, T, par):
    if par.periodicity == "COSH":
        return np.exp(-(E) * (t)) + np.exp(-(E) * (T - t))
    if par.periodicity == "EXP":
        return mp.exp(-(E) * (t))


def generate(par, espace, STATES):
    """
    generates a correlator of dimension [1/a] with n=STATES states
    """
    first_peak = np.random.uniform(0.8 * pion_mass, 1.2 * pion_mass, 1)
    peaks_location = np.random.uniform(
        np.random.uniform(2 * pion_mass, 3 * pion_mass), (3 * par.emax), STATES
    )
    peaks_location = np.concatenate([first_peak, peaks_location])
    peaks_location *= a

    weights = np.random.uniform(0, 0.1, len(peaks_location))
    # weights /= a #  gives dimension

    exact_correlator = np.zeros(par.time_extent)  #   Exact correlator
    exact_cov = np.zeros((par.time_extent, par.time_extent))

    for _t in range(par.time_extent):
        for _n in range(STATES):
            exact_correlator[_t] += (
                kernel_correlator((peaks_location[_n]), (_t), par.time_extent, par)
                * weights[_n]
            )

    for t in range(par.time_extent):
        for r in range(par.time_extent):
            if t == r:
                exact_cov[t][r] = 1 * (exact_correlator[t] * 0.03) ** 2
            else:
                exact_cov[t][r] = exact_cov[t][t] * np.exp(-abs(t - r) / 2.5)
    exact_cov = (exact_cov + exact_cov.T) / 2

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

    return exact_correlator, exact_cov, espace, rhoStrue


def main():
    print(LogMessage(), "Initialising")
    par = parse_synthetic_inputs()
    par.assign_values()
    par.report()
    espace = np.linspace(par.emin, par.emax, par.Ne)

    print(LogMessage(), "Energies [Gev] : ", espace)
    print(LogMessage(), " Sigma [GeV] : ", par.sigma)

    seed = generate_seed(par)
    random.seed(seed)
    np.random.seed(random.randint(0, 2 ** (32) - 1))

    NRUN = 2
    filename = (
        "data_Ne_"
        + str(par.Ne)
        + "_sigma_"
        + str(par.sigma)
        + "_RUNS_"
        + str(NRUN)
        + "_Tmax_"
        + str(par.tmax)
        + "_"
        + str(par.periodicity)
        + ".json"
    )
    all_data = {}

    for c in range(NRUN):
        print("RUN ", c)
        STATES = random.randint(8, 100)

        exact_correlator, exact_cov, espace, rhoStrue = generate(par, espace, STATES)
        fake_corr = Obs(T=par.time_extent, tmax=par.tmax, nms=nms, is_resampled=True)
        fake_corr.sample = np.random.multivariate_normal(
            exact_correlator, exact_cov, nms
        )
        fake_corr.evaluate()
        fake_corr.evaluate_covmatrix()
        fake_corr.fill_mp_sample()

        cNorm = fake_corr.central[1] ** 2
        lambdaMax = 1e4
        lambdaMin = 0.5e-6
        lcap = 6
        lplat = 2
        energies = np.linspace(par.emin, par.emax, par.Ne)

        hltParams = AlgorithmParameters(
            alphaA=0,
            lambdaMax=lambdaMax,
            lambdaStep=lambdaMax / 2,
            lambdaScanCap=lcap,
            plateau_id=lplat,
            kfactor=0.1,
            lambdaMin=lambdaMin,
            comparisonRatio=0.4,
            resize=2,
        )
        matrix_bundle = MatrixBundle(Bmatrix=fake_corr.mpcov, bnorm=cNorm)

        HLT = InverseProblemWrapper(
            par=par,
            algorithmPar=hltParams,
            matrix_bundle=matrix_bundle,
            correlator=fake_corr,
            energies=energies,
        )
        HLT.prepareHLT()
        HLT.run()

        run_data = {}

        for e_i in range(par.Ne):
            run_data[espace[e_i]] = {
                "diff_hlt": (-rhoStrue[e_i] + HLT.rhoResultHLT[e_i]),
                "diff_bayes": (-rhoStrue[e_i] + HLT.rhoResultBayes[e_i]),
                "pull_hlt": (-rhoStrue[e_i] + HLT.rhoResultHLT[e_i])
                / HLT.rho_quadrature_err_HLT[e_i],
                "pull_bayes": (-rhoStrue[e_i] + HLT.rhoResultBayes[e_i])
                / HLT.rho_quadrature_err_Bayes[e_i],
                "hlt_full_err": HLT.rho_quadrature_err_HLT[e_i],
                "bayes_full_err": HLT.rho_quadrature_err_Bayes[e_i],
                "hlt_stat": HLT.drho_result[e_i],
                "bayes_stat": HLT.drho_bayes[e_i],
            }

        all_data[f"RUN_{c}"] = run_data

        with open(filename, "w") as json_file:
            json.dump(all_data, json_file, indent=4)

    exit()

    if NRUN == 1:
        plt.plot(
            espace,
            np.array(rhoStrue, dtype=float),
            marker="o",
            markersize=3.5,
            ls="--",
            label="Exact",
            color="gray",
        )
        # plt.title("Statistical error")
        plt.errorbar(
            espace,
            HLT.rhoResultHLT,
            HLT.drho_result,
            label="BG",
            color=CB_colors[0],
            marker="o",
            capsize=3.5,
        )
        plt.errorbar(
            espace,
            HLT.rhoResultBayes,
            HLT.drho_bayes,
            label="GP",
            color=CB_colors[1],
            marker="d",
            capsize=3.5,
        )
        plt.xticks(fontsize="large")
        plt.yticks(fontsize="large")
        plt.legend(fontsize="large")
        plt.xlabel(r"$E$ [GeV]")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.plot(
            espace,
            np.array(rhoStrue, dtype=float),
            marker="o",
            markersize=3.5,
            ls="--",
            label="Exact",
            color="gray",
        )
        # plt.title("Statistica and systematic error")
        plt.errorbar(
            espace,
            HLT.rhoResultHLT,
            HLT.rho_quadrature_err_HLT,
            label="BG",
            color=CB_colors[0],
            marker="s",
            capsize=3.5,
        )
        plt.errorbar(
            espace,
            HLT.rhoResultBayes,
            HLT.rho_quadrature_err_Bayes,
            label="GP",
            color=CB_colors[1],
            marker="d",
            capsize=3.5,
        )
        plt.xticks(fontsize="large")
        plt.yticks(fontsize="large")
        plt.legend(fontsize="large")
        plt.xlabel(r"$E$  [GeV]")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
