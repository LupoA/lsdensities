import numpy as np
import os
import matplotlib.pyplot as plt
from .utils.rhoUtils import CB_colors, plot_markers, LogMessage, tnr
from .transform import combine_base_Eslice
from .utils.rhoMath import gauss_fp


def setPlotOpt(plt):
    plt.rcParams["figure.figsize"] = 5, 2
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rc("xtick", labelsize=22)
    plt.rc("ytick", labelsize=22)
    plt.rcParams.update({"font.size": 22})


def plotwErr(ax, x, y, yerr, label="", markerId=0, colorID=0):
    ax.errorbar(
        x=np.array(x, dtype=float),
        y=np.array(y, dtype=float),
        yerr=np.array(yerr, dtype=float),
        marker=plot_markers[markerId],
        markersize=4.8,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=label,
        color="black",
        ecolor=CB_colors[colorID],
        markerfacecolor=CB_colors[colorID],
    )


def plotNoErr(ax, x, y, label="", markerId=0, colorID=0):
    ax.errorbar(
        x=np.array(x, dtype=float),
        y=np.array(y, dtype=float),
        marker=plot_markers[markerId],
        markersize=4.8,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=label,
        color="black",
        ecolor=CB_colors[colorID],
        markerfacecolor=CB_colors[colorID],
    )


def stabilityPlot(invLapW, estar, savePlot=True, plot_live=False):
    setPlotOpt(plt)
    fig, ax = plt.subplots(figsize=(8, 10))
    plt.title(
        r"$E/M_{\rm ref}$"
        + "= {:2.2f}  ".format(estar / invLapW.par.massNorm)
        + r" $\;\;\; \sigma$"
        + r" = {:2.2f} ".format(invLapW.par.sigma / invLapW.par.massNorm)
        + r"$M_{\rm ref}$"
    )
    if invLapW.par.Na > 1:
        _a0label = r"$\alpha = {:1.2f}$".format(invLapW.algorithmPar.alphaA)
    elif invLapW.par.Na == 1:
        _a0label = r"$\rho_\sigma$"

    plotwErr(
        ax,
        invLapW.lambda_list[invLapW.espace_dictionary[estar]],
        invLapW.rho_list[invLapW.espace_dictionary[estar]],
        invLapW.errBoot_list[invLapW.espace_dictionary[estar]],
        markerId=0,
        colorID=0,
        label=_a0label,
    )

    if invLapW.par.Na > 1:
        plotwErr(
            ax,
            invLapW.lambda_list[invLapW.espace_dictionary[estar]],
            invLapW.rho_list_alphaB[invLapW.espace_dictionary[estar]],
            invLapW.errBoot_list_alphaB[invLapW.espace_dictionary[estar]],
            markerId=1,
            colorID=1,
            label=r"$\alpha = {:1.2f}$".format(invLapW.algorithmPar.alphaB),
        )
        if invLapW.par.Na > 2:
            plotwErr(
                ax,
                invLapW.lambda_list[invLapW.espace_dictionary[estar]],
                invLapW.rho_list_alphaC[invLapW.espace_dictionary[estar]],
                invLapW.errBoot_list_alphaC[invLapW.espace_dictionary[estar]],
                markerId=2,
                colorID=2,
                label=r"$\alpha = {:1.2f}$".format(invLapW.algorithmPar.alphaC),
            )

    ax.axhspan(
        ymin=float(
            invLapW.rhoResultHLT[invLapW.espace_dictionary[estar]]
            - invLapW.drho_result[invLapW.espace_dictionary[estar]]
        ),
        ymax=float(
            invLapW.rhoResultHLT[invLapW.espace_dictionary[estar]]
            + invLapW.drho_result[invLapW.espace_dictionary[estar]]
        ),
        alpha=0.3,
        color=CB_colors[4],
    )
    ax.set_xlabel(r"$\lambda$", fontsize=32)
    ax.set_ylabel(r"$\rho_\sigma$", fontsize=32)
    ax.legend(prop={"size": 26, "family": "Helvetica"}, frameon=False)
    ax.set_xscale("log")
    plt.tight_layout()
    if savePlot is True:
        plt.savefig(
            os.path.join(
                invLapW.par.plotpath,
                "LambdaScanE{:2.2e}".format(invLapW.espace_dictionary[estar]) + ".png",
            ),
            dpi=420,
        )
    if plot_live is True:
        plt.show()
    plt.close(fig)
    return


def sharedPlot_stabilityPlusLikelihood(invLapW, estar, savePlot=True, plot_live=False):
    setPlotOpt(plt)
    fig, (ax, ax2) = plt.subplots(
        nrows=2, sharex=True, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 2]}
    )

    plotwErr(
        ax,
        invLapW.lambda_list[invLapW.espace_dictionary[estar]],
        invLapW.rho_list[invLapW.espace_dictionary[estar]],
        invLapW.errBayes_list[invLapW.espace_dictionary[estar]],
        label="Bayesian Error",
        markerId=0,
        colorID=0,
    )

    plotwErr(
        ax,
        invLapW.lambda_list[invLapW.espace_dictionary[estar]],
        invLapW.rho_list[invLapW.espace_dictionary[estar]],
        invLapW.errBoot_list[invLapW.espace_dictionary[estar]],
        label="Bootstrap Error",
        markerId=1,
        colorID=1,
    )

    ax2.set_xlabel(r"$\lambda$", fontsize=32)
    ax.set_ylabel(r"$\rho_\sigma$", fontsize=32)
    ax.legend(prop={"size": 26, "family": "Helvetica"}, frameon=False)
    ax.set_xscale("log")

    plotNoErr(
        ax2,
        invLapW.lambda_list[invLapW.espace_dictionary[estar]],
        invLapW.likelihood_list[invLapW.espace_dictionary[estar]],
        label="NLL",
        markerId=0,
        colorID=0,
    )

    ax2.plot(
        invLapW.lambdaResultBayes[invLapW.espace_dictionary[estar]],
        invLapW.minNLL[invLapW.espace_dictionary[estar]],
        markersize=25,
        marker="*",
        markerfacecolor="red",
        markeredgecolor="black",
        color="red",
        ls="--",
        label="Min NLL",
    )

    ax2.set_ylabel("NLL", fontsize=32)
    plt.subplots_adjust(hspace=0)
    ax.set_title(
        r"$E/M_{\rm ref}$"
        + "= {:2.2f}  ".format(estar / invLapW.par.massNorm)
        + r" $\sigma$"
        + " = {:2.2f} ".format(invLapW.par.sigma / invLapW.par.massNorm)
        + r" $M_{\rm ref}$"
    )
    plt.tight_layout()
    if savePlot is True:
        plt.savefig(
            os.path.join(
                invLapW.par.plotpath,
                "LikelihoodLambdaScanE{:2.2e}".format(invLapW.espace_dictionary[estar])
                + ".png",
            ),
            dpi=420,
        )
    if plot_live is True:
        plt.show()
    plt.close(fig)
    return


def plotLikelihood(invLapW, estar, savePlot=True, plot_live=False):
    setPlotOpt(plt)
    fig, ax = plt.subplots(figsize=(8, 10))
    plt.title(
        r"$E/M_{\rm ref}$"
        + "= {:2.2f}  ".format(estar / invLapW.par.massNorm)
        + r" $\;\;\; \sigma$"
        + r" = {:2.2f} ".format(invLapW.par.sigma / invLapW.par.massNorm)
        + r" $M_{\rm ref}$"
    )
    plotNoErr(
        ax,
        invLapW.lambda_list[invLapW.espace_dictionary[estar]],
        invLapW.likelihood_list[invLapW.espace_dictionary[estar]],
        label="NLL",
        markerId=0,
        colorID=0,
    )
    if invLapW.par.Na > 1:
        plotNoErr(
            ax,
            invLapW.lambda_list[invLapW.espace_dictionary[estar]],
            invLapW.likelihood_list_alphaB[invLapW.espace_dictionary[estar]],
            label="NLL " + r"$\alpha = {:1.2f}$".format(invLapW.algorithmPar.alphaB),
            markerId=1,
            colorID=1,
        )
        if invLapW.par.Na > 2:
            plotNoErr(
                ax,
                invLapW.lambda_list[invLapW.espace_dictionary[estar]],
                invLapW.likelihood_list_alphaC[invLapW.espace_dictionary[estar]],
                label="NLL "
                + r"$\alpha = {:1.2f}$".format(invLapW.algorithmPar.alphaC),
                markerId=2,
                colorID=2,
            )
    ax.plot(
        invLapW.lambdaResultBayes[invLapW.espace_dictionary[estar]],
        invLapW.minNLL[invLapW.espace_dictionary[estar]],
        markersize=25,
        marker="*",
        markerfacecolor="red",
        markeredgecolor="black",
        color="red",
        ls="--",
        label="Minimum",
    )
    ax.set_ylabel("NLL", fontsize=32)
    ax.set_xlabel(r"$\lambda$", fontsize=32)
    ax.legend(prop={"size": 26, "family": "Helvetica"}, frameon=False)
    ax.set_xscale("log")
    plt.tight_layout()
    if savePlot is True:
        plt.savefig(
            os.path.join(
                invLapW.par.plotpath,
                "LikelihoodLambdaOnlyE{:2.2e}".format(invLapW.espace_dictionary[estar])
                + ".png",
            ),
            dpi=420,
        )
    if plot_live is True:
        plt.show()
    plt.close(fig)
    return


def plotAllKernels(invLapW):
    setPlotOpt(plt)
    print(LogMessage(), "Plotting kernel functions")
    _name = "HLTCoefficientsAlpha" + str(float(invLapW.algorithmPar.alphaA)) + ".txt"
    with open(os.path.join(invLapW.par.logpath, _name), "w") as output:
        for _e in range(invLapW.par.Ne):
            print(invLapW.espace[_e], invLapW.gt_HLT[_e], file=output)
            plotKernel(
                invLapW,
                invLapW.gt_HLT[_e],
                ne_=40,
                omega=invLapW.espace[_e],
                label="HLT",
                alpha_=invLapW.algorithmPar.alphaA,
                ker_type=invLapW.par.kerneltype,
            )

    _name = "BayesCoefficientsAlpha" + str(float(invLapW.algorithmPar.alphaA)) + ".txt"
    with open(os.path.join(invLapW.par.logpath, _name), "w") as output:
        for _e in range(invLapW.par.Ne):
            print(invLapW.espace[_e], invLapW.gt_Bayes[_e], file=output)
            plotKernel(
                invLapW,
                invLapW.gt_Bayes[_e],
                ne_=40,
                omega=invLapW.espace[_e],
                label="Bayes",
                alpha_=invLapW.algorithmPar.alphaA,
                ker_type=invLapW.par.kerneltype,
            )


def plotKernel(invLapW, gt_, omega, alpha_, label, ne_=70, ker_type="FULLNORMGAUSS"):
    setPlotOpt(plt)
    fig, ax = plt.subplots(figsize=(8, 10))
    energies = np.linspace(invLapW.par.massNorm * 0.05, invLapW.par.massNorm * 3.0, ne_)
    kernel = np.zeros(ne_)
    for _e in range(len(energies)):
        kernel[_e] = combine_base_Eslice(gt_, invLapW.par, energies[_e])
    plt.plot(
        energies / invLapW.par.massNorm,
        kernel,
        marker="o",
        markersize=3.8,
        ls="--",
        label=label
        + " Kernel at $\omega/M_{ref}$ = "
        + "{:2.1e}".format(omega / invLapW.par.massNorm),
        color="black",
        markerfacecolor=CB_colors[0],
    )

    y_to_plot = []
    if ker_type == "CAUCHY":

        def ker(k, sigma_, omega_):
            aux = omega_ - k
            aux = aux * aux + sigma_ * sigma_
            aux = sigma_ / aux
            return aux

        y_to_plot = ker(energies, invLapW.par.sigma, omega)

    elif ker_type == "FULLNORMGAUSS":
        y_to_plot = gauss_fp(energies, omega, invLapW.par.sigma, norm="full")
    elif ker_type == "HALFNORMGAUSS":
        y_to_plot = gauss_fp(energies, omega, invLapW.par.sigma, norm="half")

    plt.plot(
        energies / invLapW.par.massNorm,
        y_to_plot,
        ls="-",
        label="Target",
        color="red",
        linewidth=1.2,
    )

    with open(os.path.join(invLapW.par.logpath, f"kernel_{omega}.txt"), "a") as output:
        for e_i in range(len(energies)):
            print(energies[e_i] / invLapW.par.massNorm, kernel[e_i], file=output)

    plt.title(
        r" $\sigma$"
        + " = {:2.2f}".format(invLapW.par.sigma / invLapW.par.massNorm)
        + r"$M_{\rm ref}$ "
        + " $\;$ "
        + r"$\alpha$ = {:2.2f}".format(alpha_)
    )
    plt.xlabel(r"$E / M_{\rm ref}$", fontdict=tnr)
    plt.legend(prop={"size": 12, "family": "Helvetica"}, frameon=False)
    plt.savefig(
        os.path.join(
            invLapW.par.plotpath,
            label
            + "SmearingKernelSigma{:2.2e}".format(invLapW.par.sigma)
            + "Enorm{:2.2e}".format(invLapW.par.massNorm)
            + "Energy{:2.2e}".format(omega)
            + "Alpha{:2.2f}".format(alpha_)
            + ".png",
        ),
        dpi=400,
    )
    plt.clf()
    return


def plotSpectralDensity(invLapW):
    print(LogMessage(), "Plotting spectral density")
    setPlotOpt(plt)
    fig, ax = plt.subplots(figsize=(8, 10))
    plt.title(
        r" $\;\;\; \sigma$"
        + r" = {:2.2f}".format(invLapW.par.sigma / invLapW.par.massNorm)
        + r"$M_{\rm ref}$"
    )

    plotwErr(
        ax,
        invLapW.espace / invLapW.par.massNorm,
        np.array(invLapW.rhoResultHLT, dtype=float),
        np.array(invLapW.rho_quadrature_err_HLT, dtype=float),
        label="HLT",
        markerId=0,
        colorID=0,
    )

    plotwErr(
        ax,
        invLapW.espace / invLapW.par.massNorm,
        np.array(invLapW.rhoResultBayes, dtype=float),
        np.array(invLapW.rho_quadrature_err_Bayes, dtype=float),
        label="Bayesian",
        markerId=1,
        colorID=1,
    )

    plt.xlabel(r"$E / M_{\rm ref}$", fontdict=tnr)
    plt.legend(prop={"size": 12, "family": "Helvetica"}, frameon=False)
    plt.savefig(
        os.path.join(
            invLapW.par.plotpath,
            "RESULT{:2.2e}".format(invLapW.par.sigma) + ".png",
        ),
        dpi=400,
    )
    return
