import sys

sys.path.append("..")
from importall import *

eNorm = False

#TODO at present: float implemented, mp not implemented


def init_variables(args_):
    in_ = Inputs()
    in_.tmax = args_.tmax
    in_.periodicity = args_.periodicity
    in_.prec = args_.prec
    in_.datapath = args_.datapath
    in_.outdir = args_.outdir
    in_.massNorm = args_.mpi
    in_.num_boot = args_.nboot
    in_.sigma = args_.sigma
    in_.emax = args_.emax * args_.mpi   #   we pass it in unit of Mpi, here to turn it into lattice (working) units
    if args_.emin == 0:
        in_.emin = args_.mpi / 20   #TODO get this to be input in lattice units for consistence
    else:
        in_.emin = args_.emin
    in_.e0 = args_.e0
    in_.Ne = args_.ne
    in_.alpha = args_.alpha
    return in_


def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentRhoFromData()
    init_precision(args.prec)
    par = init_variables(args)

    #   Reading datafile, storing correlator
    rawcorr, par.time_extent, par.num_samples = u.read_datafile(par.datapath)
    par.assign_values()
    par.report()
    tmax = par.tmax
    adjust_precision(par.tmax)
    #   Here is the correlator
    rawcorr.evaluate()

    #   Here is the resampling
    corr = u.Obs(T = par.time_extent, tmax = par.tmax, nms = par.num_boot, is_resampled=True)
    resample = ParallelBootstrapLoop(par, rawcorr.sample)
    corr.sample = resample.run()
    corr.evaluate()

    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)

    #   make it into a mp sample
    print(LogMessage(), "Converting correlator into mpmath type")
    #mpcorr_sample = mp.matrix(par.num_boot, tmax)
    corr.fill_mp_sample()
    cNorm = mpf(str(corr.central[1] ** 2))

    #   Prepare
    S = Smatrix_mp(tmax, type=par.periodicity, T=par.time_extent)
    hltParams = AlgorithmParameters(alphaA=0, alphaB=-1, alphaC=0, lambdaMax=12, lambdaStep=0.5, lambdaScanPrec = 0.5, lambdaScanCap=6, kfactor = 0.1)
    matrix_bundle = MatrixBundle(Smatrix=S, Bmatrix=corr.mpcov, bnorm=cNorm)
    #   Wrapper for the Inverse Problem
    HLT = HLTWrapper(par=par, algorithmPar=hltParams, matrix_bundle=matrix_bundle, correlator=corr)
    HLT.prepareHLT()

    #   Energy
    estar = HLT.espace[3]

    rho_l_a1, drho_l_a1, gag_l_a1, rho_l_a2, drho_l_a2, gag_l_a2= HLT.scanLambdaAlpha(estar)

    _ = HLT.estimate_sys_error(estar)

    assert(HLT.result_is_filled[3] == True)
    print(LogMessage(), 'rho, drho, sys', HLT.rho_result[3], HLT.drho_result[3],HLT.rho_sys_err[3])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    plt.title(r"$E/M_{\pi}$" + "= {:2.2f}  ".format(estar / HLT.par.massNorm) + r" $\sigma$" + " = {:2.2f} Mpi".format(
        HLT.par.sigma / HLT.par.massNorm))
    ax[0].errorbar(
        x=HLT.lambda_list,
        y=HLT.rho_list,
        yerr=HLT.drho_list,
        marker=plot_markers[0],
        markersize=1.8,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=r"$\alpha = {:1.2f}$".format(hltParams.alphaA),
        color=CB_colors[0],
    )
    ax[0].errorbar(
        x=HLT.lambda_list,
        y=HLT.rho_list_alpha2,
        yerr=HLT.drho_list_alpha2,
        marker=plot_markers[1],
        markersize=3.8,
        elinewidth=1.3,
        capsize=3,
        ls="",
        label=r"$\alpha = {:1.2f}$".format(hltParams.alphaB),
        color=CB_colors[1],
    )
    ax[0].set_xlabel(r"$\lambda$", fontdict=timesfont)
    ax[0].set_ylabel(r"$\rho_\sigma$", fontdict=timesfont)
    ax[0].legend(prop={"size": 12, "family": "Helvetica"})
    ax[0].grid()

    # Second subplot with A/A_0
    ax[1].errorbar(
        x=gag_l_a1,
        y=HLT.rho_list,
        yerr=HLT.drho_list,
        marker=plot_markers[0],
        markersize=1.8,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=r"$\alpha = {:1.2f}$".format(hltParams.alphaA),
        color=CB_colors[0],
    )
    ax[1].set_xlabel(r"$A[g_\lambda] / A_0$", fontdict=timesfont)
    ax[1].set_ylabel(r"$\rho_\sigma$", fontdict=timesfont)
    ax[1].legend(prop={"size": 12, "family": "Helvetica"})
    ax[1].grid()

    plt.tight_layout()
    plt.show()



















    end()

    plt.errorbar(
        x=gag_l_a1,
        y=rho_l_a1,
        yerr=drho_l_a1,
        marker="o",
        markersize=1.5,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=r'$\rho(E_*)$'+r'$(\sigma = {:2.2f})$'.format(par.sigma / par.massNorm)+r'$M_\pi$',
        color=u.CB_color_cycle[0],
    )
    plt.xlabel(r"$A[g_\lambda] / A_0$", fontdict=u.timesfont)
    #plt.ylabel("Spectral density", fontdict=u.timesfont)
    plt.legend(prop={"size": 12, "family": "Helvetica"})
    plt.grid()
    plt.tight_layout()

    plt.errorbar(
        x=gag_l_a2,
        y=rho_l_a2,
        yerr=drho_l_a2,
        marker="^",
        markersize=1.5,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=r'$\rho(E_*)$'+r'$(\sigma = {:2.2f})$'.format(par.sigma / par.massNorm)+r'$M_\pi$',
        color=u.CB_color_cycle[1],
    )
    plt.xlabel(r"$A[g_\lambda] / A_0$", fontdict=u.timesfont)
    #plt.ylabel("Spectral density", fontdict=u.timesfont)
    plt.legend(prop={"size": 12, "family": "Helvetica"})
    plt.grid()
    plt.tight_layout()
    plt.show()
    end()



if __name__ == "__main__":
    main()