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
    hltParams = AlgorithmParameters(alphaA=0, alphaB=0, alphaC=0, lambdaMax=20, lambdaStep=0.5, lambdaScanPrec = 0.1, lambdaScanCap=6, kfactor = 0.1)
    matrix_bundle = MatrixBundle(Smatrix=S, Bmatrix=corr.mpcov, bnorm=cNorm)
    #   Wrapper for the Inverse Problem
    HLT = HLTWrapper(par=par, algorithmPar=hltParams, matrix_bundle=matrix_bundle, correlator=corr)
    HLT.prepareHLT()

    #   Energy
    estar = HLT.espace[3]

    rho_l_a1, drho_l_a1, gag_l_a1 = HLT.scanLambda(estar, alpha_=hltParams.alphaA)

    _ = HLT.estimate_sys_error(estar)

    assert(HLT.result_is_filled[3] == True)
    print(LogMessage(), 'rho, drho, sys', HLT.rho_result[3], HLT.drho_result[3],HLT.rho_sys_err[3])

    import matplotlib.pyplot as plt

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
    plt.show()

    end()

if __name__ == "__main__":
    main()