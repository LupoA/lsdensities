import lsdensities.utils.rhoUtils as u
from lsdensities.utils.rhoUtils import (
    init_precision,
    LogMessage,
    end,
    Inputs,
    create_out_paths,
    plot_markers,
    CB_colors,
)
from lsdensities.utils.rhoParser import parseArgumentPrintSamples
from lsdensities.correlator.correlatorUtils import symmetrisePeriodicCorrelator
from lsdensities.utils.rhoParallelUtils import ParallelBootstrapLoop
from mpmath import mp, mpf
from lsdensities.core import a0_array, hlt_matrix
from lsdensities.abw import gAg, gBg
from lsdensities.transform import get_ssd_scalar, y_combine_sample_Eslice_mp_ToFile
import os
import time
from lsdensities.utils.rhoMath import invert_matrix_ge
import matplotlib.pyplot as plt


def init_variables(args_):
    in_ = Inputs()
    in_.tmax = args_.tmax
    in_.periodicity = args_.periodicity
    in_.kerneltype = args_.kerneltype
    in_.prec = args_.prec
    in_.datapath = args_.datapath
    in_.outdir = args_.outdir
    in_.massNorm = args_.mpi
    in_.num_boot = args_.nboot
    in_.sigma = args_.sigma
    in_.emax = (
        args_.emax * args_.mpi
    )  #   we pass it in unit of Mpi, here to turn it into lattice (working) units
    if args_.emin == 0:
        in_.emin = (args_.mpi / 20) * args_.mpi
    else:
        in_.emin = args_.emin * args_.mpi
    in_.e0 = args_.e0
    in_.Na = args_.Na
    in_.A0cut = args_.A0cut
    return in_


def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentPrintSamples()
    init_precision(args.prec)
    par = init_variables(args)

    #   Reading datafile, storing correlator
    rawcorr, par.time_extent, par.num_samples = u.read_datafile(par.datapath)

    #   Take input for Rho
    import numpy as np

    rho_file = args.rhopath
    inputrhofile = np.genfromtxt(rho_file, comments="#")
    energy = inputrhofile[:, 0]
    lambda_e = inputrhofile[:, 1]
    in_rho = inputrhofile[:, 2]
    in_stat = inputrhofile[:, 3]
    inputrhofile[:, 4]
    inputrhofile[:, 5]
    par.Ne = len(energy)
    espace = np.zeros(par.Ne)
    rho = np.zeros(par.Ne)
    drho = np.zeros(par.Ne)
    np.zeros(par.Ne)
    for _e in range(par.Ne):
        espace[_e] = energy[_e]

    #
    par.assign_values()
    par.report()
    par.plotpath, par.logpath = create_out_paths(par)

    #   Here is the correlator
    rawcorr.evaluate()
    rawcorr.tmax = par.tmax

    #   Symmetrise
    if par.periodicity == "COSH":
        print(LogMessage(), "Folding correlator")
        symCorr = symmetrisePeriodicCorrelator(corr=rawcorr, par=par)
        symCorr.evaluate()

    #   Here is the resampling
    if par.periodicity == "EXP":
        corr = u.Obs(
            T=par.time_extent, tmax=par.tmax, nms=par.num_boot, is_resampled=True
        )
    if par.periodicity == "COSH":
        corr = u.Obs(
            T=symCorr.T,
            tmax=symCorr.tmax,
            nms=par.num_boot,
            is_resampled=True,
        )

    if par.periodicity == "COSH":
        # resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)
        resample = ParallelBootstrapLoop(par, symCorr.sample, is_folded=False)
    if par.periodicity == "EXP":
        resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)

    corr.sample = resample.run()
    corr.evaluate()

    #   Covariance
    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=True)
    corr.corrmat_from_covmat(plot=True)

    #   Make it into a mp sample
    print(LogMessage(), "Converting correlator into mpmath type")
    corr.fill_mp_sample()
    print(LogMessage(), "Cond[Cov C] = {:3.3e}".format(float(mp.cond(corr.mpcov))))
    cNorm = mpf(str(corr.central[1] ** 2))

    # from HLT class
    A0set = a0_array(espace, par, alpha=0)

    for _e in range(par.Ne):
        estar_ = espace[_e]
        fname = "lsdensitiesamplesE" + str(estar_) + "sig" + str(par.sigma)
        fpath = os.path.join(par.logpath, fname)
        _Bnorm = cNorm / (estar_ * estar_)
        _factor = (lambda_e[_e] * A0set[_e]) / _Bnorm
        S_ = hlt_matrix(
            tmax=par.tmax,
            alpha=0,
            e0=par.mpe0,
            type=par.periodicity,
            T=par.time_extent,
        )
        _M = S_ + (_factor * corr.mpcov)
        start_time = time.time()
        _Minv = invert_matrix_ge(_M)
        end_time = time.time()
        print(
            LogMessage(),
            "\t \t lambdaToRho ::: Matrix inverted in {:4.4f}".format(
                end_time - start_time
            ),
            "s",
        )
        _g_t_estar = get_ssd_scalar(_Minv, par, estar_, alpha=0)
        rho[_e], drho[_e] = y_combine_sample_Eslice_mp_ToFile(
            fpath, _g_t_estar, corr.mpsample, par
        )

        gag_estar = gAg(S_, _g_t_estar, estar_, 0, par)

        gBg_estar = gBg(_g_t_estar, corr.mpcov, _Bnorm)

        print(LogMessage(), "\t \t  B / Bnorm = ", float(gBg_estar))
        print(LogMessage(), "\t \t  A / A0 = ", float(gag_estar / A0set[_e]))

    plt.errorbar(
        x=espace,
        y=rho,
        yerr=drho,
        marker=plot_markers[0],
        markersize=2,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=r"Recomputed",
        color=CB_colors[0],
    )
    plt.errorbar(
        x=espace,
        y=in_rho,
        yerr=in_stat,
        marker=plot_markers[1],
        markersize=2,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=r"Input",
        color=CB_colors[1],
    )
    plt.title(r" $\sigma$" + " = {:2.2f} Mpi".format(par.sigma))
    plt.xlabel(r"$E / M_{\pi}$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    end()


if __name__ == "__main__":
    main()
