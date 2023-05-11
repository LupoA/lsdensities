import sys
sys.path.append("..")
from importall import *

eNorm = False

# NOT YET TESTED


def init_variables(args_):
    in_ = u.inputs()
    in_.prec = args_.prec
    in_.datapath = args_.datapath
    in_.outdir = args_.outdir
    in_.massNorm = args_.mpi
    in_.num_boot = args_.nboot
    in_.sigma = args_.sigma
    in_.emax = args_.emax
    in_.Ne = args_.ne
    in_.alpha = args_.alpha
    in_.emin = args_.emin
    in_.prec = -1
    in_.plots = args_.plots
    return in_


def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentRhoFromData()
    init_precision(args.prec)
    par = init_variables(args)
    espace = np.linspace(0.1, par.emax, par.Ne)
    espace_mp = mp.matrix(par.Ne, 1)
    for e_id in range(par.Ne):
        espace_mp[e_id] = mpf(str(espace[e_id]))

    #   Reading datafile, storing correlator
    rawcorr, par.time_extent, par.num_samples = u.read_datafile(par.datapath)
    par.assign_values()
    par.report()

    #   Here is the correlator
    rawcorr.evaluate()

    #   here is the resampling
    corr = u.obs(par.time_extent, par.num_boot, is_resampled=True)
    resample = parallel_bootstrap_loop(par, rawcorr.sample)
    corr.sample = resample.run()
    corr.evaluate()
    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    # corr.eval_corrmatrix(plot=True)

    from correlatorUtils import effective_mass
    effmass = effective_mass(corr, par, type='EXP')
    effmass.plot(logscale=False)
    exit(1)

    tmax = 56  #   soon to be parsed from command line it is the latest correlator we use
    par.tmax = tmax

    #   make it into a mp sample
    mpcorr_sample = mp.matrix(par.num_boot, tmax)
    for n in range(par.num_boot):
        for i in range(tmax):
            mpcorr_sample[n, i] = mpf(str(corr.sample[n][i + 1]))
    #   Get cov for B matrix
    mpcov = mp.matrix(tmax)
    for i in range(tmax):
        mpcov[i, i] = mpf(str(corr.cov[i + 1][i + 1]))
    cNorm = mpf(str(corr.central[1] ** 2))

    #   Get S matrix
    S = Smatrix_mp(tmax)
    invS = S ** (-1)
    diff = S * invS
    diff = norm2_mp(diff) - 1
    print(LogMessage(), "S/S - 1 ::: ", diff)

    #   Get rho
    rho = mp.matrix(par.Ne, 1)
    drho = mp.matrix(par.Ne, 1)
    #   Preparatory functions
    a0_e = A0E_mp(espace_mp, par)

    for e_i in range(par.Ne):
        estar = espace_mp[e_i]
        #   get lstar
        lstar_fp = getLstar_Eslice(
            estar,
            S,
            a0_e[e_i],
            mpcov,
            cNorm,
            par,
            eNorm_=False,
            lambda_min=0.01,
            lambda_max=0.6,
            num_lambda=20,
        )
        scale_fp = lstar_fp / (1 - lstar_fp)
        scale_mp = mpf(scale_fp)
        #a0 = A0_mp(e_=estar, sigma_=par.mpsigma, alpha=par.mpalpha, emin=par.mpemin)
        scale_mp = mp.fmul(scale_mp, a0_e[e_i])
        if eNorm == False:
            Bnorm = cNorm
        if eNorm == True:
            Bnorm = mp.fdiv(cNorm, estar)
            Bnorm = mp.fdiv(Bnorm, estar)
        scale_mp = mp.fdiv(scale_mp, Bnorm)
        T = mpcov * scale_mp
        T = T + S
        invT = T ** (-1)
        #   Get coefficients
        gt = h_Et_mp_Eslice(invT, par, estar_=estar)
        rhoE = y_combine_sample_Eslice_mp(gt, mpmatrix=mpcorr_sample, params=par)
        rho[e_i] = rhoE[0]
        drho[e_i] = rhoE[1]

    plt.errorbar(
        x=espace / par.massNorm,
        y=rho,
        yerr=drho,
        marker="o",
        markersize=1.5,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label="HLT (sigma = {:2.2f} Mpi)".format(par.sigma / par.massNorm),
        color=u.CB_color_cycle[0],
    )
    plt.xlabel("Energy/Mpi", fontdict=u.timesfont)
    plt.ylabel("Spectral density", fontdict=u.timesfont)
    plt.legend(prop={"size": 12, "family": "Helvetica"})
    plt.grid()
    plt.tight_layout()
    plt.show()

    end()


if __name__ == "__main__":
    main()
