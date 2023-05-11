import sys
sys.path.append("..")
from importall import *


out_path = "."
e_norm = 0
Mpi = 5*0.066
eNorm = False

def init_variables(args_):
    in_ = u.inputs()
    in_.prec = args_.prec
    in_.time_extent = args_.T
    in_.tmax = args_.T - 1
    in_.outdir = args_.outdir
    in_.massNorm = Mpi
    in_.num_samples = args_.nms
    in_.num_boot = args_.nboot
    in_.sigma = args_.sigma
    in_.emax = args_.emax * Mpi
    in_.Ne = args_.ne
    in_.alpha = args_.alpha
    in_.emin = args_.emin
    in_.prec = -1
    in_.plots = args_.plots
    in_.assign_values()
    return in_

def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentPeak()
    init_precision(args.prec)
    par = init_variables(args)
    espace = np.linspace(0.1, par.emax, par.Ne)
    espace_mp = mp.matrix(par.Ne, 1)
    for e_id in range(par.Ne):
        espace_mp[e_id] = mpf(str(espace[e_id]))
    par.report()
    T = par.time_extent
    tmax = par.tmax
    nms = par.num_samples
    Nb = par.num_boot

    #   one peak vector
    op = np.zeros(par.time_extent)
    for t in range(par.time_extent):
        op[t] = np.exp(-(t) * Mpi)
    cov = np.zeros((T, T))
    for i in range(T):
        cov[i, i] = (op[i] * 0.02) ** 2
    #   make it a sample and bootstrap it
    corr = u.obs(T, Nb, is_resampled=True)
    measurements = np.random.multivariate_normal(op, cov, nms)
    corr.sample = bootstrap_compact_fp(par, measurements)
    print(LogMessage(), "Correlator resampled")
    corr.evaluate()
    #   make it into a mp sample
    mpcorr_sample = mp.matrix(Nb, tmax)
    for n in range(Nb):
        for i in range(tmax):
            mpcorr_sample[n, i] = mpf(str(corr.sample[n][i+1]))
    #   Get S matrix
    S = Smatrix_mp(tmax)
    invS = S ** (-1)
    diff = S * invS
    diff = norm2_mp(diff) - 1
    print(LogMessage(), "S/S - 1 ::: ", diff)
    #   Get cov for B matrix
    mpcov = mp.matrix(tmax)
    for i in range(tmax):
        mpcov[i, i] = mpf(str(cov[i + 1][i + 1]))
    cNorm = mpf(str(op[0] ** 2))

    #   Get rho
    rho = mp.matrix(par.Ne,1)
    drho = mp.matrix(par.Ne, 1)
    #   Preparatory functions
    a0_e = A0E_mp(espace_mp, par)
    for e_i in range(par.Ne):
        estar = espace_mp[e_i]
        #   get lstar
        lstar_fp = getLstar_Eslice(estar, S, a0_e[e_i], mpcov, cNorm, par, eNorm_=False, lambda_min=0.01, lambda_max=0.6, num_lambda=20)
        scale_fp = lstar_fp / (1-lstar_fp)
        scale_mp = mpf(scale_fp)
        scale_mp = mp.fmul(scale_mp, a0_e[e_i])
        if eNorm == False:
            Bnorm = cNorm
        if eNorm == True:
            Bnorm = mp.fdiv(cNorm, estar)
            Bnorm = mp.fdiv(Bnorm, estar)
        scale_mp = mp.fdiv(scale_mp, Bnorm)
        T = mpcov*scale_mp
        T = T + S
        invT = T**(-1)
        #   Get coefficients
        gt = h_Et_mp_Eslice(invT, par, estar_=estar)
        rhoE = y_combine_sample_Eslice_mp(gt, mpmatrix=mpcorr_sample, params=par)
        rho[e_i] = rhoE[0]
        drho[e_i] = rhoE[1]

    plt.errorbar(x=espace / Mpi, y=rho, yerr=drho, marker="o", markersize=1.5, elinewidth=1.3, capsize=2,
                 ls='', label='HLT (sigma = {:2.2f} Mpi)'.format(par.sigma / Mpi), color=u.CB_color_cycle[0])
    plt.plot(espace/Mpi, gauss_fp(espace, Mpi , par.sigma, norm='half'), color=u.CB_color_cycle[2], linewidth=1, ls='--', label='Target')
    plt.xlabel('Energy/Mpi', fontdict=u.timesfont)
    plt.ylabel('Spectral density', fontdict=u.timesfont)
    plt.legend(prop={'size': 12, 'family': 'Helvetica'})
    plt.grid()
    plt.tight_layout()
    plt.show()

    return 0

if __name__ == "__main__":
    main()
