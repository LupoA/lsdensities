import sys

sys.path.append("..")
from importall import *

Mpi = 0.066
Mpi_mp = mpf("0.066")
out_path = "."
e_norm = 0
state_at = 5


def init_variables(args_):
    in_ = u.inputs()
    in_.prec = args_.prec
    in_.time_extent = args_.T
    in_.tmax = args_.T - 1
    in_.outdir = args_.outdir
    in_.num_samples = args_.nms
    in_.num_boot = args_.nboot
    in_.sigma = args_.sigma
    in_.emax = args_.emax * Mpi
    in_.Ne = args_.ne
    # in_.l = args_.l
    in_.alpha = args_.alpha
    in_.emin = args_.emin
    in_.prec = -1
    in_.plots = args_.plots
    in_.assign_mp_values()
    return in_


def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentPeak()
    init_precision(args.prec)
    par = init_variables(args)
    par.assign_mp_values()
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
        op[t] = np.exp(-(t) * 5 * Mpi)
    cov = np.zeros((T, T))
    for i in range(T):
        cov[i, i] = (op[i] * 0.02) ** 2
    #   make it a sample and bootstrap it
    corr = u.obs(T, Nb)
    measurements = np.random.multivariate_normal(op, cov, nms)
    corr.sample = bootstrap_compact_fp(par, measurements)
    print(LogMessage(), "Correlator resampled")
    corr.evaluate(resampled=True)
    #   make it into a mp sample
    mpcorr_sample = mp.matrix(Nb, T)
    for n in range(Nb):
        for i in range(T):
            mpcorr_sample[n, i] = mpf(str(corr.sample[n][i]))
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
    c1sq = mpf(str(op[1] ** 2))
    #   Check W
    getW(espace_mp, S, mpcov, c1sq, par, eNorm=False)

    return 0


def getW(espace_mp, Smat, CovDmat, csq, params, eNorm=False):
    tmax = params.tmax
    lset = np.linspace(0.01, 0.6, 20)
    num_lambda = len(lset)
    Wvec = np.zeros(num_lambda)
    for ei in range(params.Ne):
        estar = espace_mp[ei]
        lstar_ID = 0
        if eNorm==False:
            Bnorm = csq
        if eNorm==True:
            Bnorm = mp.fdiv(csq, estar)
            Bnorm = mp.fdiv(Bnorm, estar)
        for li in range(num_lambda):
            mp_l = mpf(str(lset[li]))
            scale = mpf(str(lset[li]/(1-lset[li])))
            scale = mp.fdiv(scale, Bnorm)
            a0 = A0_mp(estar, params.mpsigma, alpha=mpf(0), emin=mpf(0))
            scale = mp.fmul(scale, a0)
            W = CovDmat*scale
            W = W + Smat
            invW = W**(-1)
            #   given W, get the coefficient
            gtestar = h_Et_mp_Eslice(invW, params, estar)
            Wvec[li] = float(gWg(Smat, CovDmat, gtestar, estar, mp_l, a0, Bnorm, params, verbose=True))
        plt.plot(lset, Wvec, marker='^', ls='')
        plt.show()

if __name__ == "__main__":
    main()
