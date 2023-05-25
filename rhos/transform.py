from mpmath import mp, mpf
from progressbar import ProgressBar
import sys
from core import *
sys.path.append("../utils")
from rhoUtils import LogMessage
from rhoStat import *


def h_Et_mp(
    Tinv_, params, espacemp_
):  #     h(t,E) = sum_{r=0}    Tinv[t][r] * b[r+1,E]
    ht_ = mp.matrix(
        params.Ne, params.tmax
    )  #     r+1 only in b, since T is already shifted
    for e in range(params.Ne):
        for i in range(params.tmax):
            ht_[e, i] = 0
            for j in range(params.tmax):
                aux_ = mp.fmul(
                    Tinv_[j, i],
                    ft_mp(
                        espacemp_[e],
                        mpf(j + 1),
                        params.mpsigma,
                        params.mpalpha,
                        params.mpemin,
                    ),
                )
                ht_[e, i] = mp.fadd(aux_, ht_[e, i])
    return ht_


def h_Et_mp_Eslice(Tinv_, params, estar_):
    tmax = params.tmax
    ht_ = mp.matrix(params.tmax, 1)
    for i in range(params.tmax):
        ht_[i] = 0
        for j in range(params.tmax):
            aux_ = mp.fmul(
                Tinv_[j, i],
                ft_mp(
                    tmax, estar_, mpf(j + 1), params.mpsigma, params.mpalpha, params.mpemin
                ),
            )
            ht_[i] = mp.fadd(aux_, ht_[i])
    return ht_

def y_combine_central_mp(ht_, corr_, params):
    rho = mp.matrix(params.Ne, 1)
    for e in range(params.Ne):
        rho[e] = 0
        for i in range(params.tmax):
            aux_ = mp.fmul(ht_[e, i], corr_[i])
            rho[e] = mp.fadd(rho[e], aux_)
    return rho


def y_combine_sample_mp(ht_, corrtype_, params):
    pbar = ProgressBar()
    rhob = mp.matrix(params.Ne, params.num_boot)
    for b in pbar(range(params.num_boot)):
        y = corrtype_.sample[b][:]
        for e in range(params.Ne):
            rhob[e, b] = 0
            for i in range(params.tmax):
                aux_ = mp.fmul(ht_[e, i], y[i])
                rhob[e, b] = mp.fadd(rhob[e, b], aux_)
    return averageVector_mp(rhob)

def y_combine_sample_Eslice_mp(ht_sliced, mpmatrix, params):
    rhob = mp.matrix(params.num_boot,1)
    for b in range(params.num_boot):
        y = mpmatrix[b,:]
        rhob[b] = 0
        for i in range(params.tmax):
            aux_ = mp.fmul(ht_sliced[i], y[i])
            rhob[b] = mp.fadd(rhob[b], aux_)
    print(LogMessage(), "rho[e] +/- stat ", float(averageScalar_mp(rhob)[0]), (float(averageScalar_mp(rhob)[1])))
    return averageScalar_mp(rhob)


def getRho_dynamicL_samples(Smat, CovD, bnorm, lpar, corr_sample, estar, params):
    Nb = params.num_boot
    tmax = params.tmax
    mpll = mpf(str(lpar))

    a0 = A0_mp(estar, params.mpsigma, alpha=params.alpha, emin=params.emin)
    scale = mp.fmul(a0, mpll)
    scale = mp.fdiv(scale, bnorm)

    W = scale * CovD
    W = W + Smat
    invW = W ** (-1)

    gtE = h_Et_mp_Eslice(invW, params, estar)
    rhoEb = mp.matrix(Nb, 1)
    for b in range(Nb):
        for t in range(tmax):
            aux_ = mp.fmul(gtE[t], corr_sample[b, t + 1])
            rhoEb[b] = mp.fadd(rhoEb[b], aux_)
    rhoE = averageScalar_mp(rhoEb)
    return rhoE[0], rhoE[1]
