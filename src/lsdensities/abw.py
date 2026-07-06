from mpmath import mp, mpf
from .transform import ft_mp
from .core import a0_scalar

#   Functionals A, B and W


def gAg(smat, gt, estar, alpha, params):
    tmax = params.tmax
    term1 = mpf(0)
    vt_ = mp.matrix(tmax, 1)
    for t in range(tmax):
        vt_[t] = 0
        for r in range(tmax):
            aux_ = mp.fmul(gt[r], smat[t, r])
            vt_[t] = mp.fadd(vt_[t], aux_)
        aux_ = mp.fmul(gt[t], vt_[t])
        term1 = mp.fadd(term1, aux_)
    #   term 1 = g^T S g

    term2 = a0_scalar(
        e=estar,
        sigma=params.mpsigma,
        alpha=alpha,
        e0=params.mpe0,
        ker_type=params.kerneltype,
    )
    #   term 2 = A0

    term3 = mpf(0)
    for t in range(tmax):
        aux_ = mp.fmul(
            ft_mp(
                e=estar,
                t=mpf(t + 1),
                sigma_=params.mpsigma,
                alpha=alpha,
                e0=params.mpe0,
                type=params.periodicity,
                T=params.time_extent,
                ker_type=params.kerneltype,
            ),
            gt[t],
        )
        term3 = mp.fadd(term3, aux_)
    term3 = mp.fmul(mpf(2), term3)
    #   term 3 = 2 sum_t f_t g_t

    res = mp.fsub(term1, term3)
    res = mp.fadd(res, term2)
    return res  #    term1 + term2 - term3


def gBg(gt, bmat, bnorm):
    res = mpf(0)
    tmax = bmat.cols
    vt_ = mp.matrix(tmax, 1)
    for t in range(tmax):
        vt_[t] = 0
        for r in range(tmax):
            aux_ = mp.fmul(gt[r], bmat[r, t])
            vt_[t] = mp.fadd(aux_, vt_[t])
        aux_ = mp.fmul(gt[t], vt_[t])
        res = mp.fadd(res, aux_)
    res = mp.fdiv(res, bnorm)
    #   res = g B g / bnorm
    return res

