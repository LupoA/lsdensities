from mpmath import mp, mpf
from progressbar import ProgressBar
import sys
import numpy as np
sys.path.append("../utils")
from rhoUtils import LogMessage
from transform import *
from core import *

#   Functionals A, B and W

def gAg(smat, gt, estar, params):
    tmax = params.tmax
    term1 = mpf(0)
    vt_ = mp.matrix(tmax,1)
    for t in range(tmax):
        vt_[t]=0
        for r in range(tmax):
            aux_  = mp.fmul(gt[r],smat[t,r])
            vt_[t] = mp.fadd(vt_[t], aux_)
        aux_ = mp.fmul(gt[t], vt_[t])
        term1 = mp.fadd(term1, aux_)
    #   term 1 = g^T S g

    #print(LogMessage(), "Debug ::: ", 'A term 1  ', term1)
    term2 = A0_mp(e_=estar, sigma_=params.mpsigma, alpha=params.mpalpha, emin=params.mpemin)
    #   term 2 = A0

    #print(LogMessage(), "Debug ::: ", 'A term 2  ', term2)
    term3 = mpf(0)
    for t in range(tmax):
        aux_ = mp.fmul(ft_mp(e=estar, t=mpf(t+1), sigma_=params.mpsigma, alpha=params.mpalpha, emin=params.mpemin), gt[t])
        term3 = mp.fadd(term3, aux_)
    term3 = mp.fmul(mpf(2), term3)
    #   term 3 = 2 sum_t f_t g_t

    #print(LogMessage(), "Debug ::: ", 'A term 3  ', term3)
    res = mp.fsub(term1, term3)
    res = mp.fadd(res, term2)
    return res  #    term1 + term2 - term3


def gAgA0(smat, gt, estar, params, a0):
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
        term1 = mp.fdiv(term1, a0)
    #   term 1 = g^T S g / A0

    term2 = mpf(1)
    #   term 2 = A0 / A0

    term3 = 0
    for t in range(tmax):
        aux_ = mp.fmul(ft_mp(estar, mpf(t+1), params.mpsigma, params.mpalpha, params.mpemin), gt[t])
        term3 = mp.fadd(term3, aux_)
    term3 = mp.fmul(mpf(2), term3)
    term3 = mp.fdiv(term3,a0)
    #   term 3 = 2 sum_t f_t g_t / A0

    res = mp.fsub(term1, term3)
    res = mp.fadd(res, term2)
    return res  # term1 + term2 - term3

def gBg(gt, bmat, bnorm):
    tmax = bmat.rows
    res = mpf(0)
    vt_ = mp.matrix(tmax,1)
    for t in range(tmax):
        vt_[t]=0
        for r in range(tmax):
            aux_ = mp.fmul(gt[r], bmat[r,t])
            vt_[t] = mp.fadd(aux_, vt_[t])
        aux_ = mp.fmul(gt[t], vt_[t])
        res = mp.fadd(res,aux_)
    res = mp.fdiv(res, bnorm)
    #   res = g B g
    return res

def gWg(smat,cov,gt,estar,mplambda, a0_,bnorm, params, verbose=False):
    aterm = gAg(smat,gt,estar,params)
    #print(LogMessage(), "Debug ::: ", 'A = ', aterm)
    #print(LogMessage(), "Debug ::: ", 'A0 = ', a0_)
    aterm = mp.fdiv(aterm, a0_)
    if verbose==True:
        print(LogMessage(), 'ABW :::', 'Energy {:2.3f}'.format(float(estar)), 'Lambda {:2.3f}'.format(float(mplambda)), 'A/A0 = ', float(aterm))
    bterm = gBg(gt,cov,bnorm)
    if verbose==True:
        print(LogMessage(), 'ABW :::', 'Energy {:2.3f}'.format(float(estar)), 'Lambda {:2.3f}'.format(float(mplambda)), 'B/Bnorm = ', float(bterm))
    scale = mp.fsub(mpf(1),mplambda)
    wterm = mp.fmul(aterm, scale)   #   (1-l) A/A0
    bterm = mp.fmul(mplambda,bterm) #   l B/Bnorm
    wterm = mp.fadd(wterm, bterm)    #  ( 1-l) A/A0 + l B/Bnorm
    return wterm

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
    return Wvec
        #plt.plot(lset, Wvec, marker='^', ls='')
        #plt.show()

def getLstar(espace_mp, Smat, CovDmat, csq, params, eNorm=False, lambda_min=0.01, lambda_max=0.6, num_lambda=20):
    tmax = params.tmax
    lset = np.linspace(lambda_min, lambda_max, num_lambda)
    Wvec = np.zeros(num_lambda)
    for ei in range(params.Ne):
        estar = espace_mp[ei]
        lstar_ID = 0
        Wstar = -1
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
            if Wvec[li] > Wstar:
                Wstar = Wvec[li]
                lstar_ID = li
    return lset[lstar_ID]

def getLstar_Eslice(estar, Smat, CovDmat, csq, params, eNorm_=False, lambda_min=0.01, lambda_max=0.6, num_lambda=20):
    tmax = params.tmax
    lset = np.linspace(lambda_min, lambda_max, num_lambda)
    Wvec = np.zeros(num_lambda)

    lstar_ID = 0
    Wstar = -1
    if eNorm_==False:
        Bnorm = csq
    if eNorm_==True:
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
        if Wvec[li] > Wstar:
            Wstar = Wvec[li]
            lstar_ID = li
    print(LogMessage(), 'Lambda ::: ', "Lambda* at E = ", float(estar), ' ::: ', lset[lstar_ID])
    return lset[lstar_ID]   #l*