from mpmath import mp, mpf
from progressbar import ProgressBar
import sys

sys.path.append("../utils")


def Smatrix_sigma_mp(tmax_, sigma_):    # for gaussian processes once implemented
    exit(0)
    S_ = mp.matrix(tmax_, tmax_)
    for i in range(tmax_):
        for j in range(tmax_):
            entry = mp.quad(
                lambda x: b_times_exp_mp(x, i + 1, j + 1, sigma_),
                [0, mp.inf],
                error=True,
            )
            S_[i, j] = entry[0]
    return S_


def Smatrix_mp(tmax_: int, alpha_=mpf(0), emin_=mpf(0)):
    S_ = mp.matrix(tmax_, tmax_)
    for i in range(tmax_):
        for j in range(tmax_):
            entry = mp.fadd(mpf(i), mpf(j))
            arg = mp.fadd(entry, mpf(2))  # i+j+2
            entry = mp.fsub(arg, alpha_)    # i+j+2-a
            arg = mp.fneg(arg)
            arg = mp.fmul(arg, emin_)
            arg = mp.exp(arg)
            entry = mp.fdiv(arg, entry)
            S_[i, j] = entry
    return S_


def Zfact_mp(estar_, sigma_):  # int_0^inf dE exp{(-e-estar)^2/2s^2}
    fact_ = mp.sqrt(2)
    res_ = mp.fdiv(estar_, fact_)
    res_ = mp.fdiv(res_, sigma_)  #   e/(sqrt2 sigma)
    res_ = mp.erf(res_)  #   erf [e/(sqrt2 sigma)]
    res_ = mp.fadd(res_, 1)  #   1 + erf [e/(sqrt2 sigma)]
    res_ = mp.fmul(res_, sigma_)  #   sigma (1 + erf [e/(sqrt2 sigma)])
    fact_ = mp.fdiv(mp.pi, 2)
    fact_ = mp.sqrt(fact_)  # sqrt(pi/2)
    res_ = mp.fmul(res_, fact_)  # sqrt(pi/2) sigma (1 + erf [e/(sqrt2 sigma)])
    return res_


def ft_mp(e, t, sigma_, alpha=mpf("0"), emin=mpf("0")):
    newt = mp.fsub(t, alpha)  #
    aux = mp.fmul(sigma_, sigma_)  #   s^2
    arg = mp.fmul(aux, newt)  #   s^2 (t-alpha)
    aux = mp.fmul(arg, newt)  #   s^2 (alpha-t)^2
    aux = mp.fmul(aux, mpf("0.5"))  #   s^2 (alpha-t)^2 /2
    res = mp.exp(aux)  #   exp{s^2 (alpha-t)^2 /2}
    aux = mp.fneg(newt)  #   alpha-t
    aux = mp.fmul(e, aux)  #   e(alpha-t)
    aux = mp.exp(aux)
    res = mp.fmul(res, aux)  #   exp{s^2 (alpha-t)^2 /2} exp{estar (alpha-t) }
    arg = mp.fadd(arg, emin)
    arg = mp.fsub(arg, e)
    arg = mp.fdiv(arg, sigma_)
    aux = mp.sqrt(2)
    arg = mp.fdiv(arg, aux)
    arg = mp.erfc(arg)  #   this is the COMPLEMENTARY erf
    res = mp.fmul(res, arg)
    aux = mp.fdiv(e, aux)
    aux = mp.fdiv(aux, sigma_)
    aux = mp.erf(aux)
    aux = mp.fadd(mpf(1), aux)
    res = mp.fdiv(res, aux)
    return res


def A0_mp(e_, sigma_, alpha=mpf(0), emin=mpf(0)):
    aux = mp.fmul(sigma_, sigma_)
    aux = mp.fdiv(aux, mpf(2))
    aux = mp.fmul(aux, alpha)
    aux = mp.fadd(e_, aux)
    aux = mp.fsub(aux, emin)
    res = mp.fdiv(aux, sigma_)
    res = mp.erf(res)  #   Erf
    res = mp.fadd(1, res)  # 1+erf, the numerator
    aux_ = mp.sqrt(mp.pi)
    res = mp.fdiv(res, aux_)  # 1+erf /pi
    res = mp.fdiv(res, sigma_)  # 1+erf / (sqrt{pi} s)
    aux_ = mp.sqrt(2)
    aux_ = mp.fdiv(e_, aux_)
    aux_ = mp.fdiv(aux_, sigma_)
    aux_ = mp.erf(aux_)
    aux_ = mp.fadd(aux_, 1)
    aux_ = mp.fmul(aux_, aux_)
    res = mp.fdiv(res, aux_)
    return res

def A0E_mp(espacemp_, par):   #   vector of A0s for each energy
    a0_e = mp.matrix(par.Ne, 1)
    for ei in range(par.Ne):
        a0_e[ei] = A0_mp(e_=espacemp_[ei], sigma_=par.mpsigma, alpha=par.mpalpha, emin=par.mpemin)
    return a0_e
