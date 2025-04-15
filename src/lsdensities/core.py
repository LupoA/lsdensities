from mpmath import mp, mpf
from .utils.rhoMath import cauchy
from scipy.integrate import quad
import numpy as np

def hlt_matrix(tmax: int, alpha, e0=mpf(0), type="EXP", T=0):
    S_ = mp.matrix(tmax, tmax)
    for i in range(tmax):
        for j in range(tmax):
            entry = mp.fadd(mpf(i), mpf(j))
            arg = mp.fadd(entry, mpf(2))  # i+j+2
            entry = mp.fsub(arg, alpha)  # i+j+2-a
            arg = mp.fneg(arg)
            arg = mp.fmul(arg, e0)
            arg = mp.exp(arg)
            entry = mp.fdiv(arg, entry)
            S_[i, j] = entry
            if type == "COSH":
                assert T > 0
                entry2 = mp.fsub(mpf(i), mpf(j))
                entry3 = mp.fsub(mpf(j), mpf(i))
                entry4 = mp.fneg(mpf(i))
                entry4 = mp.fsub(entry4, mpf(j))
                arg2 = mp.fadd(entry2, mpf(T))  # T+i-j
                arg3 = mp.fadd(entry3, mpf(T))  # T+j-i
                arg4 = mp.fadd(entry4, mpf(2 * T))  # 2T-j-i
                arg4 = mp.fsub(arg4, mpf(2))  # 2T-j-i-2
                entry2 = mp.fsub(arg2, alpha)  # T+i-j-a
                entry3 = mp.fsub(arg3, alpha)  # T+j-i-a
                entry4 = mp.fsub(arg4, alpha)  # 2T-j-i-2-a
                arg2 = mp.fneg(arg2)
                arg3 = mp.fneg(arg3)
                arg4 = mp.fneg(arg4)
                arg2 = mp.fmul(arg2, e0)
                arg3 = mp.fmul(arg3, e0)
                arg4 = mp.fmul(arg4, e0)
                arg2 = mp.exp(arg2)
                arg3 = mp.exp(arg3)
                arg4 = mp.exp(arg4)
                entry2 = mp.fdiv(arg2, entry2)
                entry3 = mp.fdiv(arg3, entry3)
                entry4 = mp.fdiv(arg4, entry4)
                S_[i, j] += entry2 + entry3 + entry4
    return S_


def generalised_ft(t, alpha, sigma, e, e0):
    newt = mp.fsub(t, alpha)  #
    aux = mp.fmul(sigma, sigma)  #   s^2
    arg = mp.fmul(aux, newt)  #   s^2 (t-alpha)
    aux = mp.fmul(arg, newt)  #   s^2 (alpha-t)^2
    aux = mp.fmul(aux, mpf("0.5"))  #   s^2 (alpha-t)^2 /2
    res = mp.exp(aux)  #   exp{s^2 (alpha-t)^2 /2}
    aux = mp.fneg(newt)  #   alpha-t
    aux = mp.fmul(e, aux)  #   e(alpha-t)
    aux = mp.exp(aux)
    res = mp.fmul(res, aux)  #   exp{s^2 (alpha-t)^2 /2} exp{estar (alpha-t) }
    #   The previous expression will multiply Erfc
    arg = mp.fadd(arg, e0)
    arg = mp.fsub(arg, e)  # s^2 (t-alpha) + E0 - omega
    arg = mp.fdiv(arg, sigma)
    aux = mp.sqrt(2)
    arg = mp.fdiv(arg, aux)  # [s^2 (t-alpha) + E0 - omega] / sqrt{2} s
    arg = mp.erfc(arg)  #   this is the COMPLEMENTARY erf
    res = mp.fmul(res, arg)  # Erfc() * exp()
    return res


def generalised_ft_halfnorm(t, alpha, sigma, e, e0):
    newt = mp.fsub(t, alpha)  #
    aux = mp.fmul(sigma, sigma)  # s^2
    arg = mp.fmul(aux, newt)  # s^2 (t-alpha)
    aux = mp.fmul(arg, newt)  # s^2 (alpha-t)^2
    aux = mp.fmul(aux, mpf("0.5"))  # s^2 (alpha-t)^2 /2
    res = mp.exp(aux)  # exp{s^2 (alpha-t)^2 /2}
    aux = mp.fneg(newt)  # alpha-t
    aux = mp.fmul(e, aux)  # e(alpha-t)
    aux = mp.exp(aux)
    res = mp.fmul(res, aux)  # exp{s^2 (alpha-t)^2 /2} exp{estar (alpha-t) }
    arg = mp.fadd(arg, e0)
    arg = mp.fsub(arg, e)
    arg = mp.fdiv(arg, sigma)
    aux = mp.sqrt(2)
    arg = mp.fdiv(arg, aux)
    arg = mp.erfc(arg)  # this is the COMPLEMENTARY erf
    res = mp.fmul(res, arg)
    aux = mp.fdiv(e, aux)
    aux = mp.fdiv(aux, sigma)
    aux = mp.erf(aux)
    aux = mp.fadd(mpf(1), aux)
    res = mp.fdiv(res, aux)
    return res


def ft_mp(e, t, sigma_, alpha, e0=mpf("0"), type="EXP", T=0, ker_type="FULLNORMGAUSS"):
    if ker_type == "FULLNORMGAUSS":
        res = generalised_ft(t, alpha, sigma_, e, e0)
        if type == "COSH":
            assert T > 0
            pterm = generalised_ft(T - t, alpha, sigma_, e, e0)
            res = mp.fadd(res, pterm)
        res *= 0.5

    elif ker_type == "HALFNORMGAUSS":
        res = generalised_ft_halfnorm(t, alpha, sigma_, e, e0)
        if type == "COSH":
            assert T > 0
            pterm = generalised_ft_halfnorm(T - t, alpha, sigma_, e, e0)
            res = mp.fadd(res, pterm)
    elif ker_type == "CAUCHY":
        # Define the function to be integrated
        def integrand(k):
            aux = mp.exp(alpha * k)
            aux2 = -t * k
            aux2 = mp.exp(aux2)
            if type == "COSH":
                assert T > 0
                aux3 = -(T - t) * k
                aux3 = mp.exp(aux3)
                aux2 = aux2 + aux3
            aux = aux * aux2 * cauchy(k, sigma_, e)
            return aux


        try:
            result, err = quad(integrand, float(e0), np.inf,
                               limit=200, epsabs=1e-10, epsrel=1e-10)
            #print('Error in integral: ', err)
            if not np.isfinite(result) or result < 0 or err > 1e-6:
                raise ValueError("Suspicious result")
            res = result
        except Exception as e:
            print(f"[SciPy fallback] Reason: {e}")
            res = mp.quad(integrand, [e0, mp.inf], method="gauss-legendre")


    else:
        raise ValueError("Invalid smearing kernel (par.ker_type)")
    return res


def a0_scalar(e, sigma, alpha, e0=mpf(0), ker_type="FULLNORMGAUSS"):
    """
    Functional A[g=0], for a given energy.
    """
    if ker_type == "FULLNORMGAUSS":
        aux = mp.fmul(sigma, sigma)  # start by the argument of erf
        aux = mp.fdiv(aux, mpf(2))  # s^2 / 2
        aux = mp.fmul(aux, alpha)  # a s^2 / 2
        aux = mp.fadd(e, aux)  # omega + a s^2 / 2
        aux = mp.fsub(aux, e0)  # omega - E0 + a s^2 / 2
        res = mp.fdiv(aux, sigma)  # omega - E0 + a s^2 / 2sigma
        res = mp.erf(res)  #   Erf
        res = mp.fadd(1, res)  # 1+erf, the numerator
        aux_ = mp.sqrt(mp.pi)
        res = mp.fdiv(res, aux_)  # 1+erf / sqrt{pi}
        res = mp.fdiv(res, sigma)  # 1+erf / (sqrt{pi} s)
        res = mp.fdiv(res, 4)  # 1+erf / (4 sqrt{pi} s)
        # exp term due to alpha
        if alpha != 0:
            aux = mp.fmul(alpha, e)  # alpha*e
            aux2 = mp.fmul(alpha, sigma)  # alpha*sigma
            aux2 = mp.fmul(aux2, aux2)  # (alpha*sigma)^2
            aux2 = mp.fdiv(aux2, mpf(4))  # (alpha*sigma)^2 / 4
            aux = mp.fadd(aux, aux2)  # (alpha*sigma)^2 / 4 + alphaomega
            aux = mp.exp(aux)
            res = mp.fmul(res, aux)
    elif ker_type == "HALFNORMGAUSS":
        aux = mp.fmul(sigma, sigma)
        aux = mp.fdiv(aux, mpf(2))
        aux = mp.fmul(aux, alpha)
        aux = mp.fadd(e, aux)
        aux = mp.fsub(aux, e0)
        res = mp.fdiv(aux, sigma)
        res = mp.erf(res)  #   Erf
        res = mp.fadd(1, res)  # 1+erf, the numerator
        aux_ = mp.sqrt(mp.pi)
        res = mp.fdiv(res, aux_)  # 1+erf /pi
        res = mp.fdiv(res, sigma)  # 1+erf / (sqrt{pi} s)
        aux_ = mp.sqrt(2)
        aux_ = mp.fdiv(e, aux_)
        aux_ = mp.fdiv(aux_, sigma)
        aux_ = mp.erf(aux_)
        aux_ = mp.fadd(aux_, 1)
        aux_ = mp.fmul(aux_, aux_)
        res = mp.fdiv(res, aux_)
        # alpha implementation
        aux = mp.fmul(alpha, e)  # alpha*e
        aux2 = mp.fmul(alpha, sigma)  # alpha*sigma
        aux2 = mp.fmul(aux2, aux2)  # (alpha*sigma)^2
        aux2 = mp.fdiv(aux2, mpf(4))  # (alpha*sigma)^2 / 4
        aux = mp.fadd(aux, aux2)  # (alpha*sigma)^2 / 4 + alpha*e
        aux = mp.exp(aux)
        res = mp.fmul(res, aux)
    elif ker_type == "CAUCHY":
        # Define the function to be integrated
        def integrand2(k):
            aux = alpha * k
            aux = mp.exp(aux)
            aux2 = cauchy(k, sigma, e) ** 2
            aux = aux * aux2
            return aux

        res = mp.quad(integrand2, [e0, mp.inf], method="gauss-legendre")
    else:
        raise ValueError("Invalid smearing kernel (par.ker_type)")

    return res


def a0_array(espace_mp, par, alpha):
    """
    Functional A[g=0], as an array of dimension len(espace_mp).
    """
    e0_ = par.e0
    a0_e = mp.matrix(par.Ne, 1)
    for ei in range(par.Ne):
        a0_e[ei] = a0_scalar(
            e=espace_mp[ei],
            sigma=par.mpsigma,
            alpha=mpf(alpha),
            e0=e0_,
            ker_type=par.kerneltype,
        )
    return a0_e


def integrandSigmaMat(e1, alpha, s, t1, t2, E0, par):
    _res = ft_mp(
        e=e1,
        t=t2,
        sigma_=s,
        alpha=alpha,
        e0=E0,
        type=par.periodicity,
        T=par.time_extent,
        ker_type=par.kerneltype,
    )
    if par.periodicity == "EXP":
        _res = _res * mp.exp(-t1 * e1)
    if par.periodicity == "COSH":
        _res = _res * (mp.exp(-t1 * e1) + mp.exp((-par.time_extent + t1) * e1))
    return _res


def SigmaMat(alpha, s, e0, par):
    SigmaMat_ = mp.matrix(par.tmax, par.tmax)

    for i in range(par.tmax):
        for j in range(par.tmax):
            entry = mp.quad(
                lambda x: integrandSigmaMat(x, alpha, s, i, j, e0, par),
                [e0, mp.inf],
                error=True,
            )
            SigmaMat_[i, j] = entry[0]
    return SigmaMat_


def gte(T, t, e, periodicity):
    """
    Laplace kernel exp(-t E) and its periodic generalisation.
    """
    if periodicity == "COSH":
        return mp.fadd(mp.exp((-T + t) * e), mp.exp(-t * e))
    if periodicity == "EXP":
        return mp.exp(-t * e)
