from mpmath import mp, mpf
from progressbar import ProgressBar
from .core import ft_mp, gte
from .utils.rhoStat import averageScalar_mp, averageVector_mp


def h_Et_mp_Eslice(Tinv_, params, estar_, alpha_):
    ht_ = mp.matrix(params.tmax, 1)
    for i in range(params.tmax):
        for j in range(params.tmax):
            ht_[i] += Tinv_[i, j] * ft_mp(
                e=estar_,
                t=mpf(j + 1),
                sigma_=params.mpsigma,
                alpha=mpf(alpha_),
                e0=params.mpe0,
                type=params.periodicity,
                T=params.time_extent,
                ker_type=params.kerneltype,
            )
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
    rhob = mp.matrix(params.num_boot, 1)
    for b in range(params.num_boot):
        y = mpmatrix[b, :]
        rhob[b] = 0
        for i in range(params.tmax):
            aux_ = mp.fmul(ht_sliced[i], y[i])
            rhob[b] = mp.fadd(rhob[b], aux_)
    return averageScalar_mp(rhob)


def y_combine_central_Eslice_mp(ht_sliced, y, params):
    rho = 0
    for i in range(params.tmax):
        aux_ = mp.fmul(ht_sliced[i], y[i])
        rho = mp.fadd(rho, aux_)
    return rho


def combine_fMf_Eslice(
    ht_sliced, params, estar_, alpha_
):  #   Compute f Minv f = g_t * f_t
    out_ = 0
    for i in range(params.tmax):
        aux_ = mp.fmul(
            ht_sliced[i],
            ft_mp(
                estar_,
                mpf(i + 1),
                sigma_=params.mpsigma,
                alpha=alpha_,
                e0=params.mpe0,
                type=params.periodicity,
                T=params.time_extent,
                ker_type=params.kerneltype,
            ),
        )
        out_ = mp.fadd(out_, aux_)
    return out_


def combine_base_Eslice(ht_sliced, params, estar):
    out_ = 0
    for i in range(params.tmax):
        aux_ = mp.fmul(
            ht_sliced[i],
            gte(
                T=params.time_extent,
                t=mpf(i + 1),
                e=mpf(str(estar)),
                periodicity=params.periodicity,
            ),
        )
        out_ = mp.fadd(out_, aux_)
    return out_


def y_combine_sample_Eslice_mp_ToFile(file, ht_sliced, mpmatrix, params):
    rhob = mp.matrix(params.num_boot, 1)
    with open(file, "w") as output:
        for b in range(params.num_boot):
            y = mpmatrix[b, :]
            rhob[b] = 0
            for i in range(params.tmax):
                aux_ = mp.fmul(ht_sliced[i], y[i])
                rhob[b] = mp.fadd(rhob[b], aux_)
            print(b, float(rhob[b]), file=output)
        # print(LogMessage(), "rho[e] +/- stat ", float(averageScalar_mp(rhob)[0]), (float(averageScalar_mp(rhob)[1])))
    return averageScalar_mp(rhob)


def combine_likelihood(minv, params, mpcorr):
    out_ = 0
    aux = mp.matrix(params.tmax, 1)
    for i in range(params.tmax):
        aux[i] = 0
        for j in range(params.tmax):
            aux[i] += minv[i, j] * mpcorr[j]
        out_ += aux[i] * mpcorr[i]
    return out_
