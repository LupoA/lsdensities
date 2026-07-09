from mpmath import mp, mpf
from .core import ft_mp, gte
from .utils.rhoStat import averageScalar_mp


def coefficients_ssd(matrix, params, estar, alpha):
    """
    Computes the coefficients spanning the smeared spectral density
        gt = hlt_matrix * ft_mp

    Operation is performed for a single energy "estar"
    """
    ft = mp.matrix(params.tmax, 1)
    for j in range(params.tmax):
        ft[j] = ft_mp(
            e=estar,
            t=mpf(j + 1),
            sigma_=params.mpsigma,
            alpha=mpf(alpha),
            e0=params.mpe0,
            type=params.periodicity,
            T=params.time_extent,
            ker_type=params.kerneltype,
        )
    return matrix * ft


def get_ssd_scalar(gt, corr, params):
    """
    Computes smeared spectral density rho = sum_t g(t) c(t) at fixed energy
    Scalar version: the operation is performed at a single energy

    :param gt: mp.matrix len(params.tmax)
    :param corr: mp.matrix len(params.tmax)
    :param params: instance of Inputs class
    :return: mpf(float)
    """
    rho = 0
    for i in range(params.tmax):
        aux_ = mp.fmul(gt[i], corr[i])
        rho = mp.fadd(rho, aux_)
    return rho


def get_ssd_averaged_scalar(gt, corr_samples, params, sample_type="bootstrap"):
    """
    Computes smeared spectral density rho = sum_t g(t) c(t) at fixed energy
    Averaged version: the operation is performed on a vector of correlators, corresponding to different statistical samples. Result is averaged.
    Scalar version: the operation is performed at a single energy.

    :param gt: mp.matrix len(params.Ne, params.tmax)
    :param corr_samples: mp.matrix of dimensions (params.num_boot, params.tmax).
    :param params: instance of Inputs class
    :param sample_type: how the samples in `corr_samples` were obtained -- must
        match the correlator's own Obs.sample_type (see rhoUtils._variance_scale_factor)
        so that the error on rho is scaled consistently with the error on the correlator.
    :return: [mpf(float), mpf(float)] corresponding to avg and std
    """
    rhob = mp.matrix(params.num_boot, 1)
    for b in range(params.num_boot):
        y = corr_samples[b, :]
        rhob[b] = 0
        for i in range(params.tmax):
            aux_ = mp.fmul(gt[i], y[i])
            rhob[b] = mp.fadd(rhob[b], aux_)
    return averageScalar_mp(rhob, sample_type=sample_type)


def combine_fMf_scalar(gt, params, estar, alpha):
    """
    Computes f Minv f = g_t * f_t
    Scalar version: operation is performed at a single energy "estar"
    """
    out_ = 0
    for i in range(params.tmax):
        aux_ = mp.fmul(
            gt[i],
            ft_mp(
                estar,
                mpf(i + 1),
                sigma_=params.mpsigma,
                alpha=alpha,
                e0=params.mpe0,
                type=params.periodicity,
                T=params.time_extent,
                ker_type=params.kerneltype,
            ),
        )
        out_ = mp.fadd(out_, aux_)
    return out_


def combine_base_scalar(gt, params, estar):
    """
    Computes sum_t g(t, omega) exp(-tE) or its periodic generalisation
    Scalar version: operation is performed at a single energy "estar".
    """
    out_ = 0
    for i in range(params.tmax):
        aux_ = mp.fmul(
            gt[i],
            gte(
                T=params.time_extent,
                t=mpf(i + 1),
                e=mpf(str(estar)),
                periodicity=params.periodicity,
            ),
        )
        out_ = mp.fadd(out_, aux_)
    return out_


def y_combine_sample_Eslice_mp_ToFile(file, ht_sliced, mpmatrix, params, sample_type="bootstrap"):
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
    return averageScalar_mp(rhob, sample_type=sample_type)


def combine_likelihood(minv, params, mpcorr):
    """
    :param minv: mp.matrix
    :param params: instance of Inputs class
    :param mpcorr: mp.matrix of len params.tmax
    :return: mpf(float)
    """
    out_ = 0
    aux = mp.matrix(params.tmax, 1)
    for i in range(params.tmax):
        aux[i] = 0
        for j in range(params.tmax):
            aux[i] += minv[i, j] * mpcorr[j]
        out_ += aux[i] * mpcorr[i]
    return out_
