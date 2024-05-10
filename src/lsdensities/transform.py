from mpmath import mp, mpf
from .core import ft_mp, gte
from .utils.rhoStat import average_1d_mpmatrix, average_2d_mpmatrix


def coefficients_ssd(matrix, params, estar, alpha):
    """
    Computes the coefficients spanning the smeared spectral density
        gt = hlt_matrix * ft_mp

    Operation is performed for a single energy "estar"
    """
    gt = mp.matrix(params.tmax, 1)
    for i in range(params.tmax):
        for j in range(params.tmax):
            gt[i] += matrix[i, j] * ft_mp(
                e=estar,
                t=mpf(j + 1),
                sigma_=params.mpsigma,
                alpha=mpf(alpha),
                e0=params.mpe0,
                type=params.periodicity,
                T=params.time_extent,
                ker_type=params.kerneltype,
            )
    return gt


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


def get_ssd_vector(gt, corr, params):
    """
    Computes smeared spectral density rho = sum_t g(t) c(t) at fixed energy
    Vector version: the operation is performed on a single vector of dimension par.tmax
    for a vector of energies

    :param gt: mp.matrix len(params.Ne, params.tmax)
    :param corr: mp.matrix len(params.tmax)
    :param params: instance of Inputs class
    :return: mp.matrix(params.Ne)
    """
    rho = mp.matrix(params.Ne, 1)
    for e in range(params.Ne):
        rho[e] = 0
        for i in range(params.tmax):
            aux_ = mp.fmul(gt[e, i], corr[i])
            rho[e] = mp.fadd(rho[e], aux_)
    return rho


def get_ssd_averaged_vector(gt, corr_type, params):
    """
    Computes smeared spectral density rho = sum_t g(t) c(t) at fixed energy.
    Averaged version: the operation is performed on a vector of correlators, corresponding to different statistical samples. Result is averaged.
    Vector version: the operation is performed at an array of energies.

    :param gt: mp.matrix len(params.Ne, params.tmax)
    :param corr_type: instance of Obs type
    :param params: instance of Inputs class
    :return: [mp.matrix(params.Ne), mp.matrix(params.Ne)] corresponding to avg and std
    """
    rhob = mp.matrix(params.Ne, params.num_boot)
    for b in range(params.num_boot):
        y = corr_type.sample[b][:]
        for e in range(params.Ne):
            rhob[e, b] = 0
            for i in range(params.tmax):
                aux_ = mp.fmul(gt[e, i], y[i])
                rhob[e, b] = mp.fadd(rhob[e, b], aux_)
    return average_2d_mpmatrix(rhob)


def get_ssd_averaged_scalar(gt, corr_samples, params):
    """
    Computes smeared spectral density rho = sum_t g(t) c(t) at fixed energy
    Averaged version: the operation is performed on a vector of correlators, corresponding to different statistical samples. Result is averaged.
    Scalar version: the operation is performed at a single energy.

    :param gt: mp.matrix len(params.Ne, params.tmax)
    :param corr_samples: mp.matrix of dimensions (params.num_boot, params.tmax).
    :param params: instance of Inputs class
    :return: [mpf(float), mpf(float)] corresponding to avg and std
    """
    rhob = mp.matrix(params.num_boot, 1)
    for b in range(params.num_boot):
        y = corr_samples[b, :]
        rhob[b] = 0
        for i in range(params.tmax):
            aux_ = mp.fmul(gt[i], y[i])
            rhob[b] = mp.fadd(rhob[b], aux_)
    return average_1d_mpmatrix(rhob)


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
    Computes sum_t g(t) exp(-tE) or its periodic generalisation
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
        # print(LogMessage(), "rho[e] +/- stat ", float(average_1d_mpmatrix(rhob)[0]), (float(average_1d_mpmatrix(rhob)[1])))
    return average_1d_mpmatrix(rhob)


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
