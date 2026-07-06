import numpy as np
import math
import random
from mpmath import mp


def averageVector_fp(vector, get_error=True, get_var=False):
    sum = 0
    stdv = 0
    for i in range(len(vector)):
        sum += vector[i]
    sum /= len(vector)
    for i in range(len(vector)):
        stdv += (vector[i] - sum) ** 2
    stdv /= len(vector) - 1
    stdv = math.sqrt(stdv)
    var = stdv
    err = stdv / math.sqrt(len(vector))
    if get_error is True:
        if get_var is True:
            return sum, var
        if get_var is False:
            return sum, err
    if get_error is False:
        return sum


def parallel_bootstrap_compact_fp(par_, in_, out_, start, end, seed, is_folded=False):
    import lsdensities.utils.rhoUtils

    random.seed(seed)
    randv = np.zeros(par_.num_samples)
    if is_folded is False:
        for b in range(start, end):
            randv = lsdensities.utils.rhoUtils.ranvec(
                randv, par_.num_samples, 0, par_.num_samples
            ).astype(int)
            for i in range(par_.time_extent):
                out_[b][i] = np.mean(in_[randv[:], i])
    if is_folded is True:
        for b in range(start, end):
            randv = lsdensities.utils.rhoUtils.ranvec(
                randv, par_.num_samples, 0, par_.num_samples
            ).astype(int)
            for i in range(int(par_.time_extent / 2) + 1):
                out_[b][i] = np.mean(in_[randv[:], i])


def averageVector_mp(in_, bootstrap=True):
    xlen_ = in_.rows
    samplesize_ = in_.cols
    out_ = mp.matrix(xlen_, 2)
    for x in range(xlen_):
        out_[x, 0] = 0
        out_[x, 1] = 0
        for b in range(samplesize_):
            out_[x, 0] = mp.fadd(out_[x, 0], in_[x, b])
        out_[x, 0] = mp.fdiv(out_[x, 0], samplesize_)
        for b in range(samplesize_):
            aux_ = mp.fsub(in_[x, b], out_[x, 0])
            aux_ = mp.fmul(aux_, aux_)
            out_[x, 1] = mp.fadd(out_[x, 1], aux_)
        out_[x, 1] = mp.fdiv(out_[x, 1], samplesize_)
        out_[x, 1] = mp.sqrt(out_[x, 1])
        if bootstrap is False:
            aux_ = mp.sqrt(samplesize_)
            out_[x, 1] = mp.fdiv(out_[x, 1], aux_)
    return out_


def averageScalar_mp(in_, bootstrap=True):
    if in_.rows == 1:
        samplesize_ = in_.cols
    if in_.cols == 1:
        samplesize_ = in_.rows
    out_ = mp.matrix(2, 1)
    out_[0] = 0
    out_[1] = 0
    for b in range(samplesize_):
        out_[0] = mp.fadd(out_[0], in_[b])
    out_[0] = mp.fdiv(out_[0], samplesize_)
    for b in range(samplesize_):
        aux_ = mp.fsub(in_[b], out_[0])
        aux_ = mp.fmul(aux_, aux_)
        out_[1] = mp.fadd(out_[1], aux_)
    out_[1] = mp.fdiv(out_[1], samplesize_)
    out_[1] = mp.sqrt(out_[1])
    if bootstrap is False:
        aux_ = mp.sqrt(samplesize_)
        out_[1] = mp.fdiv(out_[1], aux_)
    return out_
