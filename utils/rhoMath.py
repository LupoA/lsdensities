import numpy as np
from numpy import linalg as LA
import math
import scipy.linalg as sp_linalg

def kronecker_fp(a, b):
    if a == b:
        return 1
    if a != b:
        return 0


def halfnorm_fp(e, s):  # int_0^inf dE exp{(-e-e0)^2/2s^2}
    res_ = math.erf(e / (np.sqrt(2) * s))
    res_ += 1
    res_ *= s
    res_ *= np.sqrt(np.pi / 2)
    return res_


def gauss_fp(x, x0, sigma, norm="Full"):
    if sigma == 0:
        return kronecker_fp(x, x0)
    if norm == "Full" or norm == "full":
        return (np.exp(-((x - x0) ** 2) / (2 * (sigma**2)))) / (
            sigma * math.sqrt(2 * math.pi)
        )
    if norm == "None" or norm == "none":
        return np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    if norm == "Half" or norm == "half":
        return (np.exp(-((x - x0) ** 2) / (2 * sigma**2))) / halfnorm_fp(x0, sigma)


def norm2_fp(matrix):  # for square matrices only
    assert matrix.shape[0] == matrix.shape[1]
    return LA.norm(matrix) / np.sqrt(matrix.shape[0])

def choelesky_invert_scipy(in_):    # invert positive definite matrix. wee faster than numpy
    _L, _lower = sp_linalg.cho_factor(in_)
    out_ = sp_linalg.cho_solve((_L, _lower), np.eye(in_.shape[0]))
    return out_

from mpmath import mp, mpf

def norm2_mp(matrix):  # for square matrices only
    assert matrix.cols == matrix.rows
    return mp.norm(matrix) / mp.sqrt(matrix.cols)
