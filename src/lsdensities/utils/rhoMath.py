import numpy as np
from numpy import linalg as LA
import math
import scipy.linalg as sp_linalg
from mpmath import mp


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


def cauchy(k, sigma_, omega_):
    aux = omega_ - k
    aux = aux * aux + sigma_ * sigma_
    aux = sigma_ / aux
    return aux


def norm2_fp(matrix):  # for square matrices only
    assert matrix.shape[0] == matrix.shape[1]
    return LA.norm(matrix) / np.sqrt(matrix.shape[0])


def choelesky_invert_scipy(
    in_,
):  # invert positive definite matrix. wee faster than numpy
    _L, _lower = sp_linalg.cho_factor(in_)
    out_ = sp_linalg.cho_solve((_L, _lower), np.eye(in_.shape[0]))
    return out_


def norm2_mp(matrix):  # for square matrices only
    assert matrix.cols == matrix.rows
    return mp.norm(matrix) / mp.sqrt(matrix.cols)


def invert_matrix_ge(mp_matrix):
    matrix = mp_matrix.tolist()
    # Verify if the matrix is square
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("Matrix must be square.")

    # Create augmented matrix [A | I]
    augmented_matrix = [row + [0] * n for row in matrix]
    for i in range(n):
        augmented_matrix[i][n + i] = 1

    # Perform Gaussian elimination with partial pivoting
    for i in range(n):
        # Find pivot row
        pivot_row = max(range(i, n), key=lambda j: abs(augmented_matrix[j][i]))

        # Swap rows if necessary
        if pivot_row != i:
            augmented_matrix[i], augmented_matrix[pivot_row] = (
                augmented_matrix[pivot_row],
                augmented_matrix[i],
            )

        pivot = augmented_matrix[i][i]

        # Divide pivot row by pivot element
        for j in range(n * 2):
            augmented_matrix[i][j] /= pivot

        # Subtract multiples of pivot row from other rows
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(n * 2):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    # Extract inverse matrix from augmented matrix
    inverse_matrix = [row[n:] for row in augmented_matrix]
    mp_inverse_matrix = mp.matrix(inverse_matrix)

    return mp_inverse_matrix
