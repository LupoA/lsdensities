import sys

sys.path.append("..")
from importall import *
from mpmath import mp, mpf


def LUdecomp(a):
    n = a.cols
    for k in range(0, n - 1):
        pivot = a[k, k]

        for i in range(k + 1, n):
            value = a[i, k]  # Store the value for reuse
            if value != mpf(0):
                lam = value / pivot
                a[i, k + 1 : n] -= lam * a[k, k + 1 : n]
                a[i, k] = lam
    return a


def LUsolve(a, b):
    n = a.cols
    for k in range(1, n):
        dot_product = mp.fdot(a[k, 0:k], b[0:k])
        b[k] -= dot_product

    b[n - 1] /= a[n - 1, n - 1]

    for k in range(n - 2, -1, -1):
        dot_product = mp.fdot(a[k, k + 1 : n], b[k + 1 : n])
        b[k] = (b[k] - dot_product) / a[k, k]

    return b


def LUinverse_separated(a):
    size = a.cols
    ainv = mp.matrix(size)
    uno = mp.matrix(size, 1)

    lua = LUdecomp(a)

    for i in range(size):
        for j in range(size):
            uno[j] = 0
        uno[i] = 1
        column = LUsolve(lua, uno)
        # print(i, column)
        ainv[:, i] = column
    return ainv


def LUinverse_working(a):
    size = a.cols

    ainv = mp.matrix(size)
    uno = mp.matrix(size, 1)
    print(LogMessage(), "Start")
    # Perform LU decomposition
    for k in range(0, size - 1):
        pivot = a[k, k]

        for i in range(k + 1, size):
            value = a[i, k]
            if value != mpf(0):
                lam = value / pivot
                a[i, k + 1 : size] -= lam * a[k, k + 1 : size]
                a[i, k] = lam
    print(LogMessage(), "LU is done. Starting solve")
    # Solve for each column of the inverse
    for i in range(size):
        for j in range(size):
            uno[j] = 0
        uno[i] = 1

        # Forward substitution
        for k in range(1, size):
            dot_product = mp.fdot(a[k, 0:k], uno[0:k])
            uno[k] -= dot_product

        # Backward substitution
        uno[size - 1] /= a[size - 1, size - 1]
        for k in range(size - 2, -1, -1):
            dot_product = mp.fdot(a[k, k + 1 : size], uno[k + 1 : size])
            uno[k] = (uno[k] - dot_product) / a[k, k]

        ainv[:, i] = uno
    print(LogMessage(), "solve done")
    return ainv


def LUinverse(a):
    size = a.cols

    ainv = mp.matrix(size)
    uno = mp.matrix(size, 1)

    # Perform LU decomposition
    for k in range(0, size - 1):
        pivot = a[k, k]

        slice_a = a[k, k + 1 : size]
        for i in range(k + 1, size):
            value = a[i, k]
            if value != mpf(0):
                lam = value / pivot
                a[i, k + 1 : size] -= lam * slice_a
                a[i, k] = lam
    # Solve for each column of the inverse
    for i in range(size):
        for j in range(size):
            uno[j] = 0
        uno[i] = 1

        # Forward substitution
        for k in range(1, size):
            dot_product = mp.fdot(a[k, 0:k], uno[0:k])
            uno[k] -= dot_product

        # Backward substitution
        uno[size - 1] /= a[size - 1, size - 1]
        for k in range(size - 2, -1, -1):
            temp = mp.fdot(a[k, k + 1 : size], uno[k + 1 : size])
            uno[k] -= temp
            uno[k] /= a[k, k]

        ainv[:, i] = uno

    return ainv


def an_invertible_matrix():
    A = mp.matrix(3)
    A[0, 0] = 333
    A[0, 1] = 2
    A[0, 2] = mpf(str(1.5))
    A[1, 0] = 1
    A[1, 1] = 2
    A[1, 2] = mpf(str(0.3))
    A[2, 0] = mpf(str(0.3))
    A[2, 1] = 12
    A[2, 2] = 32
    return A


import copy
import time

import numpy as np
import mpmath

mp.dps = 64
print(LogMessage(), " Binary precision in bit: ", mp.prec)
print(LogMessage(), " Approximate decimal precision: ", mp.dps)

S = Smatrix_mp(50)

print(LogMessage(), "Condition number is ", mp.cond(S))
Scopy = copy.deepcopy(S)

print(LogMessage(), "MP inverting")
start_time = time.time()
Sinv = S ** (-1)
end_time = time.time()
print(LogMessage(), "MP inverted in ", end_time - start_time, "seconds")
print(LogMessage(), norm2_mp(Scopy * Sinv))
print("\n")
print(LogMessage(), "LU inverting")
start_time = time.time()
myinv = LUinverse(S)
end_time = time.time()
print(LogMessage(), "LU inverted in", end_time - start_time, "seconds")

print(LogMessage(), norm2_mp(Scopy * myinv))
print("\n")
print(LogMessage(), "Third inverting")

Scopycopy = copy.deepcopy(S)

start_time = time.time()
# Find the inverse of the S matrix
Sinv2 = invert_matrix_ge(S)
end_time = time.time()
print("Gauss Elimination in ", end_time - start_time, "seconds")


print(LogMessage(), norm2_mp(Scopycopy * Sinv2))

end()
