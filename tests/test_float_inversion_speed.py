import sys

sys.path.append("..")
from importall import *

tmax = 32
lambda_ = 0.00001
S = Smatrix_float64(tmax_=tmax, alpha_=0, e0=0)
identity = np.eye(tmax)
W = S + lambda_ * identity

print(LogMessage(), "Cond(W) = ", np.linalg.cond(W))

print(LogMessage(), "Inverting (numpy)")
invW_np = np.linalg.inv(W)
print(LogMessage(), "Inverted (numpy)")

print(LogMessage(), "Inverting (scipy choelesky)")
invW_csp = choelesky_invert_scipy(W)
print(LogMessage(), "Inverted (scipy choelesky)")
