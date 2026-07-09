from lsdensities.utils.math_utils import norm2_mp
from lsdensities.core import cauchy_matrix
from lsdensities.utils.common import LogMessage
from mpmath import mp


def test_mp_invert():
    print(LogMessage(), " Initialising...")
    mp.dps = 64
    print(LogMessage(), " Binary precision in bit: ", mp.prec)
    print(LogMessage(), " Approximate decimal precision: ", mp.dps)
    tmax = 15

    S = cauchy_matrix(tmax, alpha=0)

    invS = S ** (-1)

    identity = S * invS

    shouldbeone = norm2_mp(identity)

    print(LogMessage(), "norm( S Sinv ) - 1 = ", shouldbeone - 1)
    print(LogMessage(), "Target decimal precision was", mp.dps)

    assert shouldbeone - 1 < mp.dps / 2
