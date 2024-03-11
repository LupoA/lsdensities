from lsdensities.utils.rhoUtils import LogMessage
from mpmath import mp, mpf


def test_init_precision():
    print(LogMessage(), " Initialising...")
    mp.dps = 10
    print(LogMessage(), " Binary precision in bit: ", mp.prec)
    print(LogMessage(), " Approximate decimal precision: ", mp.dps)
    a = mpf(1 / 3)
    assert a - 0.3333333333 < 1e-10
