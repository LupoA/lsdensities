from lsdensities.utils.rhoUtils import LogMessage
from mpmath import mp


def test_init_precision():
    print(LogMessage(), " Initialising...")
    mp.dps = 64
    print(LogMessage(), " Binary precision in bit: ", mp.prec)
    print(LogMessage(), " Approximate decimal precision: ", mp.dps)


test_init_precision()
