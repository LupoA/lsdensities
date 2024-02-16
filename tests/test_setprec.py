from lsdensities.utils.rhoUtils import LogMessage
from mpmath import mp



def init_precision():
    print(LogMessage(), " Initialising...")
    mp.dps = 64
    print(LogMessage(), " Binary precision in bit: ", mp.prec)
    print(LogMessage(), " pproximate decimal precision: ", mp.dps)


if __name__ == "__main__":
    init_precision()

