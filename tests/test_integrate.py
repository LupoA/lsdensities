from lsdensities.utils.rhoUtils import LogMessage
from lsdensities.core import integrandSigmaMat
from lsdensities.utils.rhoUtils import Inputs
from mpmath import mp
import time
from scipy.integrate import quad
import numpy as np


def test_time_integration():
    mp.dps = 120
    params = Inputs()
    params.periodicity = "COSH"
    params.kerneltype = "FULLNORMGAUSS"
    params.time_extent = 16

    start = time.time()
    integral = mp.quad(
        lambda x: integrandSigmaMat(x, alpha=0, s=0.1, t1=3, t2=3, E0=0, par=params),
        [0, mp.inf],
        error=False,
        method="tanh-sinh",
    )
    end = time.time()
    print(LogMessage(), float(integral), "in ", end - start, "s")

    integral2, _ = quad(
        lambda x: integrandSigmaMat(x, alpha=0, s=0.1, t1=3, t2=3, E0=0, par=params),
        0.0,
        np.inf,
    )

    print(integral)
    print(integral2)
    assert np.abs(integral2 - integral) < 1e-15
