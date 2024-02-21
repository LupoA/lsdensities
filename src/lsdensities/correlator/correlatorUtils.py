from ..utils.rhoUtils import Obs, LogMessage
import argparse
import numpy as np

#   Usage:
#       from correlatorUtils import effective_mass
#       effmass = effective_mass(corr, par, type='EXP')
#       effmass.plot(logscale=False)
#       print(effmass.central, '±', effmass.err)


def effective_mass(corr, par, type="COSH"):
    th = int(par.time_extent / 2)
    thm = th - 1
    mass = Obs(T=thm, nms=par.num_boot, is_resampled=True, tmax=thm)
    if type == "COSH":
        mass.sample[:, :] = np.arccosh(
            (corr.sample[:, 2 : th + 1] + corr.sample[:, 0 : th - 1])
            / (2 * corr.sample[:, 1:th])
        )
    elif type == "EXP":
        mass.sample[:, :] = -np.log(corr.sample[:, 1:th] / corr.sample[:, 0 : th - 1])
    else:
        raise ValueError("Invalid type specified. Only COSH and EXP are allowed.")

    mass.evaluate()
    return mass


def foldPeriodicCorrelator(corr, par, is_resampled=False):
    assert par.periodicity == "COSH"
    halfT = int(par.time_extent / 2)
    foldedCorr = Obs(
        T=halfT + 1, tmax=par.tmax, nms=par.num_samples, is_resampled=is_resampled
    )
    for n in range(par.num_samples):
        foldedCorr.sample[n, 0] = corr.sample[n, 0]
        for t in range(1, halfT + 1):
            foldedCorr.sample[n, t] = (
                corr.sample[n, t] + corr.sample[n, par.time_extent - t]
            ) / 2

    return foldedCorr


def symmetrisePeriodicCorrelator(corr, par):
    assert par.periodicity == "COSH"
    symmCorr = Obs(
        T=corr.T, tmax=corr.tmax, nms=corr.nms, is_resampled=corr.is_resampled
    )
    for n in range(par.num_samples):
        symmCorr.sample[n, 0] = corr.sample[n, 0]
        symmCorr.sample[n, int(symmCorr.T / 2)] = corr.sample[n, int(symmCorr.T / 2)]
        for t in range(1, int(symmCorr.T / 2)):
            symmCorr.sample[n, t] = (corr.sample[n, t] + corr.sample[n, corr.T - t]) / 2
            symmCorr.sample[n, symmCorr.T - t] = symmCorr.sample[n, t]

    return symmCorr


class InputsCorrelatorAnalysis:
    def __init__(
        self,
        time_extent: int = 0,
        datapath=".",
        outdir=".",
        num_boot: int = 0,
        num_samples: int = 0,
    ):
        if not isinstance(time_extent, int):
            raise TypeError("time_extent must be an integer")
        if not isinstance(num_samples, int):
            raise TypeError("num_samples must be an integer")
        if not isinstance(num_boot, int):
            raise TypeError("num_boot must be an integer")
        self.time_extent = time_extent
        self.datapath = datapath
        self.outdir = outdir
        self.num_boot = num_boot
        self.num_samples = num_samples
        self.periodicity = "EXP"
        self.tmax = 0

    def report(self):
        print(LogMessage(), "Init ::: ", "Reading file:", self.datapath)
        print(LogMessage(), "Init ::: ", "Output directory:", self.outdir)
        print(LogMessage(), "Init ::: ", "Time extent:", self.time_extent)
        print(LogMessage(), "Init ::: ", "Samples :", self.num_samples)
        print(LogMessage(), "Init ::: ", "Bootstrap samples :", self.num_boot)


def parseArgumentCorrelatorAnalysis():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-datapath",
        metavar="DataPile",
        type=str,
        help="Path to data file",
        required=True,
    )
    parser.add_argument(
        "--outdir", metavar="OutputDirectory", help="Directory for output", default="."
    )
    parser.add_argument(
        "--nboot",
        metavar="BootstrapSampleSize",
        type=int,
        help="Number of bootstrap samples. Default=300",
        default=300,
    )
    parser.add_argument(
        "--periodicity",
        type=str,
        help="Accepted stirngs are 'EXP' or 'COSH', depending on the correlator being periodic or open.",
        default="EXP",
    )
    args = parser.parse_args()
    return args
