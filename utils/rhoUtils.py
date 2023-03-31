import numpy as np
import matplotlib.pyplot as plt
import random as rd
import os
from rhoStat import averageVector_fp
import time

CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

timesfont = {
    "family": "Times",
    "color": "black",
    "weight": "normal",
    "size": 16,
}

helveticafont = {
    "family": "Helvetica",
    "color": "black",
    "weight": "normal",
    "size": 16,
}

start_time = time.time()


def LogMessage():
    return "Message ::: {:2.5f}".format(time.time() - start_time) + " s :::"


def end():
    print(LogMessage(), "Exit")
    exit()


def ranvec(vec, dim, a, b):
    for j in range(0, dim):
        vec[j] = rd.randint(a, b - 1)
    return vec


class obs:
    def __init__(self, T_, nms_=1):
        self.central = np.zeros(T_)  # Central value of the sample
        self.err = np.zeros(T_)  # Error on the central value
        self.sigma = np.zeros(T_)  # Variance of the sample
        self.T = T_
        self.nms = nms_
        self.sample = np.zeros((nms_, T_))  # Sample elements
        self.cov = np.zeros((T_, T_))  # Cov matrix estimated from sample
        self.corrmat = np.zeros((T_, T_))  # Corr matrix estimated from sample

    def evaluate(self, resampled=False):
        # Given the sample, it evaluates the average and error. Sample can be bootstrap
        for i in range(self.T):
            self.central[i], self.sigma[i] = averageVector_fp(
                self.sample[:, i], get_error=True, get_var=True
            )
        if resampled == False:
            self.err[i] = self.sigma[i] / np.sqrt(self.nms)
        if resampled == True:
            self.err = self.sigma

    def plot(self, show=True, logscale=True, label=None, yscale=1):
        plt.tight_layout()
        plt.grid(alpha=0.1)
        if logscale == True:
            plt.yscale("log")
        plt.errorbar(
            x=list(range(0, self.T)),
            y=self.central / yscale,
            yerr=self.err / yscale,
            marker="o",
            markersize=1.5,
            elinewidth=1,
            ls="",
            label=label,
            color="#377eb8",
        )
        plt.legend(prop={"size": 12, "family": "Helvetica"})
        if show == True:
            plt.show()


def print_hlt_format(mtobs, T, nms, filename, directory):
    cout = os.path.join(directory, filename)
    with open(cout, "w") as output:
        print(nms, T, T, "2", "3", file=output)
        for j in range(0, nms):
            for i in range(0, T):
                print(i, mtobs[j, i], file=output)


def read_hlt_format(filename_, directory_):
    cin = os.path.join(directory_, filename_)
    with open(cin, "r") as file:
        # Read header
        header = next(file).strip()
        print(LogMessage(), "Header: ", header)
        header_nms = int(header.split(" ")[0])
        header_T = int(header.split(" ")[1])
        print(LogMessage(), "Reading correlator with T = ", header_T)
        print(LogMessage(), "Reading correlator with measurements = ", header_nms)
        mcorr_ = obs(header_T, header_nms)
        # loop over file: read and store
        indx = 0
        for l in file:
            # Read and store
            t = int(l.split(" ")[0])
            n = int(indx / header_T)
            # print(l.rstrip(), "     ", t, n)
            mcorr_.sample[n, t] = float(l.split(" ")[1])
            indx += 1
    return mcorr_


from mpmath import mp, mpf


def init_precision(digits_):
    mp.dps = digits_
    print(LogMessage(), "Setting precision ::::", "Binary precision in bit: ", mp.prec)
    print(
        LogMessage(),
        "Setting precision ::::",
        "Approximate decimal precision: ",
        mp.dps,
    )


class inputs:
    def __init__(self):
        self.time_extent = -1
        self.tmax = -1
        self.indir = "None"
        self.outdir = "None"
        self.num_boot = -1
        self.num_samples = -1
        self.sigma = -1
        self.emax = -1
        self.Ne = 1
        self.alpha = 0
        self.emin = 0
        # self.l = -1
        self.prec = -1
        self.mpsigma = mpf("0")
        self.mpemax = mpf("0")
        self.mplambda = mpf("0")

    def assign_mp_values(self):
        self.mpsigma = mpf(str(self.sigma))
        self.mpemax = mpf(str(self.emax))
        self.mpalpha = mpf(str(self.alpha))
        self.mpemin = mpf(str(self.emin))
        # self.mplambda = mpf(str(self.l))

    def report(self):
        print(LogMessage(), "Init ::: ", "Reading directory:", self.indir)
        print(LogMessage(), "Init ::: ", "Output directory:", self.outdir)
        print(LogMessage(), "Init ::: ", "Time extent:", self.time_extent)
        print(LogMessage(), "Init ::: ", "tmax:", self.tmax)
        print(LogMessage(), "Init ::: ", "sigma :", self.sigma)
        print(LogMessage(), "Init ::: ", "mp sigma ", self.mpsigma)
        print(LogMessage(), "Init ::: ", "Samples :", self.num_samples)
        print(LogMessage(), "Init ::: ", "Bootstrap samples :", self.num_boot)
        print(LogMessage(), "Init ::: ", "Number of energies :", self.Ne)
        print(LogMessage(), "Init ::: ", "a Emax ", self.emax)
        print(LogMessage(), "Init ::: ", "mp Emax ", self.mpemax)
        print(LogMessage(), "Init ::: ", "alpha ", self.alpha)
        print(LogMessage(), "Init ::: ", "mp alpha ", self.mpalpha)
        print(LogMessage(), "Init ::: ", "a Emin ", self.emin)
        print(LogMessage(), "Init ::: ", "mp Emin ", self.mpemin)
