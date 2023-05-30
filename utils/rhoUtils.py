import numpy as np
import matplotlib.pyplot as plt
import random as rd
import os
import math
from rhoStat import averageVector_fp, getCovMatrix_fp
from core import Smatrix_mp
from rhoMath import norm2_mp
import time
import mpmath as mp

target_result_precision = 1e-8


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


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

CB_colors = [
    '#1f77b4',  # Dark Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Light Green
    '#d62728',  # Reddish Purple
    '#9467bd',  # Light Blue
    '#8c564b',  # Dark Yellow
    '#e377c2',  # Cyan
    '#7f7f7f'   # Olive Green
]

plot_markers = ['o', 's', 'D', 'v', '^', 'p', '*', 'h']

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


class Obs:
    def __init__(self, T_: int, nms_: int = 1, tmax_ : int = 1, is_resampled=False):
        self.central = np.zeros(T_)  # Central value of the sample
        self.err = np.zeros(T_)  # Error on the central value
        self.sigma = np.zeros(T_)  # Variance of the sample
        self.T = T_
        self.nms = nms_
        self.sample = np.zeros((nms_, T_))  # Sample elements
        self.cov = np.zeros((T_, T_))  # Cov matrix estimated from sample
        self.corrmat = np.zeros((T_, T_))  # Corr matrix estimated from sample
        self.is_resampled = is_resampled
        if tmax_ == 1:
            self.tmax = T_-1
        else:
            self.tmax = tmax_
        self.mpsample = mp.matrix(self.nms, self.tmax)
        self.mpcov = mp.matrix(self.tmax, self.tmax)

    def evaluate(self):
        # Given the sample, it evaluates the average and error. Sample can be bootstrap
        for i in range(self.T):
            self.central[i], self.sigma[i] = np.average(self.sample[:, i]), np.std(
                self.sample[:, i], ddof=1
            )
        if self.is_resampled == False:
            self.err = self.sigma / np.sqrt(self.nms)
        if self.is_resampled == True:
            self.err = self.sigma

    def evaluate_covmatrix(self, plot=False):
        assert self.is_resampled
        sample_matrix = np.array(self.sample).T
        self.cov = np.cov(sample_matrix, bias=False)
        if plot:
            plt.imshow(self.cov, cmap="viridis")
            plt.colorbar()
            plt.show()
            plt.clf()
        return self.cov

    def corrmat_from_covmat(self, plot=False):
        for vi in range(self.T):
            for vj in range(self.T):
                self.corrmat[vi][vj] = self.cov[vi][vj] / (
                    self.sigma[vi] * self.sigma[vj]
                )
        if plot == True:
            plt.imshow(self.corrmat)
            plt.colorbar()
            plt.show()
            plt.clf()

    def fill_mp_sample(self):
        for n in range(self.nms):
            for i in range(self.tmax):
                self.mpsample[n, i] = mpf(str(self.sample[n][i + 1]))
        #   Get cov for B matrix
        self.mpcov = mp.matrix(self.tmax)
        for i in range(self.tmax):
            for j in range(self.tmax):
                self.mpcov[i, j] = mpf(str(self.cov[i + 1][j + 1]))

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


def read_datafile(datapath_, resampled=False):  # (filename_, directory_):
    #   The input file has a header with time_extent and number of measurements.
    #   then data config by config. Example:
    #   32  100
    #   0   corr[0]
    #   1   corr[1]
    #   ...
    #   31  corr[31]
    #   0   corr[0]
    #   ... so on

    #   cin = os.path.join(directory_, filename_)
    with open(datapath_, "r") as file:
        # Read header
        header = next(file).strip()
        print(LogMessage(), "Reading file :::", "Header: ", header)
        header_nms = int(header.split(" ")[0])
        header_T = int(header.split(" ")[1])
        print(LogMessage(), "Reading file :::", "Time extent ", header_T)
        print(LogMessage(), "Reading file :::", "Measurements ", header_nms)
        mcorr_ = Obs(header_T, header_nms, is_resampled=resampled)
        # loop over file: read and store
        indx = 0
        for l in file:
            # Read and store
            t = int(l.split(" ")[0])
            n = int(indx / header_T)
            # print(l.rstrip(), "     ", t, n)
            mcorr_.sample[n, t] = float(l.split(" ")[1])
            indx += 1
    #   Returns np array of correlators
    return mcorr_, header_T, header_nms


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


class Inputs:
    def __init__(self):
        self.time_extent = -1
        self.tmax = -1
        self.datapath = "None"
        self.outdir = "None"
        self.num_boot = -1
        self.num_samples = -1
        self.sigma = -1
        self.emax = -1
        self.Ne = 1
        self.alpha = 0
        self.emin = 0
        self.e0 = 0
        self.massNorm = 1.0
        # self.l = -1
        self.prec = -1
        self.mpsigma = mpf("0")
        self.mpemax = mpf("0")
        self.mpemin = mpf("0")
        self.mpe0 = mpf("0")
        self.mplambda = mpf("0")
        self.mpMpi = mpf("0")

    def assign_values(self):
        if self.tmax==0:
            self.tmax = self.time_extent - 1
        self.mpsigma = mpf(str(self.sigma))
        self.mpemax = mpf(str(self.emax))
        self.mpemin = mpf(str(self.emin))
        self.mpalpha = mpf(str(self.alpha))
        self.mpe0 = mpf(str(self.e0))
        self.mpMpi = mpf(str(self.massNorm))

    def report(self):
        print(LogMessage(), "Init ::: ", "Reading file:", self.datapath)
        print(LogMessage(), "Init ::: ", "Output directory:", self.outdir)
        print(LogMessage(), "Init ::: ", "Time extent:", self.time_extent)
        print(LogMessage(), "Init ::: ", "Mpi:", self.massNorm)
        print(LogMessage(), "Init ::: ", "tmax:", self.tmax)
        print(
            LogMessage(), "Init ::: ", "sigma (mp):", self.sigma, "(", self.mpsigma, ")"
        )
        print(LogMessage(), "Init ::: ", "Samples :", self.num_samples)
        print(LogMessage(), "Init ::: ", "Bootstrap samples :", self.num_boot)
        print(LogMessage(), "Init ::: ", "Number of energies :", self.Ne)
        print(LogMessage(), "Init ::: ", "Emax (mp) [lattice unit]", self.emax, self.mpemax)
        print(LogMessage(), "Init ::: ", "Emax [mass units]", self.emax / self.massNorm)
        print(LogMessage(), "Init ::: ", "Emin (mp) [lattice unit]", self.emin, "(", self.mpemin, ")")
        print(LogMessage(), "Init ::: ", "Emin [mass units]", self.emin / self.massNorm)
        print(LogMessage(), "Init ::: ", "alpha (mp)", self.alpha, "(", self.mpalpha, ")")

class LambdaSearchOptions:
    def __init__(self, lmin: float = 0.01, lmax: float = 0.6, ldensity: int = 20, kfactor = 10, star_at = 1):
        self.lmin = lmin
        self.lmax = lmax
        self.ldensity = ldensity
        self.k_star = star_at  # Meaning lambda is found at A = k_star B.
        self.kfactor = kfactor  #   Meaning lambda dependence is checked at A = kfactor B
        self.lspace = np.linspace(lmin, lmax, ldensity)

class MatrixBundle:
    def __init__(self, Smatrix: mp.matrix, Bmatrix: mp.matrix, bnorm = mpf(1)):
        self.S = Smatrix
        self.B = Bmatrix
        self.bnorm = bnorm

    def compute_W(self, l, a0):
        return self.S + (l*a0)/((1-l)*self.bnorm)*self.B

def adjust_precision(tmax: int):
    #   This function should *reduce* the numerical precision
    #   from the large input value
    #   to a value suggested by the condition
    #   number of S
    #   If the starting prec is too small the function might
    #   too small of a value which results in a warning
    S_ = Smatrix_mp(tmax)
    condS = mp.cond(S_)
    n_prec = math.ceil(math.log10(condS)) + 3  #   +3 to be extra cautious
    print(
        LogMessage(),
        "Adjust precision ::: ",
        "Suggested numerical precision based on tmax is {:4d}".format(n_prec),
    )
    print(LogMessage(), "Adjust precision ::: ", "Switching to suggested precision")
    init_precision(n_prec)
    S_ = Smatrix_mp(tmax)
    condS = mp.cond(S_)
    n_prec_post = math.ceil(math.log10(condS)) + 3
    if n_prec_post != n_prec:
        print(
            LogMessage(),
            f"{bcolors.WARNING}Warning{bcolors.ENDC} ::: Suggested precision might be small. Suggest restarting with higher --prec option",
        )
        print(
            LogMessage(),
            f"{bcolors.WARNING}Warning{bcolors.ENDC} ::: Asserting whether minimal precision 1e-8 on the inversion is guaranteed ",
        )
    invS = S_ ** (-1)
    diff = S_ * invS
    diff = norm2_mp(diff) - 1
    assert float(diff) < target_result_precision
    # print(LogMessage(), "Adjust precision ::: ", "Suggested precision stable with target result precision ", target_result_precision)
    return 0
