import numpy as np
import matplotlib.pyplot as plt
import random as rd
import os
import time
from mpmath import mp, mpf
import hashlib
import logging

#   #   #   #   #   #  ----- logger -----   #   #   #   #   #   #

logger = logging.getLogger("log")
stream_handler = logging.StreamHandler()


class CustomFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self.start_time = time.time()
        super().__init__(*args, **kwargs)

    def format(self, record):
        elapsed_time_ms = time.time() - self.start_time
        record.elapsed_time = "{:.3f} s".format(elapsed_time_ms)
        return super().format(record)


formatter = CustomFormatter("Message ::: " + "%(elapsed_time)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def log(*args, **kwargs):
    level = kwargs.pop("level", logging.INFO)
    msg = " ".join(map(str, args))
    logger.log(level, msg, **kwargs)


start_time = time.time()


def LogMessage():
    return "Message ::: {:2.5f}".format(time.time() - start_time) + " s :::"


def end():
    print(LogMessage(), "Exit")
    exit()


#   #   #   #   #   #   #   #   #


def generate_seed(par):
    """
    :param par: Input class instance
    Generates seed from a hash of the inputs
    """
    # Concatenate the input parameters into a string
    input_string = f"{par.emin}{par.emax}{par.Ne}{par.time_extent}{par.sigma}"

    # Encode the string to bytes
    encoded_string = input_string.encode("utf-8")

    # Calculate the SHA-256 hash
    sha256_hash = hashlib.sha256(encoded_string).hexdigest()

    return sha256_hash


def create_out_paths(par):
    dir = os.path.join(par.outdir, par.directoryName)
    plotpath = os.path.join(dir, "Plots")
    logpath = os.path.join(dir, "Logs")
    os.makedirs(plotpath, exist_ok=True)
    os.makedirs(logpath, exist_ok=True)
    return plotpath, logpath


def ranvec(vec, dim, a, b):
    for j in range(0, dim):
        vec[j] = rd.randint(a, b - 1)
    return vec


#: Supported Obs.sample_type values.
SAMPLE_TYPES = ("montecarlo", "bootstrap", "jackknife")


def _variance_scale_factor(sample_type, n):
    """
    Var[central value] = _variance_scale_factor(sample_type, n) * Var[single config, ddof=1].

    - montecarlo: raw, independent measurements -> standard error of the mean, 1/n.
    - bootstrap: replicates' own spread already estimates the error directly, 1.
    - jackknife: delete-1 replicates are strongly correlated (each shares n-1 of n
      configs with every other), so their raw spread underestimates the true
      error; the textbook correction is (n-1)**2/n. See e.g. Efron & Tibshirani,
      "An Introduction to the Bootstrap", Ch. 11.
    """
    if sample_type == "montecarlo":
        return 1.0 / n
    if sample_type == "bootstrap":
        return 1.0
    if sample_type == "jackknife":
        return (n - 1) ** 2 / n
    raise ValueError(
        f"Invalid sample_type '{sample_type}' (expected one of {SAMPLE_TYPES})"
    )


class Obs:
    """
    Class for an array of observables
    T: lenght of the array.
    tmax: highest element of the array that is used for analysis.
    sample_type: one of "montecarlo" (raw, independent measurements), "bootstrap"
        or "jackknife" (samples already resampled from raw data). Determines how
        the error on the central value (and the covariance matrix) is obtained
        from the spread of `sample` -- see _variance_scale_factor.

    Attributes:
    central: values of the array.
    err: error on central (sample_type-dependent; see _variance_scale_factor).
    sigma: std of a single configuration (ddof=1), regardless of sample_type.
           NOT the error on central unless sample_type == "bootstrap" (where the two
           coincide by construction).
    nms: number of measurements,
    sample: an array of measurements (len = nms) for each array in the observable.
    cov: covariance matrix of the central value (same sample_type scaling as err,
         so that cov[i, i] == err[i]**2 always).
    corrmat: correlation matrix
    mpsample: sample, converted into mp variables, and of leght reduced from T to tmax
    mpcov: cov from mpsample
    mpcentral: central from mpsample
    """

    def __init__(self, T: int, tmax: int, nms: int = 1, sample_type="montecarlo"):
        if sample_type not in SAMPLE_TYPES:
            raise ValueError(
                f"Invalid sample_type '{sample_type}' (expected one of {SAMPLE_TYPES})"
            )
        self.central = np.zeros(T)  # Central value of the sample
        self.err = np.zeros(T)  # Error on the central value
        self.sigma = np.zeros(T)  # Std of a single configuration (ddof=1)
        self.T = T  # number of time slices
        self.tmax = tmax  # Max t we use
        self.nms = nms
        self.sample = np.zeros((nms, T))  # Sample elements
        self.cov = np.zeros((T, T))  # Cov matrix of the central value
        self.cholesky = np.zeros((T, T))
        self.corrmat = np.zeros((T, T))  # Corr matrix estimated from sample
        self.sample_type = sample_type
        self.mpsample = mp.matrix(self.nms, self.tmax)
        self.mpcov = mp.matrix(self.tmax, self.tmax)
        self.mpcholesky = mp.matrix(self.tmax, self.tmax)
        self.mpcentral = mp.matrix(self.tmax, 1)

        self.cholesky_evaluated = False
        self.central_err_evaluated = False
        self.cov_evaluated = False

    def evaluate(self):
        """
        From sample, computes central and err and store them into
        self.central, self.err
        """
        scale = np.sqrt(_variance_scale_factor(self.sample_type, self.nms))
        for i in range(self.T):
            self.central[i], self.sigma[i] = (
                np.average(self.sample[:, i]),
                np.std(self.sample[:, i], ddof=1),
            )
        self.err = self.sigma * scale
        self.central_err_evaluated = True

    def evaluate_covmatrix(self, plot=False, symmetrise=False, regularise=0):
        """
        From sample, computes the covariance matrix of the central value in self.cov
        (scaled consistently with self.err, i.e. self.cov[i, i] == self.err[i] ** 2).
        """
        scale = _variance_scale_factor(self.sample_type, self.nms)
        sample_matrix = np.array(self.sample).T
        self.cov = np.cov(sample_matrix, bias=False) * scale
        if plot:
            plt.imshow(self.cov, cmap="viridis")
            plt.colorbar()
            plt.show()
        self.cov_evaluated = True
        if symmetrise:
            self.cov = (self.cov + self.cov.T) * 0.5
        if regularise > 0:
            self.cov += np.eye(self.cov.shape[0]) * regularise
        return self.cov

    def evaluate_cholesky(self):
        assert self.cov_evaluated is True
        self.cholesky = np.linalg.cholesky(self.cov)
        self.cholesky_evaluated = True
        return self.cholesky

    def corrmat_from_covmat(self, plot=False):
        """
        Computes the correlation matrix from the covariance matrix
        and saves it into self.corrmat
        """
        cov_diag = np.diagonal(self.cov)
        for vi in range(self.T):
            for vj in range(self.T):
                self.corrmat[vi][vj] = self.cov[vi][vj] / np.sqrt(
                    cov_diag[vi] * cov_diag[vj]
                )
        if plot is True:
            plt.imshow(self.corrmat)
            plt.colorbar()
            plt.show()

    def fill_mp_sample(self, w_cholesky=False):
        """
        This operation also includes the shifting of the correlator index
        so that corr(0) is never used
        """
        assert self.cov_evaluated is True

        for n in range(self.nms):
            for i in range(self.tmax):  # tmax = T/2 if folded otherwise T-1
                self.mpsample[n, i] = mpf(str(self.sample[n][i + 1]))
        #   Get cov for B matrix
        self.mpcov = mp.matrix(self.tmax)
        for i in range(self.tmax):
            self.mpcentral[i] = self.central[i + 1]
            for j in range(self.tmax):
                self.mpcov[i, j] = mpf(str(self.cov[i + 1][j + 1]))

        if w_cholesky:
            assert self.cholesky_evaluated is True
            for i in range(self.tmax):
                for j in range(self.tmax):
                    self.mpcholesky[i, j] = mpf(str(self.cholesky[i + 1][j + 1]))

    def fill_mp_sample_NOSHIFT(self):
        for n in range(self.nms):
            for i in range(self.tmax):  # tmax = T/2 if folded otherwise T-1
                self.mpsample[n, i] = mpf(str(self.sample[n][i]))
        #   Get cov for B matrix
        self.mpcov = mp.matrix(self.tmax)
        for i in range(self.tmax):
            self.mpcentral[i] = self.central[i]
            for j in range(self.tmax):
                self.mpcov[i, j] = mpf(str(self.cov[i][j]))

    def plot(self, show=True, logscale=True, label=None, yscale=1):
        plt.grid(alpha=0.1)
        if logscale is True:
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
            color="b",
        )
        plt.tight_layout()
        if label is not None:
            plt.legend()
        if show is True:
            plt.show()


def read_datafile(datapath_, sample_type="montecarlo"):  # (filename_, directory_):
    """
    You should write your own, compatible with your format!
    Here we assume that the input file has a header with time_extent and number of measurements.
    then data config by config. Example:
    #   32  100
    #   0   corr[0]
    #   1   corr[1]
    #   ...
    #   31  corr[31]
    #   0   corr[0]
    #   ... so on

    :param sample_type: rows are raw, independent measurements ("montecarlo i.e. not resampled" by default), or samples
        already resampled elsewhere ("bootstrap" or "jackknife"). See
        Obs.sample_type / common._variance_scale_factor: this determines how
        the error on the correlator (and downstream quantities, e.g. the
        smeared spectral density) is computed, so it must match how the rows were
        actually produced.
    """
    with open(datapath_, "r") as file:
        header = next(file).strip()
        print(LogMessage(), "Reading file :::", "Header: ", header)
        header_nms = int(header.split(" ")[0])
        header_T = int(header.split(" ")[1])
        print(LogMessage(), "Reading file :::", "Time extent ", header_T)
        print(LogMessage(), "Reading file :::", "Measurements ", header_nms)
        mcorr_ = Obs(
            T=header_T,
            tmax=header_T - 1,
            nms=header_nms,
            sample_type=sample_type,
        )
        # loop over file: read and store
        for indx, lndex in enumerate(file):
            # Read and store
            t = int(lndex.split(" ")[0])
            n = int(indx / header_T)
            # print(l.rstrip(), "     ", t, n)
            mcorr_.sample[n, t] = float(lndex.split(" ")[1])
    #   Returns np array of correlators
    return mcorr_, header_T, header_nms


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
        self.tmax = 0
        self.datapath = "None"
        self.outdir = "None"
        self.logpath = "None"
        self.plotpath = "None"
        self.num_boot = 1
        self.num_samples = 1
        self.sigma = 0
        self.emax = -1
        self.Ne = 1
        self.Na = 1
        self.emin = 0
        self.e0 = 0
        self.periodicity = "EXP"
        self.A0cut = 0
        # self.l = -1
        self.prec = -1
        self.mpsigma = mpf("0")
        self.mpemax = mpf("0")
        self.mpemin = mpf("0")
        self.mpe0 = mpf("0")
        self.mplambda = mpf("0")
        self.directoryName = "."
        self.kerneltype = "FULLNORMGAUSS"
        self.loglevel = "WARNING"

    def assign_values(self):
        """
        Assigns tmax based on time_extent and periodicity if tmax was not specified
        Creates mpf(var) from float type var
        """
        if self.tmax == 0:
            if self.periodicity == "EXP":
                self.tmax = self.time_extent - 1  # Can't use c[0]
            elif self.periodicity == "COSH":
                self.tmax = int(
                    self.time_extent / 2
                )  # Can't use C[0] but can use c[T/2]
        if self.periodicity == "EXP":
            assert (self.tmax) < self.time_extent
        if self.periodicity == "COSH":
            assert (self.tmax) < self.time_extent / 2 + 1
        self.mpsigma = mpf(str(self.sigma))
        self.mpemax = mpf(str(self.emax))
        self.mpemin = mpf(str(self.emin))
        self.mpe0 = mpf(str(self.e0))
        self.directoryName = (
            "tmax"
            + str(self.tmax)
            + "sigma"
            + str(self.sigma)
            + "Ne"
            + str(self.Ne)
            + "nboot"
            + str(self.num_boot)
            + "prec"
            + str(self.prec)
            + "Na"
            + str(self.Na)
            + "KerType"
            + str(self.kerneltype)
        )

    def apply_loglevel(self):
        """
        Applies self.loglevel to the shared "log" logger, so that log() calls
        (e.g. in hlt_stability.py) are actually emitted at that level. Split
        out of init() so that scripts which build their own output paths
        (rather than calling init()) can still opt in to this.
        """
        if self.loglevel == "INFO":
            logger.setLevel(logging.INFO)
        elif self.loglevel == "DEBUG":
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

    def init(self):
        self.assign_values()
        init_precision(self.prec)
        self.plotpath, self.logpath = create_out_paths(self)
        self.apply_loglevel()

    def report(self):
        print(LogMessage(), "Init ::: ", "Reading file:", self.datapath)
        print(LogMessage(), "Init ::: ", "Output directory:", self.outdir)
        print(LogMessage(), "Init ::: ", "Log directory:", self.logpath)
        print(LogMessage(), "Init ::: ", "Plot directory:", self.plotpath)
        print(LogMessage(), "Init ::: ", "Periodicity:", self.periodicity)
        print(LogMessage(), "Init ::: ", "Time extent:", self.time_extent)
        print(LogMessage(), "Init ::: ", "Smearing Kernel", self.kerneltype)
        print(LogMessage(), "Init ::: ", "tmax:", self.tmax)
        print(
            LogMessage(), "Init ::: ", "sigma (mp):", self.sigma, "(", self.mpsigma, ")"
        )
        print(LogMessage(), "Init ::: ", "Samples :", self.num_samples)
        print(LogMessage(), "Init ::: ", "Bootstrap samples :", self.num_boot)
        print(LogMessage(), "Init ::: ", "Number of energies :", self.Ne)
        print(
            LogMessage(),
            "Init ::: ",
            "Emax (mp)",
            self.emax,
            self.mpemax,
        )
        print(
            LogMessage(),
            "Init ::: ",
            "Emin (mp)",
            self.emin,
            "(",
            self.mpemin,
            ")",
        )
        print(LogMessage(), "Init ::: ", "Number of alphas", self.Na)
        print(LogMessage(), "Init ::: ", "Minimum value of A/A0 accepted ", self.A0cut)
        print(LogMessage(), "Init :::", "A integral from E0 = ", float(self.mpe0))


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


CB_colors = [
    "#1f77b4",  # Dark Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Light Green
    "#d62728",  # Reddish Purple
    "#9467bd",  # Light Blue
    "#8c564b",  # Dark Yellow
    "#e377c2",  # Cyan
    "#7f7f7f",  # Olive Green
]

plot_markers = ["o", "s", "D", "v", "^", "p", "*", "h"]
