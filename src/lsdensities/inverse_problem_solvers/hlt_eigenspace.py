import logging

from ..core import cauchy_matrix
from ..transform import ft_mp
from ..utils.stat_utils import averageVector_fp
from mpmath import mp, mpf
import numpy as np
from ..utils.common import Inputs, Obs, log
from .. import io_utils


class HETpar:
    """
    alphaA, alphaB, alphaC : values of the smearing-kernel exponent used to
        cross-check the eigen-space truncation (mirrors the stability
        analysis' use of several alpha, arXiv:2605.14652 Fig. 6/7). Only
        alphaA is required; set par.Na to 1, 2 or 3 to also use alphaB/alphaC.
    n_consecutive : number of consecutive eigenmodes whose contribution to
        rho must be compatible with zero before the sum is truncated there
        (arXiv:2605.14652 Eq. 34).
    """

    def __init__(
        self,
        alphaA=0,
        alphaB=1 / 2,
        alphaC=1.99,
        n_consecutive=2,
    ):
        assert alphaA != alphaB
        assert alphaA != alphaC
        self.alphaA = float(alphaA)
        self.alphaB = float(alphaB)
        self.alphaC = float(alphaC)
        self.n_consecutive = n_consecutive


class SigmaMatrix:
    """
    Eigen-basis representation of the Backus-Gilbert matrix A_N (arXiv:2605.14652,
    Sec. III.B): Htild = Bh . A_N . Bh^T (Bh whitens the data covariance) is
    symmetric by construction, so its eigendecomposition is real and orthogonal.
    """

    def __init__(self, par: Inputs, Bh, alphaMP=0.0):
        self.par = par
        self.tmax = par.tmax
        self.alpha = alphaMP
        self.Bh = Bh

        self.matrix = mp.matrix(self.tmax, self.tmax)
        self.Htild = None
        self.eigvals = None
        self.eigvecs = None

    def _compute_eigendecomposition(self):
        eigvals, eigvecs = mp.eigsy(self.Htild)

        idx = sorted(range(len(eigvals)), key=lambda i: eigvals[i], reverse=True)
        eigvals = [eigvals[i] for i in idx]

        eigvecs_sorted = mp.zeros(self.tmax, self.tmax)
        for new_col, old_col in enumerate(idx):
            for row in range(self.tmax):
                eigvecs_sorted[row, new_col] = eigvecs[row, old_col]

        self.eigvals = eigvals
        self.eigvecs = eigvecs_sorted

    def evaluate(self):
        log(" Saving H Matrix ")
        self.matrix = cauchy_matrix(
            tmax=self.par.tmax,
            alpha=self.alpha,
            e0=self.par.mpe0,
            type=self.par.periodicity,
            T=self.par.time_extent,
        )
        log(" Building Htild")
        self.Htild = self.Bh * self.matrix * self.Bh.T
        log(" Computing eigendecomposition")
        self._compute_eigendecomposition()
        log(" Done!")


class EigenspaceChannel:
    """Bookkeeping for one value of the kernel exponent alpha in the eigen-space analysis."""

    def __init__(self, label: str, alpha: float, num_energies: int):
        self.label = label
        self.alpha = alpha
        self.sigma_matrix = None  # set by HLTWithSVD.prepare()
        self.kstop = np.zeros(num_energies)
        self.res = np.zeros(num_energies)
        self.err = np.zeros(num_energies)


class HLTWithSVD:
    def __init__(
        self,
        par: Inputs,
        algorithmPar: HETpar,
        correlator: Obs,
        energies,
        l_reg,
        useCOV=True,
    ):
        self.par = par
        self.correlator = correlator
        self.algorithmPar = algorithmPar
        self.Bh = mp.matrix(self.par.tmax, self.par.tmax)
        self.l_reg = l_reg
        self.par.Ne = len(energies)
        self.espace = energies
        self.useCOV = useCOV
        # Round trip via a string to avoid introducing spurious precision
        # per recommendations at https://mpmath.org/doc/current/basics.html
        self.e0MP = mpf(str(par.e0))
        self.espaceMP = mp.matrix(par.Ne, 1)
        self.sigmaMP = mpf(str(par.sigma))
        self.emaxMP = mpf(str(par.emax))
        self.eminMP = mpf(str(par.emin))
        self.espace_dictionary = {}  #   Usage: espace_dictionary[espace[n]] = n

        #   Alpha channels: "A" always present, "B" and "C" enabled via par.Na.
        self.channels = {}
        self.secondary_channels = []

        def _add_channel(label, alpha):
            channel = EigenspaceChannel(label, alpha, self.par.Ne)
            self.channels[label] = channel
            return channel

        self.channelA = _add_channel("A", self.algorithmPar.alphaA)
        if self.par.Na > 1:
            self.channelB = _add_channel("B", self.algorithmPar.alphaB)
            self.secondary_channels.append(self.channelB)
            if self.par.Na > 2:
                self.channelC = _add_channel("C", self.algorithmPar.alphaC)
                self.secondary_channels.append(self.channelC)

        #   Backward-compatible flat aliases
        self.kstopA, self.resA, self.errA = (
            self.channelA.kstop,
            self.channelA.res,
            self.channelA.err,
        )
        if "B" in self.channels:
            self.kstopB, self.resB, self.errB = (
                self.channelB.kstop,
                self.channelB.res,
                self.channelB.err,
            )
        if "C" in self.channels:
            self.kstopC, self.resC, self.errC = (
                self.channelC.kstop,
                self.channelC.res,
                self.channelC.err,
            )

        #   Control variables
        self.espace_is_filled = False
        self.result_is_filled = np.full(par.Ne, False, dtype=bool)
        self._scan_data = None  # filled by run(), consumed by save()

    def _fillEspaceMP(self):
        """
        Fill array of energies : array[int] = energy
        and provides a dictionary such that dictionary[array[int]] = int
        """
        for e_id in range(self.par.Ne):
            self.espaceMP[e_id] = mpf(str(self.espace[e_id]))
            self.espace_dictionary[self.espace[e_id]] = e_id
        self.espace_is_filled = True

    def prepare(self):
        self._fillEspaceMP()
        self.Bh = (
            self.correlator.mpcholesky ** (-1) if self.useCOV else mp.eye(self.par.tmax)
        )
        for channel in [self.channelA, *self.secondary_channels]:
            channel.sigma_matrix = SigmaMatrix(self.par, self.Bh, channel.alpha)
            channel.sigma_matrix.evaluate()

    def eigProject(self, lambda_, estar_, channel):
        tmax = self.par.tmax
        nboot = self.par.num_boot

        eigvecs = channel.sigma_matrix.eigvecs
        eigvals = channel.sigma_matrix.eigvals

        f = mp.matrix(tmax, 1)
        for i in range(tmax):
            f[i] = ft_mp(
                e=mpf(str(estar_)),
                t=mpf(i + 1),
                sigma_=self.par.mpsigma,
                alpha=mpf(str(channel.alpha)),
                e0=self.par.mpe0,
                type=self.par.periodicity,
                T=mpf(str(self.par.time_extent)),
                ker_type=self.par.kerneltype,
            )
        ftild = self.Bh * f
        f_proj_mp = eigvecs.T * ftild
        f_proj = np.array(f_proj_mp)
        eigvals = np.array(eigvals)

        contrib_k_mean = np.zeros(tmax)
        contrib_k_err = np.zeros(tmax)
        cumulative_k_mean = np.zeros(tmax)
        cumulative_k_err = np.zeros(tmax)
        beta_jk = mp.matrix(nboot, tmax)

        for j in range(nboot):
            cj = self.correlator.mpsample[j, :].T
            ctild = self.Bh * cj
            beta = eigvecs.T * ctild  # beta, vector of length tmax
            beta_jk[j, :] = beta.T  # beta, double array nms, tmax

        cumulative_samples = np.zeros(nboot)
        for k in range(tmax):
            bk = np.array(beta_jk[:, k])  # beta[k], array containing stats

            contrib_k_samples = (
                bk * f_proj[k] / (eigvals[k] + lambda_)
            )  # a b / mu + l, stat samples

            cumulative_samples = cumulative_samples + contrib_k_samples
            contrib_k_mean[k], contrib_k_err[k] = averageVector_fp(
                contrib_k_samples, get_var=True
            )
            cumulative_k_mean[k], cumulative_k_err[k] = averageVector_fp(
                cumulative_samples, get_var=True
            )

        return contrib_k_mean, contrib_k_err, cumulative_k_mean, cumulative_k_err, f_proj_mp

    def _effective_gt(self, channel, f_proj_mp, kstar, lambda_):
        """
        The coefficient vector g_t (same t-indexing as HLT/GP's gt: index i
        pairs with physical time t=i+1) that reproduces this channel's
        kstar-truncated reconstruction as a plain dot product with the
        correlator's central value -- i.e. the effective smearing kernel
        actually realised by this eigenspace-truncated result. Derived from
        cumulative = sum_{k<=kstar} beta_jk[k] f_proj[k]/(eigvals[k]+lambda_),
        beta = eigvecs.T Bh c, by swapping the k- and t-sums: gt = Bh.T eigvecs w,
        w[k] = f_proj[k]/(eigvals[k]+lambda_) for k<=kstar, else 0.
        """
        tmax = self.par.tmax
        eigvals = channel.sigma_matrix.eigvals
        eigvecs = channel.sigma_matrix.eigvecs
        w = mp.matrix(tmax, 1)
        for k in range(kstar + 1):
            w[k] = f_proj_mp[k] / (eigvals[k] + lambda_)
        return self.Bh.T * (eigvecs * w)

    def which_k_saturates(self, contrib_k_mean, contrib_k_err):
        counter = 0
        for k in range(self.par.tmax):
            if abs(contrib_k_mean[k]) < contrib_k_err[k]:
                log(rf"{k}-th term compatible with zero")
                counter += 1
                if counter >= self.algorithmPar.n_consecutive:
                    log(rf"Result found at k* = {k}")
                    return k
        log(
            rf"Failure: Eigenvalues did not saturate {self.algorithmPar.n_consecutive} times. Using all values.",
            level=logging.WARNING,
        )
        return self.par.tmax - 1

    def run(self):
        channels = [self.channelA, *self.secondary_channels]
        self._scan_data = []

        for e_i in range(self.par.Ne):
            energy = self.espace[e_i]
            scan = {}
            result = {}

            for channel in channels:
                (
                    contrib_k_mean,
                    contrib_k_err,
                    cumulative_k_mean,
                    cumulative_k_err,
                    f_proj_mp,
                ) = self.eigProject(self.l_reg, energy, channel)

                kstar = self.which_k_saturates(contrib_k_mean, contrib_k_err)

                channel.kstop[e_i] = kstar
                channel.res[e_i] = cumulative_k_mean[kstar]
                channel.err[e_i] = cumulative_k_err[kstar]

                gt = self._effective_gt(channel, f_proj_mp, kstar, self.l_reg)

                scan[channel.label] = {
                    "k": list(range(self.par.tmax)),
                    "contrib_mean": contrib_k_mean.tolist(),
                    "contrib_err": contrib_k_err.tolist(),
                    "cumulative_mean": cumulative_k_mean.tolist(),
                    "cumulative_err": cumulative_k_err.tolist(),
                }
                result[channel.label] = {
                    "kstar": int(kstar),
                    "res": float(channel.res[e_i]),
                    "err": float(channel.err[e_i]),
                    "gt": [float(x) for x in gt],
                }

            self._scan_data.append(
                {"energy": float(energy), "scan": scan, "result": result}
            )

    def save(self, path=None):
        """
        Writes the eigen-space analysis (arXiv:2605.14652 Fig. 7: the
        cumulative contribution to rho as eigenmodes of A_N are added, for
        every alpha channel) to a single JSON file (see io_utils.py for the
        shared schema). Use examples/plot_output.py to plot from it.
        """
        if self._scan_data is None:
            raise RuntimeError("Nothing to save: call run() first.")

        channels = [self.channelA, *self.secondary_channels]
        metadata = io_utils.base_metadata(self.par, "EigenspaceAnalysis")
        metadata["alphas"] = {ch.label: ch.alpha for ch in channels}
        metadata["algorithm"] = {
            "l_reg": float(self.l_reg),
            "n_consecutive": self.algorithmPar.n_consecutive,
            "useCOV": self.useCOV,
        }

        path = path or io_utils.default_output_path(self.par, metadata["method"])
        return io_utils.write_json(path, metadata, self._scan_data)
