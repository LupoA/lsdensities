from .core import hlt_matrix
from .transform import ft_mp
from .utils.rhoStat import averageVector_fp
from mpmath import mp, mpf
import numpy as np
from .utils.rhoUtils import Inputs, Obs, log
import os
import json


class HETpar:
    """ """

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
    def __init__(self, par: Inputs, Bh, alphaMP=0.0):
        self.par = par
        self.tmax = par.tmax
        self.alpha = alphaMP
        self.Bh = Bh

        self.matrix = mp.matrix(self.tmax, self.tmax)
        self.inverse = mp.matrix(self.tmax, self.tmax)

        self.Htild = None
        self.eigvals = None
        self.eigvecs = None

    def _compute_eigendecomposition(self):
        eigvals, eigvecs = mp.eig(self.Htild)

        if 0:
            print("\n--- Eigenvalues debug ---")
            for i, ev in enumerate(eigvals):
                try:
                    re = mp.re(ev)
                    im = mp.im(ev)
                    print(f"{i}: real={re}, imag={im}, |imag|={abs(im)}\n")
                except Exception:
                    # in case it's already real (mpf)
                    print(f"{i}: real={ev}, imag=0\n")

            print("------------------------\n")

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
        self.matrix = hlt_matrix(
            tmax=self.par.tmax,
            alpha=self.alpha,
            e0=self.par.mpe0,
            type=self.par.periodicity,
            T=self.par.time_extent,
        )

        # log(" Evaluating H inverse")
        # self.inverse = invert_matrix_ge(self.matrix)

        log(" Building Htild")
        self.Htild = self.Bh * self.matrix * self.Bh.T

        log(" Computing eigendecomposition")
        self._compute_eigendecomposition()

        log(" Done!")


class HilbertEigTruncWrapper:
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
        self.selectSigmaMat = {}  #   Usage: selectSigmaMat[alpha] = Sigma
        #   First alpha
        self.SigmaMatA = None  # SigmaMatrix(self.par, algorithmPar.alphaA)
        self.kstopA = np.ndarray(self.par.Ne, dtype=np.float64)
        self.resA = np.ndarray(self.par.Ne, dtype=np.float64)
        self.errA = np.ndarray(self.par.Ne, dtype=np.float64)

        #   Second alpha
        if self.par.Na > 1:
            self.SigmaMatB = None  # SigmaMatrix(self.par, algorithmPar.alphaB)
            self.kstopB = np.ndarray(self.par.Ne, dtype=np.float64)
            self.resB = np.ndarray(self.par.Ne, dtype=np.float64)
            self.errB = np.ndarray(self.par.Ne, dtype=np.float64)
            #   Third alpha
            if self.par.Na > 2:
                self.SigmaMatC = None  # SigmaMatrix(self.par, algorithmPar.alphaC)
                self.kstopC = np.ndarray(self.par.Ne, dtype=np.float64)
                self.resC = np.ndarray(self.par.Ne, dtype=np.float64)
                self.errC = np.ndarray(self.par.Ne, dtype=np.float64)

        #   Control variables
        self.espace_is_filled = False
        self.result_is_filled = np.full(par.Ne, False, dtype=bool)
        self.log_path = "."
        # - - - - - - - - - - - - - - - End of INIT - - - - - - - - - - - - - - - #

    def _fillEspaceMP(self):
        """
        Fill array of energies : array[int] = energy
        and provides a dictionary such that dictionary[array[int]] = int
        """
        for e_id in range(self.par.Ne):
            self.espaceMP[e_id] = mpf(str(self.espace[e_id]))
            self.espace_dictionary[self.espace[e_id]] = e_id
        self.espace_is_filled = True
        return

    def prepare(self):
        self._fillEspaceMP()
        if self.useCOV:
            self.Bh = self.correlator.mpcholesky ** (-1)
        else:
            self.Bh = mp.eye(self.par.tmax)
        # First alpha
        self.SigmaMatA = SigmaMatrix(self.par, self.Bh, self.algorithmPar.alphaA)
        self.SigmaMatA.evaluate()
        self.selectSigmaMat[str(self.algorithmPar.alphaA)] = self.SigmaMatA

        # Second alpha
        if self.par.Na > 1:
            self.SigmaMatB = SigmaMatrix(self.par, self.Bh, self.algorithmPar.alphaB)
            self.SigmaMatB.evaluate()
            self.selectSigmaMat[str(self.algorithmPar.alphaB)] = self.SigmaMatB

            if self.par.Na > 2:
                self.SigmaMatC = SigmaMatrix(
                    self.par, self.Bh, self.algorithmPar.alphaC
                )
                self.SigmaMatC.evaluate()
                self.selectSigmaMat[str(self.algorithmPar.alphaC)] = self.SigmaMatC

        os.makedirs(self.par.logpath, exist_ok=True)
        self.log_path = os.path.join(
            self.par.logpath,
            rf"HET_tmax{self.par.tmax}_sigma{self.par.sigma}_Ne{self.par.Ne}_Na{self.par.Na}_kernel{self.par.kerneltype}.json",
        )
        # - - - - - - - - - - - - - - - Main functions - - - - - - - - - - - - - - - #

    def eigProject(self, lambda_, estar_, alpha_):
        tmax = self.par.tmax
        nboot = self.par.num_boot

        matrix = self.selectSigmaMat[str(alpha_)]

        eigvecs = matrix.eigvecs
        eigvals = matrix.eigvals

        # get f
        f = mp.matrix(tmax, 1)
        for i in range(tmax):
            f[i] = ft_mp(
                e=mpf(str(estar_)),
                t=mpf(i + 1),
                sigma_=self.par.mpsigma,
                alpha=mpf(str(alpha_)),
                e0=self.par.mpe0,
                type=self.par.periodicity,
                T=mpf(str(self.par.time_extent)),
                ker_type=self.par.kerneltype,
            )
        ftild = self.Bh * f
        f_proj = eigvecs.T * ftild

        contrib_k_mean = np.zeros(tmax)
        contrib_k_err = np.zeros(tmax)
        cumulative_k_mean = np.zeros(tmax)
        cumulative_k_err = np.zeros(tmax)
        beta_jk = mp.matrix(nboot, tmax)

        eigvals = np.array(eigvals)
        f_proj = np.array(f_proj)

        for j in range(nboot):
            cj = self.correlator.mpsample[j, :].T
            ctild = self.Bh * cj
            beta = eigvecs.T * ctild  # beta, vector of lenght tmax
            beta_jk[j, :] = beta.T  # beta, double array nms, tmax

        cumulative_samples = np.zeros(nboot)
        for k in range(tmax):
            bk = np.array(beta_jk[:, k])  # beta[k], array containing stats
            # beta_k_mean[k], beta_k_err[k] = averageVector_fp(bk, get_var=True)  # beta[k] mean and stdv

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

        if 0:  # for debug only
            import matplotlib.pyplot as plt

            plt.errorbar(
                x=(np.arange(tmax))[:16],
                y=cumulative_k_mean[:16],
                yerr=cumulative_k_err[:16],
                fmt="o-",
                ecolor="red",
                capsize=3,
                label="Cumulative Mean ± Error",
            )
            plt.xlabel("k")
            plt.ylabel("Cumulative Mean")
            plt.title(
                f"Cumulative contribution projection for alpha={alpha_}, estar={estar_}"
            )
            plt.grid(True)
            plt.legend()
            plt.show()

        return contrib_k_mean, contrib_k_err, cumulative_k_mean, cumulative_k_err

    def which_k_saturates(self, contrib_k_mean, contrib_k_err):
        counter = 0
        for k in range(self.par.tmax):
            if abs(contrib_k_mean[k]) < contrib_k_err[k]:
                log(rf"{k}-th term compatible with zero")
                counter += 1
                if counter >= self.algorithmPar.n_consecutive:
                    log(rf"Result found at k* = {k}")
                    return k
        print(
            rf"Failure: Eigenvalues did not saturate {self.algorithmPar.n_consecutive} times. Using all values."
        )
        return self.par.tmax - 1

    def run(self):
        alphas = [self.algorithmPar.alphaA]
        if self.par.Na > 1:
            alphas.append(self.algorithmPar.alphaB)
        if self.par.Na > 2:
            alphas.append(self.algorithmPar.alphaC)

        log_data = {"energies": [], "result": []}

        for e_i in range(self.par.Ne):
            energy = self.espace[e_i]

            energy_entry = {"energy": float(energy)}
            result_entry = {"energy": float(energy)}

            for idx, alpha in enumerate(alphas):
                label = chr(ord("A") + idx)  # 'A', 'B', 'C'

                (
                    contrib_k_mean,
                    contrib_k_err,
                    cumulative_k_mean,
                    cumulative_k_err,
                ) = self.eigProject(self.l_reg, energy, alpha)

                kstar = self.which_k_saturates(contrib_k_mean, contrib_k_err)

                # store results dynamically
                getattr(self, f"kstop{label}")[e_i] = kstar
                getattr(self, f"res{label}")[e_i] = cumulative_k_mean[kstar]
                getattr(self, f"err{label}")[e_i] = cumulative_k_err[kstar]

                energy_entry[label] = {
                    "contrib_k_mean": contrib_k_mean.tolist(),
                    "contrib_k_err": contrib_k_err.tolist(),
                    "cumulative_k_mean": cumulative_k_mean.tolist(),
                    "cumulative_k_err": cumulative_k_err.tolist(),
                }

                result_entry[label] = {
                    "kstar": int(kstar),
                    "res": float(getattr(self, f"res{label}")[e_i]),
                    "err": float(getattr(self, f"err{label}")[e_i]),
                }

            log_data["energies"].append(energy_entry)
            log_data["result"].append(result_entry)

        with open(self.log_path, "w") as f:
            json.dump(log_data, f, indent=4)
