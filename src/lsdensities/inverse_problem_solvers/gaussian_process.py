import logging
import os
import time

import numpy as np
from mpmath import mp, mpf

from ..abw import gAg
from ..core import integrandSigmaMat
from .stability_analysis import (
    A0_t,
    AlgorithmParameters,
    AlphaChannel,
    are_ranges_compatible,
    save_stability_output,
    scan_secondary_channels,
)
from ..transform import (
    coefficients_ssd,
    combine_fMf_scalar,
    combine_likelihood,
    get_ssd_averaged_scalar,
)
from ..utils.math_utils import invert_matrix_ge
from ..utils.common import Inputs, Obs, bcolors, log

__all__ = ["AlgorithmParameters", "A0_t", "SigmaMatrix", "GaussianProcessWrapper"]

# # # # # # # # # # # #
# Still needs a lot of work, not a state-of-art gaussian process implementation!
# # # # # # # # # # # #

class SigmaMatrix:
    """
    The Gaussian-process analogue of the Backus-Gilbert matrix A_N: evaluated by
    numerical quadrature under a Gaussian-process prior (as opposed to
    HLTWithBackusGilbert's SigmaMatrix, which has a closed form). Can also be
    read back from the file it writes out, to avoid recomputing an expensive
    quadrature across repeated runs.
    """

    def __init__(self, par: Inputs, alphaMP=0):
        self.par = par
        self.tmax = par.tmax
        self.alpha = alphaMP
        self.matrix = mp.matrix(par.tmax, par.tmax)

    def _file_name(self):
        return (
            "SMat_Sigma"
            + str(self.par.sigma)
            + "Alpha"
            + str(self.alpha)
            + "Prec"
            + str(self.par.prec)
            + "tmax"
            + str(self.par.tmax)
            + ".txt"
        )

    def evaluate(self):
        log(
            "Computing b_t(E) K(E,E') b_r(E') for Alpha = {:2.2f}".format(self.alpha),
        )
        with open(os.path.join(self.par.logpath, self._file_name()), "w") as output:
            for i in range(self.tmax):
                for j in range(self.tmax):
                    entry = mp.quad(
                        lambda x: integrandSigmaMat(
                            x,
                            # Round trip via a string to avoid introducing spurious precision
                            # per recommendations at https://mpmath.org/doc/current/basics.html
                            mpf(str(self.alpha)),
                            mpf(str(self.par.sigma)),
                            mpf(i + 1),
                            mpf(j + 1),
                            mpf(str(self.par.e0)),
                            self.par,
                        ),
                        [self.par.e0, mp.inf],
                        error=True,
                        method="tanh-sinh",
                    )
                    log("\t (t,r) = (", i, j, ") = ", entry[0])
                    print(i, j, entry[0], file=output)
                    self.matrix[i, j] = entry[0]

    def read(self):
        path_to_matrix = os.path.join(self.par.logpath, self._file_name())
        log("Reading Sigma Matrix from file: ", self._file_name())
        with open(path_to_matrix, "r") as file:
            for line in file:
                a, b, c = line.split()
                self.matrix[int(a), int(b)] = mpf(str(c))


class GaussianProcessWrapper:
    def __init__(
        self,
        par: Inputs,
        algorithmPar: AlgorithmParameters,
        B: mp.matrix,
        bnorm,
        correlator: Obs,
        energies,
        read_SIGMA=False,
    ):
        self.par = par
        self.correlator = correlator
        self.algorithmPar = algorithmPar
        self.B = B
        self.bnorm = bnorm
        par.Ne = len(energies)
        self.espace = energies
        self.read_SIGMA = read_SIGMA
        # Round trip via a string to avoid introducing spurious precision
        # per recommendations at https://mpmath.org/doc/current/basics.html
        self.e0MP = mpf(str(par.e0))
        self.espaceMP = mp.matrix(par.Ne, 1)
        self.sigmaMP = mpf(str(par.sigma))
        self.emaxMP = mpf(str(par.emax))
        self.eminMP = mpf(str(par.emin))
        self.espace_dictionary = {}  #   Usage: espace_dictionary[espace[n]] = n

        #   Alpha channels: "A" always present, "B" and "C" enabled via par.Na.
        #   "A" drives the plateau search; "B"/"C" only cross-check it (Fig. 6, top panel).
        self.selectA0 = {}  #   Usage: selectA0[alpha] = A0_t
        self.selectSigmaMat = {}  #   Usage: selectSigmaMat[alpha] = SigmaMatrix
        self.channels = {}
        self.secondary_channels = []

        def _add_channel(label, alpha_mp, alpha_float):
            a0 = A0_t(alpha=alpha_mp, emin=self.eminMP, par=self.par)
            sigma_matrix = SigmaMatrix(self.par, alpha_float)
            channel = AlphaChannel(label, alpha_mp, a0, sigma_matrix, self.par.Ne)
            self.channels[label] = channel
            self.selectA0[alpha_float] = a0
            self.selectSigmaMat[alpha_float] = sigma_matrix
            return channel

        self.channelA = _add_channel("A", algorithmPar.alphaAmp, algorithmPar.alphaA)
        if self.par.Na > 1:
            self.channelB = _add_channel("B", algorithmPar.alphaBmp, algorithmPar.alphaB)
            self.secondary_channels.append(self.channelB)
            if self.par.Na > 2:
                self.channelC = _add_channel(
                    "C", algorithmPar.alphaCmp, algorithmPar.alphaC
                )
                self.secondary_channels.append(self.channelC)

        #   Backward-compatible flat aliases (used by plot_utils.py and external callers)
        self.A0_A, self.SigmaMatA = self.channelA.a0, self.channelA.sigma_matrix
        self.rho_list = self.channelA.rho_list
        self.errBoot_list = self.channelA.errBoot_list
        self.errBayes_list = self.channelA.errBayes_list
        self.likelihood_list = self.channelA.likelihood_list
        if "B" in self.channels:
            self.A0_B, self.SigmaMatB = self.channelB.a0, self.channelB.sigma_matrix
            self.rho_list_alphaB = self.channelB.rho_list
            self.errBoot_list_alphaB = self.channelB.errBoot_list
            self.likelihood_list_alphaB = self.channelB.likelihood_list
        if "C" in self.channels:
            self.A0_C, self.SigmaMatC = self.channelC.a0, self.channelC.sigma_matrix
            self.rho_list_alphaC = self.channelC.rho_list
            self.errBoot_list_alphaC = self.channelC.errBoot_list
            self.likelihood_list_alphaC = self.channelC.likelihood_list

        self.lambda_list = [[] for _ in range(self.par.Ne)]

        #   Results
        self.minNLL = np.zeros(self.par.Ne)  # minimum of Negative Log Likelihood
        self.lambdaResultHLT = np.zeros(self.par.Ne)  #   from plateau in lambda
        self.lambdaResultBayes = np.zeros(self.par.Ne)  #   from min of NLL
        self.rhoResultHLT = np.zeros(self.par.Ne)  #   from plateau in lambda
        self.drho_result = np.zeros(self.par.Ne)  #   from plateau in lambda
        self.rhoResultBayes = np.zeros(self.par.Ne)  #   from min of NLL
        self.drho_bayes = np.zeros(self.par.Ne)  #   from min of NLL
        self.rho_sys_err_HLT = np.zeros(self.par.Ne)
        self.rho_quadrature_err_HLT = np.zeros(self.par.Ne)
        self.rho_sys_err_Bayes = np.zeros(self.par.Ne)
        self.rho_quadrature_err_Bayes = np.zeros(self.par.Ne)
        self.gt_HLT = [None] * self.par.Ne  #   from plateau in lambda
        self.gt_Bayes = [None] * self.par.Ne  #   from min of NLL
        self.aa0 = np.zeros(self.par.Ne)  # A / A0 for the result (HLT only)
        #   Control variables
        self.espace_is_filled = False
        self.A0_is_filled = False
        self.result_is_filled = np.full(par.Ne, False, dtype=bool)

    def fillEspaceMP(self):
        for e_id in range(self.par.Ne):
            self.espaceMP[e_id] = mpf(str(self.espace[e_id]))
            self.espace_dictionary[self.espace[e_id]] = e_id
        self.espace_is_filled = True

    def prepareGP(self):
        self.fillEspaceMP()
        for channel in [self.channelA, *self.secondary_channels]:
            channel.a0.evaluate(self.espaceMP)
            if self.read_SIGMA:
                channel.sigma_matrix.read()
            else:
                channel.sigma_matrix.evaluate()
        self.A0_is_filled = True

    def _store(self, estar, rho, errBayes, errBoot, likelihood, channel):
        """Appends the result at this (estar, lambda, alpha=channel) to the channel's in-memory history."""
        idx = self.espace_dictionary[estar]
        channel.rho_list[idx].append(rho)
        channel.errBayes_list[idx].append(errBayes)
        channel.errBoot_list[idx].append(errBoot)
        channel.likelihood_list[idx].append(likelihood)

    def lambdaToRho(self, lambda_, estar_, alpha_):
        """
        For a given lambda, at each energy and value of alpha
        computes and returns the following
            rho_estar : the result for the smeared spectral density
            drho_estar_Bayes : Bayesian error
            drho_estar_Bootstrap : frequentist error
            likelihood_estar : likelihood of the result
            gAg_estar : the scalar product gAg
            _g_t_estar : the vector of coefficients giving the result

        The coefficients are obtained by inverting the matrix
        S + factor B
        where factor = lambda A0 / Bnorm.
        Bnorm makes B dimensionless.
        """
        _Bnorm = self.bnorm / (estar_ * estar_)
        a0_value = self.selectA0[float(alpha_)].valute_at_E_dictionary[estar_]
        _factor = (lambda_ * a0_value) / _Bnorm
        log("Normalising factor A*l/B = {:2.2e}".format(float(_factor)))

        S = self.selectSigmaMat[float(alpha_)].matrix
        _Matrix = S + (_factor * self.B)
        start_time = time.time()
        _MatrixInv = invert_matrix_ge(_Matrix)
        end_time = time.time()
        log(
            "Time ::: Matrix inverted in {:4.4f}".format(end_time - start_time),
            "s",
        )
        start_time = time.time()
        _g_t_estar = coefficients_ssd(_MatrixInv, self.par, estar_, alpha=alpha_)
        end_time = time.time()
        log(
            "Time ::: Coefficients computed in {:4.4f}".format(end_time - start_time),
            "s",
        )
        rho_estar, drho_estar_Bootstrap = get_ssd_averaged_scalar(
            _g_t_estar,
            self.correlator.mpsample,
            self.par,
            sample_type=self.correlator.sample_type,
        )
        start_time = time.time()
        log(
            "Time ::: Bootstrapped result in {:4.4f}".format(start_time - end_time),
            "s",
        )
        gAg_estar = gAg(S, _g_t_estar, estar_, alpha_, self.par)

        varianceRho = combine_fMf_scalar(
            gt=_g_t_estar, params=self.par, estar=estar_, alpha=alpha_
        )
        log("\t\t gt ft = ", float(varianceRho))

        assert (
            self.par.kerneltype == "FULLNORMGAUSS"
        ), "Gaussian Process only admit FULLNORMGAUSS as a prior. Consider using HLTWithBackusGilbert, or implement your prior."
        _prior_diag = 1 / (np.sqrt(2 * np.pi) * self.par.sigma)
        _prior_diag *= mp.exp(alpha_ * estar_)
        varianceRho = mp.fsub(_prior_diag, varianceRho)
        log("\t\t exp(alpha E) - gt ft (E) = {:2.2e}".format(float(varianceRho)))
        varianceRho = mp.fdiv(varianceRho, _factor)
        varianceRho = mp.fdiv(varianceRho, mpf(2))
        drho_estar_Bayes = mp.sqrt(abs(varianceRho))

        log(
            "\t \t lambdaToRho ::: Central Value = {:2.4e}".format(float(rho_estar)),
        )
        log(
            "\t \t lambdaToRho ::: Bayesian Error = {:2.4e}".format(
                float(drho_estar_Bayes)
            ),
        )
        log(
            "\t \t lambdaToRho ::: Bootstrap Error   = {:2.4e}".format(
                float(drho_estar_Bootstrap)
            ),
        )

        fullCov_inv = _MatrixInv * _factor

        #   Compute the likelihood
        likelihood_estar = combine_likelihood(
            fullCov_inv, self.par, self.correlator.mpcentral
        )
        likelihood_estar *= 0.5
        det = mp.det(_Matrix / _factor)
        likelihood_estar = mp.fadd(likelihood_estar, 0.5 * mp.log(det))
        likelihood_estar = mp.fadd(
            likelihood_estar, (self.par.tmax * mp.log(2 * mp.pi)) * 0.5
        )
        log(
            "\t \t lambdaToRho ::: NLL = {:2.4e}".format(float(likelihood_estar)),
        )

        return (
            rho_estar,
            drho_estar_Bayes,
            drho_estar_Bootstrap,
            likelihood_estar,
            gAg_estar,
            _g_t_estar,
        )

    # - - - - - - - - - - - - - - - Scan over parameters: Lambda, Alpha (optional) - - - - - - - - - - - - - - - #

    def scanParameters(self, estar_):
        """
        This function will scan over lambda and, if specified, alpha, until conditions are met
        The stopping conditions are one of the following:
            - compatibility is achieved for a subsequent number of values of lambda specified by self.algorithmPar.lambdaScanCap : this is the intended way.
            OR
            - Reaching self.algorithmPar.lambdaMin : if compatibility conditions were never met. Raises a Warning.
        The compatibility between results at different lambda (or alpha) is achieved when the following conditions are simultaneously met:
            - A/A0 < self.par.A0cut
            AND
            - rho(lambda) = rho(lambda') within N sigma, where N is given by self.par.comparisonRatio.
              Setting comparisonRatio=1 means results are considered compatible when their errorbands overlap.
              Using values smaller than 1 can be useful since rho at different values of lambda can be very correlated. The default value
              is 0.3 which has been set empirically.
            AND
            - rho(lambda , alpha) = rho(lambda, alpha') = rho(lambda, alpha'') within 1 sigma, if more values of alpha are used.
        The scan is done between lambdaMax and lambdaMin, decreasing lambda by a fixed step at each iteration.
        """
        channelA = self.channelA
        secondary_channels = self.secondary_channels
        lambda_ = self.algorithmPar.lambdaMax
        lambda_step = self.algorithmPar.lambdaStep
        _cap = self.algorithmPar.lambdaScanCap
        _resize = self.algorithmPar.resize
        _countPositiveResult = 0
        _compRatio = self.algorithmPar.comparisonRatio
        idx = self.espace_dictionary[estar_]

        log(" --- ")
        log("At Energy {:2.2e}".format(estar_))
        log(
            "Setting Lambda ::: Lambda (0,inf) = {:1.3e}".format(float(lambda_)),
        )
        log(
            "Setting Lambda ::: Lambda (0,1) = {:1.3e}".format(
                float(lambda_ / (1 + lambda_))
            ),
        )

        #   First runs on initial values, then loops

        self.lambda_list[idx].append(lambda_)

        _rho, _errBayes, _errBoot, _likelihood, _gAg, _gt = self.lambdaToRho(
            lambda_, estar_, channelA.alpha_mp
        )
        self._store(estar_, _rho, _errBayes, _errBoot, _likelihood, channelA)

        #   "previous" lambda-step result for each secondary channel, used to check
        #   that channel's own stability independently of the primary channel
        previous = {}
        for ch in secondary_channels:
            rho, errBayes, errBoot, likelihood, _, _ = self.lambdaToRho(
                lambda_, estar_, ch.alpha_mp
            )
            self._store(estar_, rho, errBayes, errBoot, likelihood, ch)
            previous[ch.label] = (rho, errBoot)

        #   Flag these until better results (Bayesian)
        minNLL, lambdaStarBayes, rhoBayes, drhoBayes, gtBAYES = (
            _likelihood,
            lambda_,
            _rho,
            _errBayes,
            _gt,
        )

        #   #   #   #   #   #   #   #   Loops over values of lambda #   #   #   #   #   #   #   #
        lambda_ -= lambda_step
        while _countPositiveResult < _cap and lambda_ > self.algorithmPar.lambdaMin:
            #   -   -   -   -   -   -
            log(
                "Setting Lambda ::: Lambda (0,inf) = {:1.3e}".format(float(lambda_)),
            )
            log(
                "Setting Lambda ::: Lambda (0,1) = {:1.3e}".format(
                    float(lambda_ / (1 + lambda_))
                ),
            )
            self.lambda_list[idx].append(lambda_)

            #   Flag these until better results (HLT). Inside the loop contrary to the Bayesian equivalent
            #   because _countPositiveResult can be set to zero inside the loop
            if _countPositiveResult == 0:
                lambdaStarHLT, rhoHLT, drhoHLT, gtHLT, gag_flag = (
                    lambda_,
                    _rho,
                    _errBoot,
                    _gt,
                    _gAg,
                )
            #   -   -   -   -   -   -

            (
                _rhoUpdated,
                _errBayesUpdated,
                _errBootUpdated,
                _likelihoodUpdated,
                _gAgUpdated,
                _gtUpdated,
            ) = self.lambdaToRho(lambda_, estar_, channelA.alpha_mp)
            self._store(
                estar_,
                _rhoUpdated,
                _errBayesUpdated,
                _errBootUpdated,
                _likelihoodUpdated,
                channelA,
            )

            _skip, flagged_overlap = scan_secondary_channels(
                self,
                lambda_,
                estar_,
                secondary_channels,
                previous,
                primary_rho=_rhoUpdated,
                primary_errBoot=_errBootUpdated,
                flagged_rho=rhoHLT,
                flagged_errBoot=drhoHLT,
                comp_ratio=_compRatio,
            )

            newLambda_Overlap = are_ranges_compatible(
                _rhoUpdated, _compRatio * _errBootUpdated, _rho, _compRatio * _errBoot
            )  # comparison with previous lambda

            flagLambda_Overlap = are_ranges_compatible(
                _rhoUpdated, _compRatio * _errBootUpdated, rhoHLT, _compRatio * drhoHLT
            )  # comparison with flagged lambda

            if _likelihoodUpdated < minNLL:  # NLL, if less than before; flag the results
                minNLL, lambdaStarBayes, rhoBayes, drhoBayes, gtBAYES = (
                    _likelihoodUpdated,
                    lambda_,
                    _rhoUpdated,
                    _errBayesUpdated,
                    _gtUpdated,
                )

            #   Checks if Rho at this lambda overlaps with Rho at flagged lambda
            if not newLambda_Overlap:
                log(
                    "\t Result at this Lambda does not overlap with previous: REJECTING result",
                )
                _skip = True
            if not flagLambda_Overlap:
                log(
                    "\t Result at this Lambda does not overlap with flagged: REJECTING result",
                )
                _skip = True
            #   Each secondary channel's agreement with the flagged (candidate plateau)
            #   result is checked independently, as prescribed by the docstring above.
            for ch in secondary_channels:
                if not flagged_overlap[ch.label]:
                    log(
                        f"\t Result at this Lambda, {ch.label} Alpha, does not overlap with flagged: REJECTING result",
                    )
                    _skip = True

            #   Checks A/A0 is acceptable
            aa0_updated = _gAgUpdated / channelA.a0.valute_at_E_dictionary[estar_]
            if aa0_updated > self.par.A0cut:
                log(
                    "\t A/A0 is too large: rejecting result  (",
                    float(aa0_updated),
                    ")",
                )
                _skip = True

            #   Having analysed all possible stopping conditions, proceed

            if not _skip:
                #   Flag the first compatible result, because it is the one with the smaller error
                if (
                    _countPositiveResult == self.algorithmPar.plateau_id
                ):  #   At future alphas we compare with rho_s at _countPositiveResult = 1. This can be changed.
                    lambdaStarHLT, rhoHLT, drhoHLT, gtHLT, gag_flag = (
                        lambda_,
                        _rho,
                        _errBoot,
                        _gtUpdated,
                        _gAgUpdated,
                    )
                _countPositiveResult += 1
                log(
                    f"{bcolors.OKGREEN}Stopping Condition{bcolors.ENDC}",
                    _countPositiveResult,
                    "/",
                    _cap,
                )
            else:
                _countPositiveResult = 0

            #   Update variables before restarting the loop
            _rho = _rhoUpdated
            _errBoot = _errBootUpdated
            lambda_ -= lambda_step
            #   Resize lambda_step
            if lambda_ <= 0:
                lambda_step /= _resize
                lambda_ += lambda_step * (_resize - 1 / _resize)
                log(
                    "Resize LambdaStep to ",
                    lambda_step,
                    "Setting Lambda = ",
                    lambda_,
                )

            if lambda_ < self.algorithmPar.lambdaMin:
                log(
                    f"{bcolors.WARNING}Warning{bcolors.ENDC} ::: Stopping ::: Reached lower limit for lambda. Try decreasing 'algorithmPar.lambdaMin' or increase the smearing radius.",
                    level=logging.WARNING,
                )

        #   End of WHILE
        if _countPositiveResult == 0:
            log(
                f"{bcolors.WARNING}WARNING{bcolors.ENDC} ::: Could NOT find a plateau in lambda",
                level=logging.WARNING,
            )

        #   hlt
        self.lambdaResultHLT[idx] = lambdaStarHLT
        self.rhoResultHLT[idx] = rhoHLT
        self.drho_result[idx] = drhoHLT
        self.gt_HLT[idx] = gtHLT
        self.aa0[idx] = gag_flag
        #   bayesian
        self.minNLL[idx] = minNLL
        self.lambdaResultBayes[idx] = lambdaStarBayes
        self.rhoResultBayes[idx] = rhoBayes
        self.drho_bayes[idx] = drhoBayes
        self.gt_Bayes[idx] = gtBAYES

        return (
            lambdaStarHLT,
            rhoHLT,
            drhoHLT,
            minNLL,
            lambdaStarBayes,
            rhoBayes,
            drhoBayes,
            gtHLT,
            gtBAYES,
            gag_flag,
        )

    def estimate_sys_error(self, e_i):
        _this_y_HLT = self.rhoResultHLT[e_i]  # rho at lambda*
        _that_y_HLT, _, _, _, _, _ = self.lambdaToRho(
            self.lambdaResultHLT[e_i] * self.algorithmPar.kfactor,
            self.espace[e_i],
            self.channelA.alpha_mp,
        )

        _this_y_Bayes = self.rhoResultBayes[e_i]  # rho at lambda*
        _that_y_Bayes, _, _, _, _, _ = self.lambdaToRho(
            self.lambdaResultBayes[e_i] * self.algorithmPar.kfactor,
            self.espace[e_i],
            self.channelA.alpha_mp,
        )

        self.rho_sys_err_HLT[e_i] = abs(_this_y_HLT - _that_y_HLT) / 2
        self.rho_quadrature_err_HLT[e_i] = np.sqrt(
            self.rho_sys_err_HLT[e_i] ** 2 + self.drho_result[e_i] ** 2
        )

        self.rho_sys_err_Bayes[e_i] = abs(_this_y_Bayes - _that_y_Bayes) / 2
        self.rho_quadrature_err_Bayes[e_i] = np.sqrt(
            self.rho_sys_err_Bayes[e_i] ** 2 + self.drho_bayes[e_i] ** 2
        )

        return self.rho_sys_err_HLT[e_i], self.rho_sys_err_Bayes[e_i]

    def run(self):
        for e_i in range(self.par.Ne):
            self.scanParameters(self.espace[e_i])
            self.estimate_sys_error(e_i)
        return 0

    def save(self, path=None):
        """
        Writes the full stability-analysis scan and results to a single JSON
        file (see stability_analysis.save_stability_output / io_utils.py for the
        schema). Use examples/plot_output.py to plot from it.
        """
        return save_stability_output(self, "GP", path)
