from .core import A0E_mp, Smatrix_mp
from .transform import (
    h_Et_mp_Eslice,
    y_combine_sample_Eslice_mp,
    combine_fMf_Eslice,
    combine_likelihood,
)
from .abw import gAg
from mpmath import mp, mpf
from .plotutils import (
    stabilityPlot,
    sharedPlot_stabilityPlusLikelihood,
    plotLikelihood,
    plotAllKernels,
    plotSpectralDensity,
)
import numpy as np
from .utils.rhoUtils import LogMessage, Inputs, MatrixBundle, Obs, bcolors
from .utils.rhoMath import invert_matrix_ge
import os

_big = 100


class AlgorithmParameters:
    def __init__(
        self,
        alphaA=0,
        alphaB=1 / 2,
        alphaC=1.99,
        lambdaMax=50,  #   Starting lambda
        lambdaStep=0.5,  #   Step in lambda for the scan
        lambdaScanCap=6,  #   Search for plateau in lambda stops after lambdaScanCap subsequent compatible measurements. Affected by comparisonRatio.
        kfactor=0.1,  #   Systematic error estimated comparing result at lambda with result at kfactor*lambda
        lambdaMin=1e-6,  #   Stop scan at lambdaMin
        comparisonRatio=1,  #   Measurements at different lambdas will be considered compatible within comparisonRatio*sigma
        resize=4,  #   When lambda <=0 but lambda> lambdaMin: lambdaStep = lambdaStep /= resize and lambda += lambdaStep * (resize - 1 / resize). Allows probing lambda over different scales.
    ):
        assert alphaA != alphaB
        assert alphaA != alphaC
        self.alphaA = float(alphaA)
        self.alphaB = float(alphaB)
        self.alphaC = float(alphaC)
        self.lambdaMax = lambdaMax
        self.lambdaStep = lambdaStep
        self.lambdaScanCap = lambdaScanCap
        self.kfactor = kfactor
        # Round trip via a string to avoid introducing spurious precision
        # per recommendations at https://mpmath.org/doc/current/basics.html
        self.alphaAmp = mpf(str(alphaA))
        self.alphaBmp = mpf(str(alphaB))
        self.alphaCmp = mpf(str(alphaC))
        self.lambdaMin = lambdaMin
        self.comparisonRatio = comparisonRatio
        self.resize = resize


class A0_t:
    def __init__(self, par_: Inputs, alphaMP_=0, eminMP_=0):
        self.valute_at_E = mp.matrix(par_.Ne, 1)
        self.valute_at_E_dictionary = {}  # Auxiliary dictionary: A0espace[n] = A0espace_dictionary[espace[n]] # espace must be float
        self.is_filled = False
        self.alphaMP = alphaMP_
        self.eminMP = eminMP_
        self.par = par_

    def evaluate(self, espaceMP_):
        print(
            LogMessage(),
            "Computing A0 at all energies with Alpha = {:2.2e}".format(
                float(self.alphaMP)
            ),
        )
        self.valute_at_E = A0E_mp(
            espaceMP_, self.par, alpha_=self.alphaMP, e0_=self.par.mpe0
        )
        for e_id in range(self.par.Ne):
            self.valute_at_E_dictionary[float(espaceMP_[e_id])] = self.valute_at_E[e_id]
        self.is_filled = True


class SigmaMatrix:
    def __init__(self, par: Inputs, alphaMP=0):
        self.par = par
        self.tmax = par.tmax
        self.alpha = alphaMP
        self.matrix = mp.matrix(par.tmax, par.tmax)

    def evaluate(self):
        print(LogMessage(), " Saving Sigma Matrix ".format())
        self.matrix = Smatrix_mp(
            tmax_=self.par.tmax,
            alpha_=self.alpha,
            e0_=self.par.mpe0,
            type=self.par.periodicity,
            T=self.par.time_extent,
        )


class InverseProblemWrapper:
    def __init__(
        self,
        par: Inputs,
        algorithmPar: AlgorithmParameters,
        matrix_bundle: MatrixBundle,
        correlator: Obs,
        energies,
    ):
        self.par = par
        self.correlator = correlator
        self.algorithmPar = algorithmPar
        self.matrix_bundle = matrix_bundle
        print(LogMessage(), "InverseProblemWrapper ::: Reading input energies")
        par.Ne = len(energies)
        self.espace = energies
        # Round trip via a string to avoid introducing spurious precision
        # per recommendations at https://mpmath.org/doc/current/basics.html
        self.e0MP = mpf(str(par.e0))
        self.espaceMP = mp.matrix(par.Ne, 1)
        self.sigmaMP = mpf(str(par.sigma))
        self.emaxMP = mpf(str(par.emax))
        self.eminMP = mpf(str(par.emin))
        self.espace_dictionary = {}  #   Usage: espace_dictionary[espace[n]] = n
        self.selectSigmaMat = {}  #   Usage: selectSigmaMat[alpha] = Sigma
        #   Containers for the factor A0
        self.selectA0 = {}
        self.A0_A = A0_t(
            alphaMP_=self.algorithmPar.alphaAmp, eminMP_=self.eminMP, par_=self.par
        )
        self.selectA0[algorithmPar.alphaA] = self.A0_A
        if self.par.Na > 1:
            self.A0_B = A0_t(
                alphaMP_=self.algorithmPar.alphaBmp, eminMP_=self.eminMP, par_=self.par
            )
            self.selectA0[algorithmPar.alphaB] = self.A0_B
            if self.par.Na > 2:
                self.A0_C = A0_t(
                    alphaMP_=self.algorithmPar.alphaCmp,
                    eminMP_=self.eminMP,
                    par_=self.par,
                )
                self.selectA0[algorithmPar.alphaC] = self.A0_C
        self.lambda_list = [[] for _ in range(self.par.Ne)]
        #   First alpha
        self.rho_list = [[] for _ in range(self.par.Ne)]
        self.errBoot_list = [[] for _ in range(self.par.Ne)]
        self.errBayes_list = [[] for _ in range(self.par.Ne)]
        self.gAA0g_list = [[] for _ in range(self.par.Ne)]
        self.likelihood_list = [[] for _ in range(self.par.Ne)]
        self.SigmaMatA = SigmaMatrix(self.par, algorithmPar.alphaA)
        self.selectSigmaMat[algorithmPar.alphaA] = self.SigmaMatA
        #   Second alpha
        if self.par.Na > 1:
            self.rho_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.errBoot_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.errBayes_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.gAA0g_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.likelihood_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.SigmaMatB = SigmaMatrix(self.par, algorithmPar.alphaB)
            self.selectSigmaMat[algorithmPar.alphaB] = self.SigmaMatB
            #   Third alpha
            if self.par.Na > 2:
                self.rho_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.errBoot_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.errBayes_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.gAA0g_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.likelihood_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.SigmaMatC = SigmaMatrix(self.par, algorithmPar.alphaC)
                self.selectSigmaMat[algorithmPar.alphaC] = self.SigmaMatC
        #   Results
        self.minNLL = np.ndarray(
            self.par.Ne, dtype=np.float64
        )  # minimum of Negative Log Likelihood
        self.lambdaResultHLT = np.ndarray(
            self.par.Ne, dtype=np.float64
        )  #   from plateau in lambda
        self.lambdaResultBayes = np.ndarray(
            self.par.Ne, dtype=np.float64
        )  #   from min of NLL
        self.rhoResultHLT = np.ndarray(
            self.par.Ne, dtype=np.float64
        )  #   from plateau in lambda
        self.drho_result = np.ndarray(
            self.par.Ne, dtype=np.float64
        )  #   from plateau in lambda
        self.rhoResultBayes = np.ndarray(
            self.par.Ne, dtype=np.float64
        )  #   from min of NLL
        self.drho_bayes = np.ndarray(self.par.Ne, dtype=np.float64)  #   frin min of NLL
        self.rho_sys_err_HLT = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_quadrature_err_HLT = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_sys_err_Bayes = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_quadrature_err_Bayes = np.ndarray(self.par.Ne, dtype=np.float64)
        self.gt_HLT = [[] for _ in range(self.par.Ne)]  #   from plateau in lambda
        self.gt_Bayes = [[] for _ in range(self.par.Ne)]  #   from min of NLL
        self.aa0 = np.ndarray(
            self.par.Ne, dtype=np.float64
        )  # A / A0 for the result (HLT only)
        #   Control variables
        self.espace_is_filled = False
        self.A0_is_filled = False
        self.result_is_filled = np.full(par.Ne, False, dtype=bool)
        return
        # - - - - - - - - - - - - - - - End of INIT - - - - - - - - - - - - - - - #

    def fillEspaceMP(self):
        print(LogMessage(), "Chosen energies: ", self.espace)
        for e_id in range(self.par.Ne):
            self.espaceMP[e_id] = mpf(str(self.espace[e_id]))
            self.espace_dictionary[
                self.espace[e_id]
            ] = e_id  #   pass the FLOAT, get the INTEGER
        self.espace_is_filled = True
        return

    def prepareHLT(self):
        self.fillEspaceMP()
        self.A0_A.evaluate(self.espaceMP)
        self.SigmaMatA.evaluate()
        with open(
            os.path.join(self.par.logpath, "InverseProblemLOG_AlphaA.log"), "w"
        ) as output:
            print(
                "# estar\t",
                "lambda\t",
                "rho\t",
                "errBayes\t",
                "errBoot\t",
                "A/A0\t",
                "likelihood\n",
                file=output,
            )
        if self.par.Na > 1:
            self.A0_B.evaluate(self.espaceMP)
            self.SigmaMatB.evaluate()
            with open(
                os.path.join(self.par.logpath, "InverseProblemLOG_AlphaB.log"), "w"
            ) as output:
                print(
                    "# estar\t",
                    "lambda\t",
                    "rho\t",
                    "errBayes\t",
                    "errBoot\t",
                    "A/A0\t",
                    "likelihood\n",
                    file=output,
                )
            if self.par.Na > 2:
                self.A0_C.evaluate(self.espaceMP)
                self.SigmaMatC.evaluate()
                with open(
                    os.path.join(self.par.logpath, "InverseProblemLOG_AlphaC.log"), "w"
                ) as output:
                    print(
                        "# estar\t",
                        "lambda\t",
                        "rho\t",
                        "errBayes\t",
                        "errBoot\t",
                        "A/A0\t",
                        "likelihood\n",
                        file=output,
                    )
        self.A0_is_filled = True
        return

    # TODO: do we also want to print gt?
    def _store(
        self,
        estar,
        rho,
        errBayes,
        errBoot,
        likelihood,
        gag,
        gt,
        lambda_,
        whichAlpha="A",
    ):
        if whichAlpha == "A":
            self.rho_list[self.espace_dictionary[estar]].append(rho)
            self.errBayes_list[self.espace_dictionary[estar]].append(errBayes)
            self.errBoot_list[self.espace_dictionary[estar]].append(errBoot)
            self.gAA0g_list[self.espace_dictionary[estar]].append(
                gag
                / self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[estar]
            )
            self.likelihood_list[self.espace_dictionary[estar]].append(likelihood)
            with open(
                os.path.join(self.par.logpath, "InverseProblemLOG_AlphaA.log"), "a"
            ) as output:
                print(
                    float(estar),
                    float(lambda_),
                    float(rho),
                    float(errBayes),
                    float(errBoot),
                    float(gag),
                    float(likelihood),
                    file=output,
                )
        if whichAlpha == "B":
            self.rho_list_alphaB[self.espace_dictionary[estar]].append(rho)
            self.errBayes_list_alphaB[self.espace_dictionary[estar]].append(errBayes)
            self.errBoot_list_alphaB[self.espace_dictionary[estar]].append(errBoot)
            self.gAA0g_list_alphaB[self.espace_dictionary[estar]].append(
                gag
                / self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[estar]
            )
            self.likelihood_list_alphaB[self.espace_dictionary[estar]].append(
                likelihood
            )
            with open(
                os.path.join(self.par.logpath, "InverseProblemLOG_AlphaB.log"), "a"
            ) as output:
                print(
                    float(estar),
                    float(lambda_),
                    float(rho),
                    float(errBayes),
                    float(errBoot),
                    float(gag),
                    float(likelihood),
                    file=output,
                )
        if whichAlpha == "C":
            self.rho_list_alphaC[self.espace_dictionary[estar]].append(rho)
            self.errBayes_list_alphaC[self.espace_dictionary[estar]].append(errBayes)
            self.errBoot_list_alphaC[self.espace_dictionary[estar]].append(errBoot)
            self.gAA0g_list_alphaC[self.espace_dictionary[estar]].append(
                gag
                / self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[estar]
            )
            self.likelihood_list_alphaC[self.espace_dictionary[estar]].append(
                likelihood
            )
            with open(
                os.path.join(self.par.logpath, "InverseProblemLOG_AlphaC.log"), "a"
            ) as output:
                print(
                    float(estar),
                    float(lambda_),
                    float(rho),
                    float(errBayes),
                    float(errBoot),
                    float(gag),
                    float(likelihood),
                    file=output,
                )
        return

    def _areRangesCompatible(self, x, deltax, y, deltay):
        x2 = x + deltax
        x1 = x - deltax
        y1 = y - deltay
        y2 = y + deltay
        return x1 <= y2 and y1 <= x2

    def _flagResult(self, *args):
        return args

        # - - - - - - - - - - - - - - - Main function: given lambda computes rho_s - - - - - - - - - - - - - - - #

    def lambdaToRho(self, lambda_, estar_, alpha_):
        import time

        _Bnorm = self.matrix_bundle.bnorm / (estar_ * estar_)
        _factor = (
            lambda_ * self.selectA0[float(alpha_)].valute_at_E_dictionary[estar_]
        ) / _Bnorm
        print(LogMessage(), "Normalising factor A*l/B = {:2.2e}".format(float(_factor)))

        S_ = self.selectSigmaMat[float(alpha_)].matrix
        _Matrix = S_ + (_factor * self.matrix_bundle.B)
        start_time = time.time()
        _MatrixInv = invert_matrix_ge(_Matrix)
        end_time = time.time()
        print(
            LogMessage(),
            "Time ::: Matrix inverted in {:4.4f}".format(end_time - start_time),
            "s",
        )
        start_time = time.time()
        _g_t_estar = h_Et_mp_Eslice(_MatrixInv, self.par, estar_, alpha_=alpha_)
        end_time = time.time()
        print(
            LogMessage(),
            "Time ::: Coefficients computed in {:4.4f}".format(end_time - start_time),
            "s",
        )
        rho_estar, drho_estar_Bootstrap = y_combine_sample_Eslice_mp(
            _g_t_estar, self.correlator.mpsample, self.par
        )
        start_time = time.time()
        print(
            LogMessage(),
            "Time ::: Bootstrapped result in {:4.4f}".format(start_time - end_time),
            "s",
        )
        gAg_estar = gAg(S_, _g_t_estar, estar_, alpha_, self.par)

        varianceRho = combine_fMf_Eslice(
            ht_sliced=_g_t_estar, params=self.par, estar_=estar_, alpha_=alpha_
        )
        print(LogMessage(), "\t\t gt ft = ", float(varianceRho))
        print(
            LogMessage(),
            "\t\t A0 is ",
            float(self.selectA0[float(alpha_)].valute_at_E_dictionary[estar_]),
        )
        varianceRho = mp.fsub(
            float(self.selectA0[float(alpha_)].valute_at_E_dictionary[estar_]),
            varianceRho,
        )
        print(LogMessage(), "\t\t A0 - gt ft = {:2.2e}".format(float(varianceRho)))
        varianceRho = mp.fdiv(varianceRho, _factor)
        varianceRho = mp.fdiv(varianceRho, mpf(2))
        drho_estar_Bayes = mp.sqrt(abs(varianceRho))

        print(
            LogMessage(),
            "\t \t lambdaToRho ::: Central Value = {:2.4e}".format(float(rho_estar)),
        )
        print(
            LogMessage(),
            "\t \t lambdaToRho ::: Bayesian Error = {:2.4e}".format(
                float(drho_estar_Bayes)
            ),
        )
        print(
            LogMessage(),
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
        print(
            LogMessage(),
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
        lambda_ = self.algorithmPar.lambdaMax
        lambda_step = self.algorithmPar.lambdaStep
        _cap = self.algorithmPar.lambdaScanCap
        _resize = self.algorithmPar.resize
        _countPositiveResult = 0
        _compRatio = self.algorithmPar.comparisonRatio

        print(LogMessage(), " --- ")
        print(LogMessage(), "At Energy {:2.2e}".format(estar_))
        print(
            LogMessage(),
            "Setting Lambda ::: Lambda (0,inf) = {:1.3e}".format(float(lambda_)),
        )
        print(
            LogMessage(),
            "Setting Lambda ::: Lambda (0,1) = {:1.3e}".format(
                float(lambda_ / (1 + lambda_))
            ),
        )

        #   First runs on initial values, then loops

        self.lambda_list[self.espace_dictionary[estar_]].append(lambda_)

        _rho, _errBayes, _errBoot, _likelihood, _gAg, _gt = self.lambdaToRho(
            lambda_, estar_, self.algorithmPar.alphaAmp
        )
        self._store(
            estar_,
            _rho,
            _errBayes,
            _errBoot,
            _likelihood,
            _gAg,
            _gt,
            lambda_,
            whichAlpha="A",
        )
        if self.par.Na > 1:
            _rhoB, _errBayesB, _errBootB, _likelihoodB, _gAgB, _gtB = self.lambdaToRho(
                lambda_, estar_, self.algorithmPar.alphaBmp
            )
            self._store(
                estar_,
                _rhoB,
                _errBayesB,
                _errBootB,
                _likelihoodB,
                _gAgB,
                _gtB,
                lambda_,
                whichAlpha="B",
            )
            if self.par.Na > 2:
                (
                    _rhoC,
                    _errBayesC,
                    _errBootC,
                    _likelihoodC,
                    _gAgC,
                    _gtC,
                ) = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaCmp)
                self._store(
                    estar_,
                    _rhoC,
                    _errBayesC,
                    _errBootC,
                    _likelihoodC,
                    _gAgC,
                    _gtC,
                    lambda_,
                    whichAlpha="C",
                )

        #   Flag these until better results
        _minNLL, _lambdaStarBayes, _rhoBayes, _drhoBayes, _gtBAYES = self._flagResult(
            _likelihood, lambda_, _rho, _errBayes, _gt
        )

        #   Loops over values of lambda
        lambda_ -= lambda_step
        while _countPositiveResult < _cap and lambda_ > self.algorithmPar.lambdaMin:
            #   -   -   -   -   -   -
            print(
                LogMessage(),
                "Setting Lambda ::: Lambda (0,inf) = {:1.3e}".format(float(lambda_)),
            )
            print(
                LogMessage(),
                "Setting Lambda ::: Lambda (0,1) = {:1.3e}".format(
                    float(lambda_ / (1 + lambda_))
                ),
            )
            self.lambda_list[self.espace_dictionary[estar_]].append(lambda_)

            #   Flag these until better results
            if _countPositiveResult == 0:
                _lambdaStarHLT, _rhoHLT, _drhoHLT, _gtHLT, gag_flag = self._flagResult(
                    lambda_, _rho, _errBoot, _gt, _gAg
                )
            #   -   -   -   -   -   -

            print(
                LogMessage(),
                "\t Setting Alpha ::: First Alpha = ",
                self.algorithmPar.alphaA,
            )
            (
                _rhoUpdated,
                _errBayesUpdated,
                _errBootUpdated,
                _likelihoodUpdated,
                _gAgUpdated,
                _gtUpdated,
            ) = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaAmp)
            self._store(
                estar_,
                _rhoUpdated,
                _errBayesUpdated,
                _errBootUpdated,
                _likelihoodUpdated,
                _gAgUpdated,
                _gtUpdated,
                lambda_,
                whichAlpha="A",
            )
            if self.par.Na > 1:
                print(
                    LogMessage(),
                    "\t Setting Alpha ::: Second Alpha = ",
                    self.algorithmPar.alphaB,
                )
                (
                    _rhoUpdatedB,
                    _errBayesUpdatedB,
                    _errBootUpdatedB,
                    _likelihoodUpdatedB,
                    _gAgUpdatedB,
                    _gtUpdatedB,
                ) = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaBmp)
                self._store(
                    estar_,
                    _rhoUpdatedB,
                    _errBayesUpdatedB,
                    _errBootUpdatedB,
                    _likelihoodUpdatedB,
                    _gAgUpdatedB,
                    _gtUpdatedB,
                    lambda_,
                    whichAlpha="B",
                )
                _AB_Overlap = self._areRangesCompatible(
                    _rhoUpdated, _errBootUpdated, _rhoUpdatedB, _errBootUpdatedB
                )
                if self.par.Na > 2:
                    print(
                        LogMessage(),
                        "\t Setting Alpha ::: Third Alpha = ",
                        self.algorithmPar.alphaC,
                    )
                    (
                        _rhoUpdatedC,
                        _errBayesUpdatedC,
                        _errBootUpdatedC,
                        _likelihoodUpdatedC,
                        _gAgUpdatedC,
                        _gtUpdatedC,
                    ) = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaCmp)
                    self._store(
                        estar_,
                        _rhoUpdatedC,
                        _errBayesUpdatedC,
                        _errBootUpdatedC,
                        _likelihoodUpdatedC,
                        _gAgUpdatedC,
                        _gtUpdatedC,
                        lambda_,
                        whichAlpha="C",
                    )
                    _AC_Overlap = self._areRangesCompatible(
                        _rhoUpdated, _errBootUpdated, _rhoUpdatedC, _errBootUpdatedC
                    )

            newLambda_Overlap = self._areRangesCompatible(
                _rhoUpdated, _compRatio * _errBootUpdated, _rho, _compRatio * _errBoot
            )  # comparison with previous lambda

            if _likelihoodUpdated < _minNLL:  # NLL
                (
                    _minNLL,
                    _lambdaStarBayes,
                    _rhoBayes,
                    _drhoBayes,
                    _gtBAYES,
                ) = self._flagResult(
                    _likelihoodUpdated,
                    lambda_,
                    _rhoUpdated,
                    _errBayesUpdated,
                    _gtUpdated,
                )

            _skip = False
            #   Checks compatibility between different alphas
            if self.par.Na > 1 and not _AB_Overlap:
                print(LogMessage(), "\t First and Second Alpha not compatible")
                _skip = True
            if self.par.Na > 2 and not _AC_Overlap:
                print(LogMessage(), "\t First and Third Alpha not compatible")
                _skip = True

            #   Checks if Rho at this lambda overlaps with Rho at flagged lambda
            if newLambda_Overlap is False:
                print(
                    LogMessage(),
                    "\t Result at this Lambda does not overlap with previous: REJECTING result",
                )
            if (
                _gAgUpdated
                / self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[estar_]
                > self.par.A0cut
            ):
                print(
                    LogMessage(),
                    "\t A/A0 is too large: rejecting result  (",
                    float(
                        _gAgUpdated
                        / self.selectA0[
                            self.algorithmPar.alphaA
                        ].valute_at_E_dictionary[estar_]
                    ),
                    ")",
                )
                _skip = True
            if (
                (newLambda_Overlap is True)
                and (
                    _gAgUpdated
                    / self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[
                        estar_
                    ]
                    < self.par.A0cut
                )
                and (_skip is False)
            ):
                #   Flag the first compatible result
                if (
                    _countPositiveResult == 1
                ):  #   At future alphas we compare with rho_s at _countPositiveResult = 1. This can be changed.
                    (
                        _lambdaStarHLT,
                        _rhoHLT,
                        _drhoHLT,
                        _gtHLT,
                        gag_flag,
                    ) = self._flagResult(
                        lambda_, _rho, _errBoot, _gtUpdated, _gAgUpdated
                    )
                _countPositiveResult += 1
                print(
                    LogMessage(),
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
            if lambda_ <= 0:
                lambda_step /= _resize
                lambda_ += lambda_step * (_resize - 1 / _resize)
                print(
                    LogMessage(),
                    "Resize LambdaStep to ",
                    lambda_step,
                    "Setting Lambda = ",
                    lambda_,
                )

            if lambda_ < self.algorithmPar.lambdaMin:
                print(
                    LogMessage(),
                    f"{bcolors.WARNING}Warning{bcolors.ENDC} ::: Stopping ::: Reached lower limit for lambda. Try decreasing 'algorithmPar.lambdaMin' or increase the smearing radius.",
                )

        #   End of WHILE
        if _countPositiveResult == 0:
            print(
                LogMessage(),
                f"{bcolors.WARNING}WARNING{bcolors.ENDC} ::: Could NOT find a plateau in lambda",
            )

        #   hlt
        self.lambdaResultHLT[self.espace_dictionary[estar_]] = _lambdaStarHLT
        self.rhoResultHLT[self.espace_dictionary[estar_]] = _rhoHLT
        self.drho_result[self.espace_dictionary[estar_]] = _drhoHLT
        self.gt_HLT[self.espace_dictionary[estar_]] = _gtHLT
        self.aa0[self.espace_dictionary[estar_]] = gag_flag
        #   bayesian
        self.minNLL[self.espace_dictionary[estar_]] = _minNLL
        self.lambdaResultBayes[self.espace_dictionary[estar_]] = _lambdaStarBayes
        self.rhoResultBayes[self.espace_dictionary[estar_]] = _rhoBayes
        self.drho_bayes[self.espace_dictionary[estar_]] = _drhoBayes
        self.gt_Bayes[self.espace_dictionary[estar_]] = _gtBAYES

        return (
            _lambdaStarHLT,
            _rhoHLT,
            _drhoHLT,
            _minNLL,
            _lambdaStarBayes,
            _rhoBayes,
            _drhoBayes,
            _gtHLT,
            _gtBAYES,
            gag_flag,
        )

    def estimate_sys_error(self, e_i):
        _this_y_HLT = self.rhoResultHLT[e_i]  # rho at lambda*
        _that_y_HLT, _, _, _, _, _ = self.lambdaToRho(
            self.lambdaResultHLT[e_i] * self.algorithmPar.kfactor,
            self.espace[e_i],
            alpha_=0,
        )

        _this_y_Bayes = self.rhoResultBayes[e_i]  # rho at lambda*
        _that_y_Bayes, _, _, _, _, _ = self.lambdaToRho(
            self.lambdaResultBayes[e_i] * self.algorithmPar.kfactor,
            self.espace[e_i],
            alpha_=0,
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

    # TODO: find a good place for printing g_t
    def run(self, savePlots=True, livePlots=False):
        with open(os.path.join(self.par.logpath, "ResultHLT.txt"), "w") as output:
            print(
                "# Energy \t Lambda(HLT) \t Rho(HLT) \t Stat(HLT) \t Sys(HLT) \t Quadrature \t A/A0",
                file=output,
            )
        with open(os.path.join(self.par.logpath, "ResultBayes.txt"), "w") as output:
            print(
                "# Energy \t Lambda(Bayes) \t Rho(Bayes) \t Stat(Bayes) \t Sys(Bayes) \t Quadrature \t NLL",
                file=output,
            )

        for e_i in range(self.par.Ne):
            (
                lambdaHLT,
                rhoHLT,
                errHLT,
                minNLL,
                lambdaBayes,
                rhoBayes,
                errBayes,
                gtHLT,
                gtBayes,
                gaa0g,
            ) = self.scanParameters(self.espace[e_i])
            _ = self.estimate_sys_error(e_i)
            with open(os.path.join(self.par.logpath, "ResultHLT.txt"), "a") as output:
                print(
                    self.espace[e_i],
                    self.lambdaResultHLT[e_i],
                    float(self.rhoResultHLT[e_i]),
                    float(self.drho_result[e_i]),
                    float(self.rho_sys_err_HLT[e_i]),
                    float(self.rho_quadrature_err_HLT[e_i]),
                    float(self.aa0[e_i]),
                    file=output,
                )
            with open(os.path.join(self.par.logpath, "ResultBayes.txt"), "a") as output:
                print(
                    self.espace[e_i],
                    self.lambdaResultBayes[e_i],
                    float(self.rhoResultBayes[e_i]),
                    float(self.drho_bayes[e_i]),
                    float(self.rho_sys_err_Bayes[e_i]),
                    float(self.rho_quadrature_err_Bayes[e_i]),
                    float(self.minNLL[e_i]),
                    file=output,
                )

        return 0

    def stabilityPlot(
        self,
        generateHLTscan=True,
        generateLikelihoodShared=True,
        generateLikelihoodPlot=True,
        generateKernelsPlot=True,
    ):
        for e_i in range(self.par.Ne):
            if generateHLTscan is True:
                stabilityPlot(self, self.espace[e_i], savePlot=True, plot_live=False)
            if generateLikelihoodShared is True:
                sharedPlot_stabilityPlusLikelihood(
                    self, self.espace[e_i], savePlot=True, plot_live=False
                )
            if generateLikelihoodPlot is True:
                plotLikelihood(self, self.espace[e_i], savePlot=True, plot_live=False)
            if generateKernelsPlot is True:
                plotAllKernels(self)

    def plotResult(self):
        plotSpectralDensity(self)
        return
