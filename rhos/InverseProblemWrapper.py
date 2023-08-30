from mpmath import mp, mpf
import sys
import numpy as np
import os
sys.path.append("../utils")
from rhoMath import *
from rhoUtils import LogMessage, Inputs, MatrixBundle
from rhoUtils import *
from transform import *
from abw import *
from core import *
from core import A0E_mp
from abw import gAg, gBg
import matplotlib.pyplot as plt

_big = 100

class AlgorithmParameters:
    def __init__(
        self,
        alphaA=0,
        alphaB=1/2,
        alphaC=1.99,
        lambdaMax=50,
        lambdaStep=0.5,
        lambdaScanPrec=0.1,
        lambdaScanCap=6,
        kfactor=0.1,
        lambdaMin=0.,
    ):
        assert alphaA != alphaB
        assert alphaA != alphaC
        self.alphaA = float(alphaA)
        self.alphaB = float(alphaB)
        self.alphaC = float(alphaC)
        self.lambdaMax = lambdaMax
        self.lambdaStep = lambdaStep
        self.lambdaScanPrec = lambdaScanPrec
        self.lambdaScanCap = lambdaScanCap
        self.kfactor = kfactor
        self.alphaAmp = mpf(str(alphaA))
        self.alphaBmp = mpf(str(alphaB))
        self.alphaCmp = mpf(str(alphaC))
        self.lambdaMin = lambdaMin

class A0_t:
    def __init__(self, par_: Inputs, alphaMP_=0, eminMP_=0):
        self.valute_at_E = mp.matrix(par_.Ne, 1)
        self.valute_at_E_dictionary = (
            {}
        )  # Auxiliary dictionary: A0espace[n] = A0espace_dictionary[espace[n]] # espace must be float
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
        self.alpha=alphaMP
        self.matrix = mp.matrix(par.tmax, par.tmax)
    def evaluate(self):
        print(LogMessage(), " Saving Sigma Matrix ".format(self.alpha))
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
        read_energies = 0,
    ):
        self.par = par
        self.correlator = correlator
        self.algorithmPar = algorithmPar
        self.matrix_bundle = matrix_bundle
        #
        if read_energies==0:
            self.espace = np.linspace(par.emin, par.emax, par.Ne)
        elif read_energies != 0:
            print(LogMessage(), "InverseProblemWrapper ::: Reading input energies")
            par.Ne = len(read_energies)
            self.espace = read_energies
        #
        self.e0MP = mpf(str(par.e0))
        self.espaceMP = mp.matrix(par.Ne, 1)
        self.sigmaMP = mpf(str(par.sigma))
        self.emaxMP = mpf(str(par.emax))
        self.eminMP = mpf(str(par.emin))
        self.espace_dictionary = {}  #   Auxiliary dictionary: espace_dictionary[espace[n]] = n
        self.selectSigmaMat = {}

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
                    alphaMP_=self.algorithmPar.alphaCmp, eminMP_=self.eminMP, par_=self.par
                )
                self.selectA0[algorithmPar.alphaC] = self.A0_C

        self.lambda_list = [[] for _ in range(self.par.Ne)]
        #   First alpha
        self.rho_list = [[] for _ in range(self.par.Ne)]
        self.errBoot_list = [[] for _ in range(self.par.Ne)]
        self.errBayes_list = [[] for _ in range(self.par.Ne)]
        self.gAA0g_list = [[] for _ in range(self.par.Ne)]
        self.likelihood_list = [[] for _ in range(self.par.Ne)]
        self.gt_list = [[] for _ in range(self.par.Ne)]
        self.SigmaMatA = SigmaMatrix(self.par, algorithmPar.alphaA)
        self.selectSigmaMat[algorithmPar.alphaA] = self.SigmaMatA

        #   Second alpha
        if self.par.Na > 1:
            self.rho_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.errBoot_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.errBayes_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.gAA0g_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.likelihood_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.gt_list_alphaB = [[] for _ in range(self.par.Ne)]
            self.SigmaMatB = SigmaMatrix(self.par, algorithmPar.alphaB)
            self.selectSigmaMat[algorithmPar.alphaB] = self.SigmaMatB

        #   Third alpha
            if self.par.Na > 2:
                self.rho_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.errBoot_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.errBayes_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.gAA0g_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.likelihood_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.gt_list_alphaC = [[] for _ in range(self.par.Ne)]
                self.SigmaMatC = SigmaMatrix(self.par, algorithmPar.alphaC)
                self.selectSigmaMat[algorithmPar.alphaC] = self.SigmaMatC

        #   Results
        self.lambda_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.likelihood_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.drho_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_sys_err = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_quadrature_err = np.ndarray(self.par.Ne, dtype=np.float64)
        #   Control variables
        self.espace_is_filled = False
        self.A0_is_filled = False
        self.result_is_filled = np.full(par.Ne, False, dtype=bool)
        return
        # - - - - - - - - - - - - - - - End of INIT - - - - - - - - - - - - - - - #

    def fillEspaceMP(self):
        for e_id in range(self.par.Ne):
            self.espaceMP[e_id] = mpf(str(self.espace[e_id]))
            self.espace_dictionary[self.espace[e_id]] = e_id    #   pass the FLOAR, get the INTEGER
        self.espace_is_filled = True
        return

    def prepareHLT(self):
        self.fillEspaceMP()
        self.A0_A.evaluate(self.espaceMP)
        self.SigmaMatA.evaluate()
        with open(os.path.join(self.par.logpath, 'InverseProblemLOG_AlphaA.txt'), "w") as output:
            print("# estar\t", "lambda\t", "rho\t", "errBayes\t", "errBoot\t", "A/A0\t", "gt\t", "likelihood\t")
        if self.par.Na > 1:
            self.A0_B.evaluate(self.espaceMP)
            self.SigmaMatB.evaluate()
            with open(os.path.join(self.par.logpath, 'InverseProblemLOG_AlphaB.txt'), "w") as output:
                print("# estar\t", "lambda\t", "rho\t", "errBayes\t", "errBoot\t", "A/A0\t", "gt\t", "likelihood\t")
            if self.par.Na > 2:
                self.A0_C.evaluate(self.espaceMP)
                self.SigmaMatC.evaluate()
                with open(os.path.join(self.par.logpath, 'InverseProblemLOG_AlphaC.txt'), "w") as output:
                    print("# estar\t", "lambda\t", "rho\t", "errBayes\t", "errBoot\t", "A/A0\t", "gt\t", "likelihood\t")
        self.A0_is_filled = True
        return

    def store(self, estar, rho, errBayes, errBoot, likelihood, gag, gt, lambda_, whichAlpha='A'):
        if whichAlpha == 'A':
            self.rho_list[self.espace_dictionary[estar]].append(rho)
            self.errBayes_list[self.espace_dictionary[estar]].append(errBayes)
            self.errBoot_list[self.espace_dictionary[estar]].append(errBoot)
            self.gAA0g_list[self.espace_dictionary[estar]].append(gag / self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[estar])  # store
            self.likelihood_list[self.espace_dictionary[estar]].append(likelihood)
            self.gt_list[self.espace_dictionary[estar]].append(gt)
            with open(os.path.join(self.par.logpath, 'InverseProblemLOG_AlphaA.log'), "a") as output:
                print(float(estar), float(lambda_), float(rho), float(errBayes), float(errBoot), float(gag), gt, float(likelihood))
        if whichAlpha == 'B':
            self.rho_list_alphaB[self.espace_dictionary[estar]].append(rho)
            self.errBayes_list_alphaB[self.espace_dictionary[estar]].append(errBayes)
            self.errBoot_list_alphaB[self.espace_dictionary[estar]].append(errBoot)
            self.gAA0g_list_alphaB[self.espace_dictionary[estar]].append(gag / self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[estar])  # store
            self.likelihood_list_alphaB[self.espace_dictionary[estar]].append(likelihood)
            self.gt_list_alphaB[self.espace_dictionary[estar]].append(gt)
            with open(os.path.join(self.par.logpath, 'InverseProblemLOG_AlphaB.log'), "a") as output:
                print(float(estar), float(lambda_), float(rho), float(errBayes), float(errBoot), float(gag), gt, float(likelihood))
        if whichAlpha == 'C':
            self.rho_list_alphaC[self.espace_dictionary[estar]].append(rho)
            self.errBayes_list_alphaC[self.espace_dictionary[estar]].append(errBayes)
            self.errBoot_list_alphaC[self.espace_dictionary[estar]].append(errBoot)
            self.gAA0g_list_alphaC[self.espace_dictionary[estar]].append(gag / self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[estar])  # store
            self.likelihood_list_alphaC[self.espace_dictionary[estar]].append(likelihood)
            self.gt_list_alphaC[self.espace_dictionary[estar]].append(gt)
            with open(os.path.join(self.par.logpath, 'InverseProblemLOG_AlphaC.log'), "a") as output:
                print(float(estar), float(lambda_), float(rho), float(errBayes), float(errBoot), float(gag), gt, float(likelihood))
        return

        # - - - - - - - - - - - - - - - Main function: given lambda computes rho_s - - - - - - - - - - - - - - - #

    def lambdaToRho(self, lambda_, estar_, alpha_):
        import time

        _Bnorm = (self.matrix_bundle.bnorm / (estar_ * estar_))
        _factor = (lambda_ * self.selectA0[float(alpha_)].valute_at_E_dictionary[estar_]) / _Bnorm
        print(LogMessage(), "Normalising factor A*l/B = {:2.2e}".format(float(_factor)))

        S_ = self.selectSigmaMat[float(alpha_)].matrix
        _Matrix = S_ + (_factor * self.matrix_bundle.B)
        start_time = time.time()
        _MatrixInv = invert_matrix_ge(_M)
        end_time = time.time()
        print(LogMessage(), "\t \t lambdaToRho ::: Matrix inverted in {:4.4f}".format(end_time - start_time), "s")

        _g_t_estar = h_Et_mp_Eslice(_MatrixInv, self.par, estar_, alpha_=alpha_)

        rho_estar, drho_estar_Bootstrap = y_combine_sample_Eslice_mp(_g_t_estar, self.correlator.mpsample, self.par)

        gAg_estar = gAg(S_, _g_t_estar, estar_, alpha_, self.par)

        varianceRho = combine_fMf_Eslice(ht_sliced=_g_t_estar, params=self.par, estar_=estar_, alpha_=alpha_)
        print(LogMessage(), "\t\t gt ft = ", float(varianceRho))
        print(LogMessage(), '\t\t A0 is ', float(self.selectA0[float(alpha_)].valute_at_E_dictionary[estar_]))
        varianceRho = mp.fsub(float(self.selectA0[float(alpha_)].valute_at_E_dictionary[estar_]), varianceRho)
        print(LogMessage(), "\t\t A0 - gt ft = {:2.2e}".format(float(varianceRho)))
        varianceRho = mp.fdiv(varianceRho, _factor)  # 2 gamma^2
        varianceRho = mp.fdiv(varianceRho, mpf(2))  # gamma^2
        drho_estar_Bayes = mp.sqrt(varianceRho)  # gamma

        print(LogMessage(), "\t \t lambdaToRho ::: Central Value = {:2.4e}".format(float(rho_estar)))
        print(LogMessage(), "\t \t lambdaToRho ::: Bayesian Error = {:2.4e}".format(float(drho_estar_Bayes)))
        print(LogMessage(), "\t \t lambdaToRho ::: Bootstrap Error   = {:2.4e}".format(float(drho_estar_Bootstrap)))

        fullCov_inv = _MatrixInv * _factor

        #   Compute the likelihood
        likelihood_estar = combine_likelihood(fullCov_inv, self.par, self.correlator.mpcentral)
        likelihood_estar *= 0.5
        det = mp.det(fullCov)
        likelihood_estar = mp.fadd(likelihood_estar, 0.5 * mp.log(det))
        likelihood_estar = mp.fadd(likelihood_estar, (self.par.tmax * mp.log(2 * mp.pi)) * 0.5)
        print(LogMessage(),  "\t \t lambdaToRho ::: Likelihood = {:2.4e}".format(float(likelihood_estar)))

        return rho_estar, drho_estar_Bayes, drho_estar_Bootstrap, likelihood_estar, gAg_estar, _g_t_estar

    # - - - - - - - - - - - - - - - Scan over parameters: Lambda, Alpha (optional) - - - - - - - - - - - - - - - #

    def scanParameters(self, estar_, how_many_alphas=2):
        lambda_ = self.algorithmPar.lambdaMax
        lambda_step = self.algorithmPar.lambdaStep
        prec_ = self.algorithmPar.lambdaScanPrec
        cap_ = self.algorithmPar.lambdaScanCap
        _count = 0
        resize = 4
        lambda_flag = 0
        rho_flag = 0
        drho_flag = 0
        comp_diff_AC = _big
        comp_diff_AB = _big

        print(LogMessage(), " --- ")
        print(LogMessage(), "At Energy {:2.2e}".format(estar_))
        print(LogMessage(), "Setting Lambda ::: Lambda (0,inf) = {:1.3e}".format(float(lambda_)))
        print(LogMessage(), "Setting Lambda ::: Lambda (0,1) = {:1.3e}".format(float(lambda_ / (1 + lambda_))))

        #   First we run on initial values, the we loop

        self.lambda_list[self.espace_dictionary[estar_]].append(lambda_)
        _rho, _errBayes, _errBoot, _likelihood, _gAg, _gt = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaAmp)
        self.store(estar_, _rho, _errBayes, _errBoot, _likelihood, _gAg, _gt, lambda_, whichAlpha='A')

        if self.par.Na > 1:
            _rho, _errBayes, _errBoot, _likelihood, _gAg, _gt = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaBmp)
            self.store(estar_, _rho, _errBayes, _errBoot, _likelihood, _gAg, _gt, lambda_, whichAlpha='B')

        if self.par.Na > 2:
            _rho, _errBayes, _errBoot, _likelihood, _gAg, _gt = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaCmp)
            self.store(estar_, _rho, _errBayes, _errBoot, _likelihood, _gAg, _gt, lambda_, whichAlpha='C')

        lambda_ -= lambda_step

        #   Loop over values of lambda
        while _count < cap_ and lambda_ > self.algorithmPar.lambdaMin:
            print(LogMessage(), "Setting Lambda ::: Lambda (0,inf) = {:1.3e}".format(float(lambda_)))
            print(LogMessage(), "Setting Lambda ::: Lambda (0,1) = {:1.3e}".format(float(lambda_ / (1 + lambda_))))
            self.lambda_list[self.espace_dictionary[estar_]].append(lambda_)
            #   Lambda is set
            print(LogMessage(), "\t Setting Alpha ::: First Alpha = ", self.algorithmPar.alphaA)
            _rhoUpdated, _errBayesUpdated, _errBootUpdated, _likelihoodUpdated, _gAgUpdated, _gtUpdated = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaAmp)
            self.store(estar_, _rhoUpdated, _errBayesUpdated, _errBootUpdated, _likelihoodUpdated, _gAgUpdated, _gtUpdated, lambda_, whichAlpha='A')
            if self.par.Na > 1:
                print(LogMessage(), "\t Setting Alpha ::: Second Alpha = ", self.algorithmPar.alphaB)
                _rhoUpdatedB, _errBayesUpdatedB, _errBootUpdatedB, _likelihoodUpdatedB, _gAgUpdatedB, _gtUpdatedB = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaBmp)
                self.store(estar_, _rhoUpdatedB, _errBayesUpdatedB, _errBootUpdatedB, _likelihoodUpdatedB, _gAgUpdatedB, _gtUpdatedB, lambda_, whichAlpha='B')
                if self.par.Na > 2:
                    print(LogMessage(), "\t Setting Alpha ::: Third Alpha = ", self.algorithmPar.alphaC)
                    _rhoUpdatedC, _errBayesUpdatedC, _errBootUpdatedC, _likelihoodUpdatedC, _gAgUpdatedC, _gtUpdatedC = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaCmp)
                    self.store(estar_, _rhoUpdatedC, _errBayesUpdatedC, _errBootUpdatedC, _likelihoodUpdatedC, _gAgUpdatedC, _gtUpdatedC, lambda_, whichAlpha='C')
            #   Control if we are within 1 sigma from previous result