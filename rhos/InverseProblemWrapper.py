from mpmath import mp, mpf
import sys
import numpy as np
import math
sys.path.append("../utils")
from rhoMath import *
from rhoUtils import LogMessage, Inputs, LambdaSearchOptions, MatrixBundle
from rhoUtils import *
from transform import *
from abw import *
from core import *
from core import A0E_mp, A0E_float64
from abw import gAg, gBg
import scipy.linalg as sp_linalg

class InverseProblemWrapper:
    def __init__(self, par: Inputs, lambda_config: LambdaSearchOptions, matrix_bundle: MatrixBundle, correlator : Obs):
        self.par = par
        self.lambda_config = lambda_config
        self.correlator = correlator
        self.matrix_bundle = matrix_bundle
        self.espace = np.linspace(par.emin, par.emax, par.Ne)
        self.alphaMP = mpf(str(par.alpha))
        self.e0MP = mpf(str(par.e0))
        self.espaceMP = mp.matrix(par.Ne, 1)
        self.sigmaMP = mpf(str(par.sigma))
        self.emaxMP = mpf(str(par.emax))
        self.eminMP = mpf(str(par.emin))
        self.espace_dictionary = {} #   Auxiliary dictionary: espace_dictionary[espace[n]] = n
        self.A0espace = mp.matrix(self.par.Ne,1)
        self.A0espace_dictionary = {} #   Auxiliary dictionary: A0espace[n] = A0espace_dictionary[espace[n]] # espace must be float
        self.espace_is_filled = False
        self.A0_is_filled = False
        self.requires_MP = np.full(par.Ne, False, dtype=bool)
        self.precision_is_always_arbitrary = False
        self.precision_is_never_arbitrary = False
        self.S_float = np.ndarray((self.par.tmax,self.par.tmax), dtype=np.float64)
        self.B_float = np.ndarray((self.par.tmax,self.par.tmax), dtype=np.float64)
        self.W_float = np.ndarray((self.par.tmax, self.par.tmax), dtype=np.float64)
        self.bnorm_float = float(1)
        self.A0_float = np.ndarray(self.par.Ne, dtype = np.float64)
        self.A0espace_float_dictionary = {}
        self.is_float64_initialised = False

    def fillEspaceMP(self):
        for e_id in range(self.par.Ne):
            self.espaceMP[e_id] = mpf(str(self.espace[e_id]))
            self.espace_dictionary[self.espace[e_id]] = e_id
        self.espace_is_filled = True

    def computeA0(self):
        assert(self.espace_is_filled == True)
        print(LogMessage(), "Computing A0 at all energies")
        self.A0espace = A0E_mp(self.espaceMP, self.par)
        for e_id in range(self.par.Ne):
            self.A0espace_dictionary[self.espace[e_id]] = self.A0espace[e_id]
        self.A0_is_filled = True

    def init_float64(self):
        Smatrix_float64(self.par.tmax, self.par.alpha, self.par.e0, S_in=self.S_float)
        self.A0_float = A0E_float64(self.espace, self.par)
        for i in range(self.par.tmax):
            for j in range(self.par.tmax):
                self.B_float[i][j] = self.correlator.cov[i][j]
        self.bnorm_float = float(self.matrix_bundle.bnorm)
        for e_id in range(self.par.Ne):
            self.A0espace_float_dictionary[self.espace[e_id]] = self.A0_float[e_id]
        self.is_float64_initialised = True

    def prepareHLT(self):
        self.fillEspaceMP()
        self.computeA0()

    def report(self):
        print(LogMessage(), "Inverse problem ::: Time extent:", self.par.time_extent)
        print(LogMessage(), "Inverse problem ::: tmax:", self.par.tmax)
        print(LogMessage(), "Inverse problem ::: Mpi:", self.par.massNorm)
        print(LogMessage(), "Inverse problem ::: sigma (mp):", self.par.sigma, "(", self.sigmaMP, ")")
        print(LogMessage(), "Inverse problem ::: Bootstrap samples:", self.par.num_boot)
        print(LogMessage(), "Inverse problem ::: Number of energies:", self.par.Ne)
        print(LogMessage(), "Inverse problem ::: Emax (mp) [lattice unit]:", self.par.emax, self.emaxMP)
        print(LogMessage(), "Inverse problem ::: Emax [mass units]:", self.par.emax / self.par.massNorm)
        print(LogMessage(), "Inverse problem ::: Emin (mp) [lattice unit]:", self.par.emin, self.eminMP)
        print(LogMessage(), "Inverse problem ::: Emin [mass units]:", self.par.emin / self.par.massNorm)
        print(LogMessage(), "Inverse problem ::: alpha (mp):", self.par.alpha, "(", self.alphaMP, ")")
        print(LogMessage(), "Inverse problem ::: E0 (mp):", self.par.e0, "(", self.e0MP, ")")

    def computeMinimumPrecision(self, estar: float, min_bit : int = 64, max_bit : int = 128):
        #   Check whether the result changes within statistical error
        #   going from 64 to 128 bits
        #   this is done at each energy for the smaller lambda
        #   since increasing lambda better conditions the matrices
        #   This is a very verbose check
        #   silent function follows below
        self.prepareHLT()
        assert(self.A0_is_filled == True)
        lambdamin = self.lambda_config.lspace[0]
        factor = (self.A0espace_dictionary[estar]*lambdamin)/((1-lambdamin)*self.matrix_bundle.bnorm)
        factorMP = mpf(str(factor))
        W = self.matrix_bundle.S + factorMP*self.matrix_bundle.B
        condW = mp.cond(W)
        prec = math.ceil(math.log10(condW)) + 1
        print(LogMessage(), 'E = ', estar, "Post-regularisation estimated precision [decimal digits] ", prec)
        mp.prec = min_bit
        Winv = W**(-1)
        print(LogMessage(), "{:4d} Bit W * Winv ".format(min_bit), norm2_mp(W*Winv) - 1)
        gt = h_Et_mp_Eslice(Winv, self.par, estar_=mpf(estar))
        rhoE_32 = y_combine_sample_Eslice_mp(gt, mpmatrix=self.correlator.sample, params=self.par)
        print(LogMessage(), "{:4d} Bit rhoE +/- stat".format(min_bit), float(rhoE_32[0]), '+/-', float(rhoE_32[1]))
        mp.prec = max_bit
        Winv = W ** (-1)
        print(LogMessage(), "{:4d} Bit W * Winv ".format(max_bit), norm2_mp(W*Winv) - 1)
        gt = h_Et_mp_Eslice(Winv, self.par, estar_=mpf(estar))
        rhoE_128 = y_combine_sample_Eslice_mp(gt, mpmatrix=self.correlator.sample, params=self.par)
        print(LogMessage(), "{:4d} Bit rhoE +/- stat".format(max_bit), float(rhoE_128[0]), '+/-', float(rhoE_128[1]))
        diff_rho = abs(rhoE_128[0] - rhoE_32[0])
        allowed_diff = min(rhoE_32[1], rhoE_128[1])
        print(LogMessage(), "At energy ", estar, "Change in Rho (min-max) bits: ", float(diff_rho), "Statistical errors:", float(allowed_diff))
        if diff_rho < allowed_diff:
            print(LogMessage(), f"{bcolors.OKGREEN}Multiple precision not necessary{bcolors.ENDC}")
        else:
            print(LogMessage(), f"{bcolors.OKGREEN}Multiple precision strongly recommended{bcolors.ENDC}")

    def tagAlgebraLibrary(self, estar: float):
        #   Establish if, at each energy, the inversions are performed with mpmath or numpy
        assert (self.A0_is_filled == True)
        lambdamin = self.lambda_config.lspace[0]
        factor = (self.A0espace_dictionary[estar] * lambdamin) / ((1 - lambdamin) * self.matrix_bundle.bnorm)
        factorMP = mpf(str(factor))
        W = self.matrix_bundle.S + factorMP * self.matrix_bundle.B
        mp.prec = 64
        Winv = W ** (-1)
        gt = h_Et_mp_Eslice(Winv, self.par, estar_=mpf(estar))
        rhoE_32 = y_combine_sample_Eslice_mp(gt, mpmatrix=self.correlator.sample, params=self.par)
        mp.prec = 128
        Winv = W ** (-1)
        gt = h_Et_mp_Eslice(Winv, self.par, estar_=mpf(estar))
        rhoE_128 = y_combine_sample_Eslice_mp(gt, mpmatrix=self.correlator.sample, params=self.par)
        diff_rho = abs(rhoE_128[0] - rhoE_32[0])
        allowed_diff = min(rhoE_32[1], rhoE_128[1])
        if diff_rho < allowed_diff:
            self.requires_MP[self.espace_dictionary[estar]] = False
        else:
            self.requires_MP[self.espace_dictionary[estar]] = True

    def _aggregate_lambdaCheckPointOne_float64(self, lambda_float64: float, estar: float):  #   allows to cast as a funciton of lambda only at each estar
        factor = (self.A0espace_float_dictionary[estar] * lambda_float64) / ((1 - lambda_float64) * self.bnorm_float)
        self.W_float = (self.B_float * factor) + self.S_float
        #Winv = np.linalg.inv(self.W_float)
        #Winv = fast_positive_definite_inverse(self.W_float)
        L, lower = sp_linalg.cho_factor(self.W_float)
        Winv = sp_linalg.cho_solve((L, lower), np.eye(self.W_float.shape[0]))
        gt = h_Et_mp_Eslice_float64(Winv, self.par, estar_=estar)
        this_A = gAgA0_float64(self.S_float, gt, estar, self.par, self.A0_float[self.espace_dictionary[estar]])
        this_B = gBg_float64(gt, self.B_float, self.matrix_bundle.bnorm, self.par.tmax)
        return this_A, this_B

    def lambdaCheckPointOne_float64(self, estar, maxiter: int = 100, relative_tol = 0.01):
        assert(self.is_float64_initialised == True)
        assert(self.requires_MP[self.espace_dictionary[estar]] == False)
        #   Search the value of lambda for which A = B
        _this_lambda = self.lambda_config.lspace[int(self.lambda_config.ldensity/2)]
        _startingA, _startingB = self._aggregate_lambdaCheckPointOne_float64(_this_lambda, estar)
        #rho = y_combine_sample_Eslice_mp(gt, mpmatrix=self.correlator.sample, params=self.par)
        _lmin = self.lambda_config.lmin
        _lmax = self.lambda_config.lmax
        for _ in range(maxiter):
            _Aleft, _Bleft = self._aggregate_lambdaCheckPointOne_float64(_lmin, estar)
            _dleft = _Aleft - _Bleft
            _Aright, _Bright = self._aggregate_lambdaCheckPointOne_float64(_lmax, estar)
            _dright = _Aright - _Bright
            _thisA, _thisB = self._aggregate_lambdaCheckPointOne_float64(_this_lambda, estar)
            _dcentral = _thisA - _thisB
            print(LogMessage(), "Energy = ", estar, "lambda", _this_lambda, "A, B", _thisA, _thisB)
            if (abs(_dcentral / _thisA) < relative_tol) and ( abs(_dcentral / _thisB) < relative_tol) :
                print(LogMessage(), "Converged at iteration ", _, " ::: ", "A, B ", _thisA, _thisB)
                return _this_lambda
            if _dleft*_dcentral < 0:
                _lmax = _this_lambda
            else:
                _lmin = _this_lambda
            _this_lambda = (_lmax + _lmin ) / 2

    def scanInputLambdaRange_float64(self, estar, maxiter: int = 300, relative_tol = 0.1):
        assert(self.is_float64_initialised == True)
        assert(self.requires_MP[self.espace_dictionary[estar]] == False)
        #   Search the value of lambda for which A = B
        for _nl in range(self.lambda_config.ldensity):
            _thisA, _thisB = self._aggregate_lambdaCheckPointOne_float64(self.lambda_config.lspace[_nl], estar)
            print(LogMessage(), "lambda =", self.lambda_config.lspace[_nl], "A B",  _thisA, _thisB)

    def scanInputLambdaRange_MP(self, estar, eNorm_=False):
        tmax = self.par.tmax
        Wvec = np.zeros(self.lambda_config.ldensity)
        lstar_ID = 0
        Wstar = -1
        if eNorm_==False:
           Bnorm = self.matrix_bundle.bnorm
        if eNorm_==True:
           Bnorm = mp.fdiv(self.matrix_bundle.bnorm, estar)
           Bnorm = mp.fdiv(Bnorm, estar)
        for li in range(self.lambda_config.ldensity):
            mp_l = mpf(str(self.lambda_config.lspace[li]))
            scale = mpf(str(self.lambda_config.lspace[li]/(1-self.lambda_config.lspace[li])))
            scale = mp.fdiv(scale, Bnorm)
            scale = mp.fmul(scale, self.A0espace_dictionary[estar])
            W = self.matrix_bundle.B * scale
            W = W + self.matrix_bundle.S
            invW = W**(-1)  # slow!!!
            #   given W, get the coefficient
            gtestar = h_Et_mp_Eslice(invW, self.par, estar)
            Wvec[li] = float(gWg(self.matrix_bundle.S, self.matrix_bundle.B, gtestar, estar, mp_l, self.A0espace_dictionary[estar], self.matrix_bundle.bnorm, self.par, verbose=True))
            if Wvec[li] > Wstar:
                Wstar = Wvec[li]
                lstar_ID = li
        print(LogMessage(), 'Lambda ::: ', "Lambda* at E = ", float(estar), ' ::: ', self.lambda_config.lspace[lstar_ID])
        import matplotlib.pyplot as plt
        plt.plot(self.lambda_config.lspace, Wvec)
        plt.show()
        return self.lambda_config.lspace[lstar_ID]   #l*

