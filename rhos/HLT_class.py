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
import matplotlib.pyplot as plt

class AlgorithmParameters:
    def __init__(self, alphaA=0, alphaB=0, alphaC=0, lambdaMax=50, lambdaStep=0.5, lambdaScanPrec = 0.1, lambdaScanCap=6, kfactor = 0.1):
        self.alphaA = alphaA
        self.alphaB = alphaB
        self.alphaC = alphaC
        self.lambdaMax = lambdaMax
        self.lambdaStep = lambdaStep
        self.lambdaScanPrec = lambdaScanPrec
        self.lambdaScanCap = lambdaScanCap
        self.kfactor = kfactor
        self.alphaAmp = mpf(str(alphaA))
        self.alphaBmp = mpf(str(alphaB))
        self.alphaCmp = mpf(str(alphaC))

class A0_t:
    def __init__(self, par_ : Inputs ,alphaMP_=0, eminMP_=0):
        self.valute_at_E = mp.matrix(par_.Ne, 1)
        self.valute_at_E_dictionary = {}  # Auxiliary dictionary: A0espace[n] = A0espace_dictionary[espace[n]] # espace must be float
        self.is_filled = False
        self.alphaMP = alphaMP_
        self.eminMP = eminMP_
        self.par = par_
    def evaluate(self, espaceMP_):
        print(LogMessage(), "Computing A0 at all energies with Alpha = {:2.2e}".format(float(self.alphaMP)))
        self.valute_at_E = A0E_mp(espaceMP_, self.par, alpha_= self.alphaMP, emin_=self.eminMP)
        for e_id in range(self.par.Ne):
            self.valute_at_E_dictionary[float(espaceMP_[e_id])] = self.valute_at_E[e_id]
        self.is_filled = True

class HLTWrapper:
    def __init__(self, par: Inputs, algorithmPar: AlgorithmParameters, matrix_bundle: MatrixBundle, correlator : Obs):
        self.par = par
        self.correlator = correlator
        self.algorithmPar = algorithmPar
        self.matrix_bundle = matrix_bundle
        #
        self.espace = np.linspace(par.emin, par.emax, par.Ne)
        self.alphaMP = mpf(str(par.alpha))
        self.e0MP = mpf(str(par.e0))
        self.espaceMP = mp.matrix(par.Ne, 1)
        self.sigmaMP = mpf(str(par.sigma))
        self.emaxMP = mpf(str(par.emax))
        self.eminMP = mpf(str(par.emin))
        self.espace_dictionary = {} #   Auxiliary dictionary: espace_dictionary[espace[n]] = n
        #
        self.A0_A = A0_t(alphaMP_=self.algorithmPar.alphaAmp, eminMP_=self.eminMP, par_=self.par)
        self.A0_B = A0_t(alphaMP_=self.algorithmPar.alphaBmp, eminMP_=self.eminMP, par_=self.par)
        self.A0_C = A0_t(alphaMP_=self.algorithmPar.alphaCmp, eminMP_=self.eminMP, par_=self.par)
        self.selectA0 = {}
        self.selectA0[0] = self.A0_A
        self.selectA0[-1] = self.A0_B
        self.selectA0[-1.99] = self.A0_C
        #
        self.espace_is_filled = False
        self.labda_check = self.algorithmPar.kfactor
        #   Lists of result
        self.rho_list = []
        self.drho_list = []
        self.gAA0g_list = []
        self.lambda_list = []
        self.rho_list_alpha2 = []
        self.drho_list_alpha2 = []
        self.gAA0g_list_alpha2 = []

        self.lambda_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.drho_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.result_is_filled = np.full(par.Ne, False, dtype=bool)
        self.rho_sys_err = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_quadrature_err = np.ndarray(self.par.Ne, dtype=np.float64)

    def fillEspaceMP(self):
        for e_id in range(self.par.Ne):
            self.espaceMP[e_id] = mpf(str(self.espace[e_id]))
            self.espace_dictionary[self.espace[e_id]] = e_id
        self.espace_is_filled = True

    def prepareHLT(self):
        self.fillEspaceMP()
        self.A0_A.evaluate(self.espaceMP)
        if self.algorithmPar.alphaB != 0:
            self.A0_B.evaluate(self.espaceMP)
        if self.algorithmPar.alphaC != 0:
            self.A0_C.evaluate(self.espaceMP)

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

    def lambdaToRho(self, lambda_, estar_, alpha_=0):
        import time
        assert ( self.A0_A.is_filled == True)
        _Bnorm = (self.correlator.central[1]*self.correlator.central[1])/(estar_*estar_)
        _factor = (lambda_ * self.selectA0[alpha_].valute_at_E_dictionary[estar_]) / _Bnorm
        _M = self.matrix_bundle.S + (_factor*self.matrix_bundle.B)
        start_time = time.time()
        _Minv = invert_matrix_ge(_M)
        end_time = time.time()
        print(LogMessage(), 'lambdaToRho ::: Matrix inverted in ', end_time - start_time, 's')
        _g_t_estar = h_Et_mp_Eslice(_Minv, self.par, estar_,  alpha_=alpha_)
        rho_estar, drho_estar = y_combine_sample_Eslice_mp(_g_t_estar, self.correlator.sample, self.par)

        gag_estar = gAg(self.matrix_bundle.S, _g_t_estar, estar_, self.par)

        return rho_estar, drho_estar, gag_estar

    def scanLambda(self, estar_, alpha_=0):
        lambda_ = self.algorithmPar.lambdaMax
        lambda_step = self.algorithmPar.lambdaStep
        prec_ = self.algorithmPar.lambdaScanPrec
        cap_ = self.algorithmPar.lambdaScanCap
        _count = 0

        print(LogMessage(), ' --- ')
        print(LogMessage(), 'Scan Lambda at energy {:2.2e}'.format(estar_))

        print(LogMessage(), 'Scan Lambda ::: Lambda = ', lambda_)
        _this_rho, _this_drho, _this_gAg = self.lambdaToRho(lambda_, estar_)   #   _this_drho will remain the first one
        self.rho_list.append(_this_rho)      #   store
        self.drho_list.append(_this_drho)      #   store
        self.gAA0g_list.append(_this_gAg/self.selectA0[alpha_].valute_at_E_dictionary[estar_])  #   store
#        self.lambda_list.append(lambda_)
        lambda_ -= lambda_step

        while (_count < cap_ and lambda_ > 0):

            print(LogMessage(), 'Scan Lambda ::: Lambda = {:1.3e}'.format(float(lambda_)))
            print(LogMessage(), 'Scan Lambda ::: Lambda (old) = {:1.3e}'.format(float(lambda_/(1+lambda_))))
            _this_updated_rho, _this_updated_drho, _this_gAg = self.lambdaToRho(lambda_, estar_,  alpha_=alpha_)
            self.rho_list.append(_this_updated_rho)  #   store
            print(LogMessage(), 'Scan Lambda ::: Rho = {:1.3e}'.format(float(_this_updated_rho)))
            self.drho_list.append(_this_updated_drho)    #   store
            self.gAA0g_list.append(_this_gAg/self.selectA0[alpha_].valute_at_E_dictionary[estar_])  #   store
#            self.lambda_list.append(lambda_)
            _residual = abs((_this_updated_rho - _this_rho) / (_this_updated_drho))
            print(LogMessage(), 'Scan Lambda ::: Residual = ', float(_residual))
            if (_residual < prec_):
                _count += 1
                print(LogMessage(), 'Scan Lambda ::: count = ', _count)
            else:
                _count = 0

            _this_rho = _this_updated_rho
            lambda_ -= lambda_step

        self.lambda_result[self.espace_dictionary[estar_]] = lambda_
        self.rho_result[self.espace_dictionary[estar_]] = _this_rho
        self.drho_result[self.espace_dictionary[estar_]] = _this_updated_drho
        self.result_is_filled[self.espace_dictionary[estar_]] = True

        print(LogMessage(), 'Scan Lambda ::: Lambda * = ', lambda_)

        return self.rho_list, self.drho_list, self.gAA0g_list

    def scanLambdaAlpha(self, estar_):
        lambda_ = self.algorithmPar.lambdaMax
        lambda_step = self.algorithmPar.lambdaStep
        prec_ = self.algorithmPar.lambdaScanPrec
        cap_ = self.algorithmPar.lambdaScanCap
        _count = 0

        print(LogMessage(), ' --- ')
        print(LogMessage(), 'At Energy {:2.2e}'.format(estar_))
        print(LogMessage(), 'Scan Lambda ::: Lambda (0,inf) = {:1.3e}'.format(float(lambda_)))
        print(LogMessage(), 'Scan Lambda ::: Lambda (0,1) = {:1.3e}'.format(float(lambda_ / (1 + lambda_))))

        # Setting alpha to the first value
        print(LogMessage(), 'Setting Alpha ::: First Alpha = ', float(self.algorithmPar.alphaA))
        _this_rho, _this_drho, _this_gAg = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaAmp)   #   _this_drho will remain the first one
        self.rho_list.append(_this_rho)      #   store
        self.drho_list.append(_this_drho)      #   store
        self.gAA0g_list.append(_this_gAg/self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[estar_])  #   store
        self.lambda_list.append(lambda_)    #   store

        # Setting alpha to the second value
        print(LogMessage(), 'Setting Alpha ::: Second Alpha = ', float(self.algorithmPar.alphaB))
        _this_rho2, _this_drho2, _this_gAg2 = self.lambdaToRho(lambda_, estar_,  self.algorithmPar.alphaBmp)   #   _this_drho will remain the first one
        self.rho_list_alpha2.append(_this_rho2)      #   store
        self.drho_list_alpha2.append(_this_drho2)      #   store
        self.gAA0g_list_alpha2.append(_this_gAg2/self.selectA0[self.algorithmPar.alphaB].valute_at_E_dictionary[estar_])  #   store

        lambda_ -= lambda_step

        while (_count < cap_ and lambda_ > 0):
            print(LogMessage(), 'Scan Lambda ::: Lambda (0,inf) = {:1.3e}'.format(float(lambda_)))
            print(LogMessage(), 'Scan Lambda ::: Lambda (0,1) = {:1.3e}'.format(float(lambda_/(1+lambda_))))

            print(LogMessage(), 'Setting Alpha ::: First Alpha = ', self.algorithmPar.alphaA)
            _this_updated_rho, _this_updated_drho, _this_gAg = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaAmp)
            self.rho_list.append(_this_updated_rho)  #   store
            print(LogMessage(), 'Scan Lambda ::: Rho (Alpha = {:2.2e}) '.format(self.algorithmPar.alphaA),  '= {:1.3e}'.format(float(_this_updated_rho)), 'Stat = {:1.3e}'.format(float(_this_updated_drho)))
            self.drho_list.append(_this_updated_drho)    #   store
            self.gAA0g_list.append(_this_gAg/self.selectA0[self.algorithmPar.alphaA].valute_at_E_dictionary[estar_])  #   store
            self.lambda_list.append(lambda_)
            _residual1 = abs((_this_updated_rho - _this_rho) / (_this_updated_drho))
            print(LogMessage(), 'Scan Lambda ::: Residual = ', float(_residual1))

            print(LogMessage(), 'Setting Alpha ::: Second Alpha = ', self.algorithmPar.alphaB)
            _this_updated_rho2, _this_updated_drho2, _this_gAg2 = self.lambdaToRho(lambda_, estar_, self.algorithmPar.alphaBmp)
            self.rho_list_alpha2.append(_this_updated_rho2)  # store
            print(LogMessage(), 'Scan Lambda ::: Rho (Alpha = {:2.2e}) '.format(self.algorithmPar.alphaB), '= {:1.3e}'.format(float(_this_updated_rho2)), 'Stat = {:1.3e}'.format(float(_this_updated_drho2)))
            self.drho_list_alpha2.append(_this_updated_drho2)  # store
            self.gAA0g_list_alpha2.append(_this_gAg2 / self.selectA0[self.algorithmPar.alphaB].valute_at_E_dictionary[estar_])  # store
            _residual2 = abs((_this_updated_rho2 - _this_rho2) / (_this_updated_drho2))
            print(LogMessage(), 'Scan Lambda ::: Residual = ', float(_residual2))

            #comp_quadrature = abs(_this_updated_rho - _this_updated_rho2) /  mp.sqrt( mp.fadd(mp.fmul(_this_updated_drho,_this_updated_drho),mp.fmul(_this_updated_drho2,_this_updated_drho2)) )
            comp_diff = abs(_this_updated_rho - _this_updated_rho2) - (_this_updated_drho + _this_updated_drho2)
            print(LogMessage(), 'Scan Lambda ::: Alpha Diff ::: ', float(comp_diff))
            print(LogMessage(), 'check', -(_this_updated_drho2))
            if (_residual1 < prec_ and _residual2 < prec_ and comp_diff < -(_this_updated_drho2*0.1)):
                _count += 1
                print(LogMessage(), 'Scan Lambda ::: count = ', _count)
            else:
                _count = 0

            _this_rho = _this_updated_rho
            _this_rho2 = _this_updated_rho2
            lambda_ -= lambda_step

        self.lambda_result[self.espace_dictionary[estar_]] = lambda_
        self.rho_result[self.espace_dictionary[estar_]] = (_this_rho + _this_rho2) / 2
        self.drho_result[self.espace_dictionary[estar_]] = _this_updated_drho
        self.result_is_filled[self.espace_dictionary[estar_]] = True

        print(LogMessage(), 'Scan Lambda ::: Lambda * = ', lambda_)

        return self.rho_list, self.drho_list, self.gAA0g_list, self.rho_list_alpha2, self.drho_list_alpha2, self.gAA0g_list_alpha2

    def estimate_sys_error(self, estar_):
        assert (self.result_is_filled[self.espace_dictionary[estar_]] == True)

        _this_y = self.rho_result[self.espace_dictionary[estar_]] #   rho at lambda*
        _that_y, _that_yerr, _that_x = self.lambdaToRho(self.lambda_result[self.espace_dictionary[estar_]] * self.algorithmPar.kfactor, estar_)

        self.rho_sys_err[self.espace_dictionary[estar_]] = abs(_this_y - _that_y) / 2
        self.rho_quadrature_err[self.espace_dictionary[estar_]] = np.sqrt(self.rho_sys_err[self.espace_dictionary[estar_]]**2 + self.drho_result[self.espace_dictionary[estar_]]**2)

        return self.rho_sys_err[self.espace_dictionary[estar_]]

    def run(self, show_lambda_scan=False):
        for e_i in range(self.par.Ne):  # finds solution at a specific lambda

            rho_l, drho_l, gag_l = self.scanLambda(self.espace[e_i])

            _ = self.estimate_sys_error(self.espace[e_i])

            if show_lambda_scan==True:
                import matplotlib.pyplot as plt
                plt.errorbar(
                    x=gag_l,
                    y=rho_l,
                    yerr=drho_l,
                    marker="o",
                    markersize=1.5,
                    elinewidth=1.3,
                    capsize=2,
                    ls="",
                    label=r'$\rho({:2.2e)$'.format(self.espace[e_i]) + r'$(\sigma = {:2.2f})$'.format(self.par.sigma / self.par.massNorm) + r'$M_\pi$',
                    color=u.CB_color_cycle[0],
                )
                plt.xlabel(r"$A[g_\lambda] / A_0$", fontdict=u.timesfont)
                # plt.ylabel("Spectral density", fontdict=u.timesfont)
                plt.legend(prop={"size": 12, "family": "Helvetica"})
                plt.grid()
                plt.tight_layout()
                plt.show()

    def plotRhoOverLambda(self, estar: float):



        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        plt.title(r"$E/M_{\pi}$" + "= {:2.2f}  ".format(estar / self.par.massNorm) + r" $\sigma$" + " = {:2.2f} Mpi".format(self.par.sigma / self.par.massNorm))
        ax[0].axhspan(
            ymin=self.rho[self.espace_dictionary[estar]] - self.rho_quadrature_err[self.espace_dictionary[estar]],
            ymax=self.rho[self.espace_dictionary[estar]] + self.rho_quadrature_err[self.espace_dictionary[estar]],
            alpha=0.3,
            color=CB_colors[4]
            )
        ax[0].errorbar(
            x=self.lambda_config.lspace,
            y=_rho_l,
            yerr=_rho_stat_l,
            marker=plot_markers[0],
            markersize=1.8,
            elinewidth=1.3,
            capsize=2,
            ls="",
            color=CB_colors[0],
        )
        ax[0].errorbar(
            x=self.optimal_lambdas[self.espace_dictionary[estar],0],
            y=self.rho_kfact_dictionary[self.lambda_config.k_star][estar][0],
            yerr=self.rho_kfact_dictionary[self.lambda_config.k_star][estar][1],
            marker=plot_markers[1],
            markersize=3.8,
            elinewidth=1.3,
            capsize=3,
            ls="",
            label=r"$A[g_\lambda] / A_0 = B[g_\lambda] $",
            color=CB_colors[1],
        )
        ax[0].errorbar(
            x=self.optimal_lambdas[self.espace_dictionary[estar],1],
            y=self.rho_kfact_dictionary[self.lambda_config.kfactor][estar][0],
            yerr=self.rho_kfact_dictionary[self.lambda_config.kfactor][estar][1],
            marker=plot_markers[2],
            markersize=3.8,
            elinewidth=1.3,
            capsize=3,
            ls="",
            label=r"$A[g_\lambda] / A_0 = {:2.1f} B[g_\lambda] $".format(self.lambda_config.kfactor),
            color=CB_colors[2],
        )
        ax[0].set_xlabel(r"$\lambda$", fontdict=timesfont)
        ax[0].set_ylabel(r"$\rho_\sigma$", fontdict=timesfont)
        ax[0].legend(prop={"size": 12, "family": "Helvetica"})
        ax[0].grid()

        # Second subplot with A/A_0
        ax[1].errorbar(
            x=_A_l,
            y=_rho_l,
            yerr=_rho_stat_l,
            marker="+",
            markersize=1.8,
            elinewidth=1.3,
            capsize=2,
            ls="",
            label=r"$E/M_{\pi}$" + "= {:2.2f}".format(
                estar / self.par.massNorm) + r"$\sigma$" + " = {:2.2f} Mpi".format(self.par.sigma / self.par.massNorm),
            color=CB_colors[0],
        )
        ax[1].set_xlabel(r"$A[g_\lambda] / A_0$", fontdict=timesfont)
        ax[1].set_ylabel(r"$\rho_\sigma$", fontdict=timesfont)
        ax[1].legend(prop={"size": 12, "family": "Helvetica"})
        ax[1].grid()

        ax[1].axhspan(ymin=self.rho[self.espace_dictionary[estar]] - self.rho_quadrature_err[self.espace_dictionary[estar]],
                         ymax=self.rho[self.espace_dictionary[estar]] + self.rho_quadrature_err[self.espace_dictionary[estar]],
                         alpha=0.3,
                         color = CB_colors[4]
                         )
       # plt.fill_between(self.rho[self.espace_dictionary[estar]] - self.rho_quadrature_err[self.espace_dictionary[estar]],
        #                 self.rho[self.espace_dictionary[estar]] + self.rho_quadrature_err[self.espace_dictionary[estar]],
        #                 color=CB_colors[4], alpha=0.3)

        plt.tight_layout()
        plt.show()