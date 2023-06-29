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

class HLTWrapper:
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
        #   Lambda utilities
        self.labda_check = self.lambda_config.kfactor

        #self.optimal_lambdas = np.ndarray((self.par.Ne, 2), dtype=np.float64)
        self.optimal_lambdas_is_filled = False
        #   Lists of result
        self.rho_list = []
        self.drho_list = []
        self.gAA0g_list = []

        self.lambda_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.drho_result = np.ndarray(self.par.Ne, dtype=np.float64)
        self.result_is_filled = np.full(par.Ne, False, dtype=bool)
        self.rho_sys_err = np.ndarray(self.par.Ne, dtype=np.float64)
        self.rho_quadrature_err = np.ndarray(self.par.Ne, dtype=np.float64)

        #   Result, float64
        #self.rho = np.ndarray(self.par.Ne, dtype = np.float64)
        #self.rho_stat_err = np.ndarray(self.par.Ne, dtype=np.float64)

        #
        #self.rho_kfact_dictionary = {}


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

    def lambdaToRho(self, lambda_, estar_):
        import time
        assert ( self.A0_is_filled == True)
        _Bnorm = (self.correlator.central[1]*self.correlator.central[1])/(estar_*estar_)
        #_Bnorm = (self.correlator.central[1] * self.correlator.central[1]) / 1
        _factor = (lambda_ * self.A0espace_dictionary[estar_]) / _Bnorm
        _M = self.matrix_bundle.S + (_factor*self.matrix_bundle.B)
        start_time = time.time()
        _Minv = invert_matrix_ge(_M)
        end_time = time.time()
        print(LogMessage(), 'lambdaToRho ::: Matrix inverted in ', end_time - start_time, 's')
        _g_t_estar = h_Et_mp_Eslice(_Minv, self.par, estar_)
        rho_estar, drho_estar = y_combine_sample_Eslice_mp(_g_t_estar, self.correlator.sample, self.par)

        gag_estar = gAg(self.matrix_bundle.S, _g_t_estar, estar_, self.par)

        return rho_estar, drho_estar, gag_estar

    def scanLambda(self, estar_, prec_ = 0.009):
        lambda_ = 120
        lambda_step = 0.5
        #rho_list = []
        #drho_list = []
        #gAA0g_list = []
        _count = 0

        print(LogMessage(), ' --- ')
        print(LogMessage(), 'Scan Lambda at energy {:2.2e}'.format(estar_))

        print(LogMessage(), 'Scan Lambda ::: Lambda = ', lambda_)
        _this_rho, _this_drho, _this_gAg = self.lambdaToRho(lambda_, estar_)   #   _this_drho will remain the first one
        self.rho_list.append(_this_rho)      #   store
        self.drho_list.append(_this_drho)      #   store
        self.gAA0g_list.append(_this_gAg/self.A0espace_dictionary[estar_])  #   store
        lambda_ -= lambda_step

        while (_count < 6 and lambda_ > 1e-4):
        #while (lambda_ > 0):

            print(LogMessage(), 'Scan Lambda ::: Lambda = {:1.3e}'.format(float(lambda_)))
            print(LogMessage(), 'Scan Lambda ::: Lambda (old) = {:1.3e}'.format(float(lambda_/(1+lambda_))))
            _this_updated_rho, _this_updated_drho, _this_gAg = self.lambdaToRho(lambda_, estar_)
            self.rho_list.append(_this_updated_rho)  #   store
            print(LogMessage(), 'Scan Lambda ::: Rho = {:1.3e}'.format(float(_this_updated_rho)))
            self.drho_list.append(_this_updated_drho)    #   store
            self.gAA0g_list.append(_this_gAg/self.A0espace_dictionary[estar_])  #   store
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

    def estimate_sys_error(self, estar_):
        assert (self.result_is_filled[self.espace_dictionary[estar_]] == True)

        #_this = self.rho_kfact_dictionary[self.lambda_config.k_star][estar_]  # input: e and lambda* ; out = rho

        _this_y = self.rho_result[self.espace_dictionary[estar_]] #   rho at lambda*

        _that_y, _that_yerr, _that_x = self.lambdaToRho(self.lambda_result[self.espace_dictionary[estar_]] * self.lambda_config.kfactor, estar_)


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







    def _aggregate_HLT_lambda_float64(self, lambda_float64: float, estar: float):  #   allows to cast as a funciton of lambda only at each estar
        factor = (self.A0espace_float_dictionary[estar] * lambda_float64) / ((1 - lambda_float64) * self.bnorm_float)
        self.W_float = (self.B_float * factor) + self.S_float
        #Winv = np.linalg.inv(self.W_float)
        Winv = choelesky_invert_scipy(self.W_float)
        gt = h_Et_mp_Eslice_float64(Winv, self.par, estar_=estar)
        _rho, _drho = y_combine_sample_Eslice_float64(gt, self.correlator.sample, self.par)
        this_A = gAgA0_float64(self.S_float, gt, estar, self.par, self.A0_float[self.espace_dictionary[estar]])
        this_B = gBg_float64(gt, self.B_float, self.bnorm_float, self.par.tmax)
        return this_A, this_B, _rho, _drho

    def scanInputLambdaRange_MP(self, estar, eNorm_=False):
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

    def solveHLT_fromLambdaList_float64(self, estar: float):
        _rho_l = np.ndarray(self.lambda_config.ldensity, dtype=np.float64)
        _rho_stat_l = np.ndarray(self.lambda_config.ldensity, dtype=np.float64)
        _A_l = np.ndarray(self.lambda_config.ldensity, dtype=np.float64)
        for _l in range(self.lambda_config.ldensity):
            _A_l[_l], _none, _rho_l[_l], _rho_stat_l[_l] = self._aggregate_HLT_lambda_float64(self.lambda_config.lspace[_l], estar)

        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        # First subplot with self.lambda_config.lspace
        plt.title(r"$E/M_{\pi}$" + "= {:2.2f}  ".format(
            estar / self.par.massNorm) + r" $\sigma$" + " = {:2.2f} Mpi".format(self.par.sigma / self.par.massNorm))
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



    def run_deprecated(self, show_lambda_scan=False):
        for e_i in range(self.par.Ne):  # finds solution at a specific lambda
            self.optimal_lambdas[e_i][0] = self.solveHLT_bisectonSearch_float64(self.espace[e_i],
                                                                              k_factor=self.lambda_config.k_star)

        for e_i in range(self.par.Ne):  # finds solution at another lambda
            self.optimal_lambdas[e_i][1] = self.solveHLT_bisectonSearch_float64(self.espace[e_i],
                                                                              k_factor=self.lambda_config.kfactor)

        self.optimal_lambdas_is_filled = True
        assert all(self.result_is_filled)

        self.estimate_sys_error()  # uses both solution to estimate systematics due to lambda

        if show_lambda_scan==True:
            for e_i in range(self.par.Ne):
                self.solveHLT_fromLambdaList_float64(self.espace[e_i])