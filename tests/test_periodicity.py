from mpmath import mp, mpf
import sys
import numpy as np
import os
import math
sys.path.append("..")
from importall import *
from mpmath import mp, mpf


def scan_S_Value():
    S_list = []
    T_list = []
    upper_bound = int(1000)

    for i in range(10, upper_bound,4):
        tmax = int(i / 2 - 1)
        Scalc = Smatrix_mp(tmax, type='COSH', alpha_ = 0.0, T=i)
        S_list.append(Scalc[3,2])
        T_list.append(i)
        print("$S_{3,2}$ = ", float(Scalc[3,2]))

    Sinf = Smatrix_mp(upper_bound, alpha_ = 0.0 ,type='EXP')
    fixed_Sinf = Sinf[3,2]
    print("$S_{3,2}_{inf}$ = ", float(fixed_Sinf))

    import matplotlib.pyplot as plt

    plt.errorbar(
        x=np.array(T_list,dtype=float),
        y=np.array(S_list,dtype=float),
        yerr=0.0,
        marker="o",
        markersize=1.5,
        elinewidth=1.3,
        capsize=2,
        ls="",
        color=u.CB_color_cycle[0],
    )

    plt.ylim(0.0, 0.5)
    plt.xlabel("T", fontdict=u.timesfont)
    plt.ylabel("$S_{3,2}(T)$", fontdict=u.timesfont)
    plt.axhline(y=fixed_Sinf, color='blue', linestyle='--')
    plt.legend(prop={"size": 12, "family": "Helvetica"})
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join('./Smatrix_periodicity_check.png'))

def scan_ft_Value():
    ft_list = []
    T_list = []
    sigma = 0.4
    fixed_t = 3
    fixed_e = 2
    upper_bound = int(1000)

    for i in range(10,upper_bound,2):
        ftcalc = ft_mp(fixed_e, fixed_t, sigma, type='COSH', T=i)
        ft_list.append(ftcalc)
        T_list.append(i)
        print("$f_{3}$ = ", float(ftcalc))

    ftinf = ft_mp(fixed_e, fixed_t, sigma, type='EXP')
    print("$f_{3}_{inf}$ = ", float(ftinf))

    import matplotlib.pyplot as plt

    plt.errorbar(
        x=np.array(T_list,dtype=float),
        y=np.array(ft_list,dtype=float),
        yerr=0.0,
        marker="o",
        markersize=1.5,
        elinewidth=1.3,
        capsize=2,
        ls="",
        color=u.CB_color_cycle[0],
    )

    plt.ylim(0.005082, 0.00514)
    plt.xlabel("T", fontdict=u.timesfont)
    plt.ylabel("$f_{3}(T)$", fontdict=u.timesfont)
    plt.axhline(y=ftinf, color='blue', linestyle='--')
    plt.legend(prop={"size": 12, "family": "Helvetica"})
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join('./ft_periodicity_check.png'))


# Run tests
scan_S_Value()
scan_ft_Value()
