from lsdensities.core import Smatrix_mp
from lsdensities.transform import ft_mp


def test_Smat_convergence():
    S_list = []
    T_list = []
    upper_bound = int(500)

    Sinf = Smatrix_mp(upper_bound, alpha_=0.0, type="EXP")
    fixed_Sinf = float(Sinf[3, 2])
    print("$S_{3,2}_{inf}$ = ", fixed_Sinf)

    for i in range(100, upper_bound, 100):
        tmax = int(i / 2 - 1)
        Scalc = Smatrix_mp(tmax, type="COSH", alpha_=0.0, T=i)
        S_list.append(float(Scalc[3, 2]))
        T_list.append(i)
        print("$S_{3,2}$ = ", float(Scalc[3, 2]))

    for i in range(1, len(S_list)):
        assert abs(S_list[i] - fixed_Sinf) < abs(
            S_list[i - 1] - fixed_Sinf
        ), f"Failed at T={T_list[i]}"

    assert abs(S_list[-1] - fixed_Sinf) < 1e-2


def test_ft_convergence():
    ft_list = []
    T_list = []
    sigma = 0.4
    fixed_t = 3
    fixed_e = 2
    upper_bound = int(200)

    ftinf = ft_mp(fixed_e, fixed_t, sigma, alpha=0, type="EXP")
    print("$f_{3}_{inf}$ = ", float(ftinf))

    for i in range(10, upper_bound, 4):
        ftcalc = ft_mp(fixed_e, fixed_t, sigma, alpha=0, type="COSH", T=i)
        ft_list.append(ftcalc)
        T_list.append(i)
        print("$f_{3}$ = ", float(ftcalc))

    for i in range(1, len(ft_list)):
        assert abs(ft_list[i] - ftinf) < abs(
            ft_list[i - 1] - ftinf
        ), f"Failed at T={T_list[i]}"

    assert abs(ft_list[-1] - ftinf) < 1e-6
