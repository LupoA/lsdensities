import sys

sys.path.append("..")
from importall import *

Mpi = 0.5
MpiA = 0.5
MpiB = 1.0
Mpi_mp = mpf(str(Mpi))
MpiA_mp = mpf(str(MpiA))
MpiB_mp = mpf(str(MpiB))


def init_variables(args_):
    in_ = Inputs()
    in_.prec = args_.prec
    in_.time_extent = args_.T
    in_.tmax = args_.T - 1
    in_.outdir = args_.outdir
    in_.num_samples = args_.nms
    in_.num_boot = args_.nboot
    in_.sigma = args_.sigma
    in_.emax = args_.emax * Mpi
    in_.Ne = args_.ne
    # in_.l = args_.l
    in_.alpha = args_.alpha
    in_.emin = args_.emin
    in_.prec = -1
    in_.plots = args_.plots
    in_.assign_values()
    return in_


def main():
    print(LogMessage(), "Initialising")
    args = parseArgumentPeak()
    init_precision(args.prec)
    par = init_variables(args)
    espace = np.linspace(0.1, par.emax, par.Ne)
    espace_mp = mp.matrix(par.Ne, 1)
    for e_id in range(par.Ne):
        espace_mp[e_id] = mpf(str(espace[e_id]))
    par.report()
    T = par.time_extent
    tmax = par.tmax

    #   Generate corr
    corr = mp.matrix(tmax, 1)
    for t in range(tmax):
        corr[t] = mp.exp(-mpf(t + 1) * MpiA_mp) + mp.exp(-mpf(t + 1) * MpiB_mp)

    S = Smatrix_mp(tmax)
    invS = S ** (-1)
    diff = S * invS
    diff = norm2_mp(diff) - 1
    print(LogMessage(), "S/S - 1 ::: ", diff)

    ht = h_Et_mp(invS, par, espace_mp)
    rho = y_combine_central_mp(ht, corr, par)

    plt.plot(
        espace / Mpi,
        gauss_fp(espace, MpiA, par.sigma) + gauss_fp(espace, MpiB, par.sigma),
        color="k",
        linestyle="dashed",
        label="BG target",
    )
    plt.grid(visible=True, axis="both")
    plt.plot(
        espace / Mpi,
        rho,
        color="b",
        label=r"BG output with $\sigma/Mpi=${:2.2f}".format(par.sigma / Mpi),
    )
    plt.xlabel(r"$E/M_{pi}$")
    plt.legend(prop={"size": 12, "family": "Helvetica"})
    plt.tight_layout()
    plt.show()

    end()


if __name__ == "__main__":
    main()
