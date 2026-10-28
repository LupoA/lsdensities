import lsdensities.utils.common as u
from lsdensities.utils.common import init_precision, LogMessage, end, generate_seed
from lsdensities.utils.parser import parse_inputs
from lsdensities.utils.common import create_out_paths
from lsdensities.correlator.correlator_utils import symmetrisePeriodicCorrelator
from lsdensities.utils.parallel_utils import ParallelBootstrapLoop
from mpmath import mp, mpf
import random
import numpy as np
from lsdensities.inverse_problem_solvers.gaussian_process import (
    AlgorithmParameters,
    GaussianProcessWrapper,
)

read_SIGMA_ = True


def main():
    print(LogMessage(), "Initialising")
    par = parse_inputs()
    par.init()
    init_precision(par.prec)
    par.report()

    seed = generate_seed(par)
    random.seed(seed)
    np.random.seed(random.randint(0, 2 ** (32) - 1))

    #   Reading datafile, storing correlator
    rawcorr, par.time_extent, par.num_samples = u.read_datafile(par.datapath)
    par.assign_values()
    par.report()
    par.plotpath, par.logpath = create_out_paths(par)

    #   Folding the correlator (if applicable)
    rawcorr.evaluate()
    rawcorr.tmax = par.tmax
    if par.periodicity == "COSH":
        print(LogMessage(), "Folding correlator")
        symCorr = symmetrisePeriodicCorrelator(corr=rawcorr, par=par)
        symCorr.evaluate()

    #   #   #   Resampling
    if par.periodicity == "EXP":
        corr = u.Obs(
            T=par.time_extent, tmax=par.tmax, nms=par.num_boot, sample_type="bootstrap"
        )
        resample = ParallelBootstrapLoop(par, rawcorr.sample, is_folded=False)
    if par.periodicity == "COSH":
        corr = u.Obs(
            T=symCorr.T,
            tmax=symCorr.tmax,
            nms=par.num_boot,
            sample_type="bootstrap",
        )
        resample = ParallelBootstrapLoop(par, symCorr.sample, is_folded=False)

    corr.sample = resample.run()
    corr.evaluate()
    #   -   -   -   -   -   -   -   -   -   -   -

    #   Covariance
    print(LogMessage(), "Evaluate covariance")
    corr.evaluate_covmatrix(plot=False)
    corr.corrmat_from_covmat(plot=False)

    #   Make it into a mp sample
    print(LogMessage(), "Converting correlator into mpmath type")
    corr.fill_mp_sample()
    print(LogMessage(), "Cond[Cov C] = {:3.3e}".format(float(mp.cond(corr.mpcov))))

    cNorm = mpf(str(corr.central[1] ** 2))
    lambdaMax = 1e4
    energies = np.linspace(par.emin, par.emax, par.Ne)

    hltParams = AlgorithmParameters(
        alphaA=0,
        alphaB=1.99,
        alphaC=0.5,
        lambdaMax=lambdaMax,
        lambdaStep=lambdaMax / 2,
        lambdaScanCap=8,
        kfactor=0.1,
        lambdaMin=5e-2,
        comparisonRatio=0.3,
    )
    #   Wrapper for the Inverse Problem
    GP = GaussianProcessWrapper(
        par=par,
        algorithmPar=hltParams,
        B=corr.mpcov,
        bnorm=cNorm,
        correlator=corr,
        energies=energies,
        read_SIGMA=read_SIGMA_,
    )
    GP.prepareGP()

    #   Run
    GP.run()
    output_file = GP.save()
    print(LogMessage(), "Wrote stability-analysis data to", output_file)
    print(
        LogMessage(),
        "Plot it with: python3 plot_output.py stability --file",
        output_file,
        "--energy <E>",
    )
    end()


if __name__ == "__main__":
    main()
