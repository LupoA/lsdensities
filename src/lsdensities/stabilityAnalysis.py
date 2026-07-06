"""
Shared building blocks for the stability analysis (arXiv:2605.14652, Sec. III.A):
the Backus-Gilbert regulator is scanned over the trade-off parameter lambda,
optionally cross-checked at several values of the kernel exponent alpha
(Fig. 6, top panel), until the reconstructed smeared spectral density
stabilises within its statistical error.

Used by both InverseProblemWrapper (frequentist/HLT) and GaussianProcessWrapper
(Bayesian): the two differ in how the regulated matrix and its associated prior
are built, but share the parameter bookkeeping and the logic that cross-checks
the alpha channels below.
"""

from mpmath import mp, mpf
from .core import a0_array
from . import ioutils
from .utils.rhoUtils import Inputs, log


class AlgorithmParameters:
    """
    Parameters controlling the stability-analysis scan over the Backus-Gilbert
    trade-off parameter lambda (arXiv:2605.14652, Sec. III.A).

    alphaA, alphaB, alphaC : values of the smearing-kernel exponent used to
        cross-check the lambda -> 0 limit (Fig. 6, top panel). Only alphaA is
        required; set par.Na to 1, 2 or 3 to also use alphaB and/or alphaC.
    lambdaMax : Starting value for the scan over lambda.
    lambdaMin : Ending value, unless stopping condition is met.
    lambdaStep : Step in lambda.
    lambdaScanCap : Search for the plateau stops after lambdaScanCap subsequent compatible measurements.
    plateau_id : number of consecutive compatible steps required before a result is flagged as the plateau.
    kfactor : Systematics on value at lambda(reference) are estimated by repeating the calculation at lambda = kfactor lambda(reference).
    resize : if lambda hits zero before hitting lambdaMin, the step is resized. Allows sampling values of lambda at different scales.
    comparisonRatio : Measurements at different lambda are considered compatible if they agree within comparisonRatio * uncertainty (1sigma if comparisonRatio = 1).
    """

    def __init__(
        self,
        alphaA=0,
        alphaB=1 / 2,
        alphaC=1.99,
        lambdaMax=50,
        lambdaStep=0.5,
        lambdaScanCap=6,
        plateau_id=1,
        kfactor=0.1,
        lambdaMin=1e-6,
        comparisonRatio=1,
        resize=4,
    ):
        assert alphaA != alphaB
        assert alphaA != alphaC
        self.alphaA = float(alphaA)
        self.alphaB = float(alphaB)
        self.alphaC = float(alphaC)
        self.lambdaMax = lambdaMax
        self.lambdaStep = lambdaStep
        self.lambdaScanCap = lambdaScanCap
        self.plateau_id = plateau_id
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
    """Functional A[g=0] (arXiv:2605.14652, the reference/unregularised norm), at every energy."""

    def __init__(self, par: Inputs, alpha=0, emin=0):
        self.valute_at_E = mp.matrix(par.Ne, 1)
        self.valute_at_E_dictionary = {}  # Auxiliary dictionary: A0espace[n] = A0espace_dictionary[espace[n]] # espace must be float
        self.is_filled = False
        self.alphaMP = alpha
        self.eminMP = emin
        self.par = par

    def evaluate(self, espace_mp):
        log(
            "Computing A0 at all energies with Alpha = {:2.2e}".format(
                float(self.alphaMP)
            ),
        )
        self.valute_at_E = a0_array(espace_mp, self.par, alpha=self.alphaMP)
        for e_id in range(self.par.Ne):
            self.valute_at_E_dictionary[float(espace_mp[e_id])] = self.valute_at_E[e_id]
        self.is_filled = True


def are_ranges_compatible(x, deltax, y, deltay):
    """Checks whether the ranges [x-deltax, x+deltax] and [y-deltay, y+deltay] overlap."""
    return (x - deltax) <= (y + deltay) and (y - deltay) <= (x + deltax)


class AlphaChannel:
    """
    Bookkeeping for one value of the smearing-kernel exponent alpha.

    "A" is the primary channel: it drives the plateau search in lambda.
    Any additional channels ("B", "C") are only used to cross-check that the
    primary channel's result is stable against a change of smearing kernel
    (arXiv:2605.14652, Fig. 6 top panel), and never themselves determine the
    flagged lambda*.
    """

    def __init__(self, label: str, alpha_mp, a0: A0_t, sigma_matrix, num_energies: int):
        self.label = label
        self.alpha_mp = alpha_mp
        self.alpha = float(alpha_mp)
        self.a0 = a0
        self.sigma_matrix = sigma_matrix
        self.rho_list = [[] for _ in range(num_energies)]
        self.errBoot_list = [[] for _ in range(num_energies)]
        self.errBayes_list = [[] for _ in range(num_energies)]
        self.likelihood_list = [[] for _ in range(num_energies)]


def scan_secondary_channels(
    wrapper,
    lambda_,
    estar_,
    secondary_channels,
    previous,
    primary_rho,
    primary_errBoot,
    flagged_rho,
    flagged_errBoot,
    comp_ratio,
):
    """
    Evaluate every non-primary alpha channel at this lambda, store the result,
    and check the two per-channel cross-checks of arXiv:2605.14652 (Fig. 6, top
    panel): agreement with the primary channel at this same lambda, and
    agreement with this channel's own result at the previous lambda step.

    `previous` is a dict {label: (rho, errBoot)} holding each channel's result
    from the previous scan step; it is updated in place with this step's
    results, so the comparison is always against the immediately preceding
    lambda (matching how the primary channel tracks its own history).

    Returns (skip, flagged_overlap): skip is True if any secondary channel
    disagrees with the primary channel (at this or the previous lambda), and
    flagged_overlap is a dict {label: bool} recording whether each channel's
    result agrees with the primary channel's currently-flagged (candidate
    plateau) result -- checked independently per channel by the caller.
    """
    skip = False
    flagged_overlap = {}
    for ch in secondary_channels:
        rho, errBayes, errBoot, likelihood, _, _ = wrapper.lambdaToRho(
            lambda_, estar_, ch.alpha_mp
        )
        wrapper._store(estar_, rho, errBayes, errBoot, likelihood, ch)

        overlap_with_primary = are_ranges_compatible(
            primary_rho, primary_errBoot, rho, errBoot
        )
        prev_rho, prev_errBoot = previous[ch.label]
        overlap_with_previous = are_ranges_compatible(
            rho, comp_ratio * errBoot, prev_rho, comp_ratio * prev_errBoot
        )
        flagged_overlap[ch.label] = are_ranges_compatible(
            rho, comp_ratio * errBoot, flagged_rho, comp_ratio * flagged_errBoot
        )

        if not overlap_with_primary:
            log(f"\t Primary Alpha and {ch.label} Alpha not compatible")
            skip = True
            if not overlap_with_previous:
                log(
                    f"\t Result at {ch.label} Alpha not compatible with previous lambda at {ch.label} Alpha"
                )
                skip = True

        previous[ch.label] = (rho, errBoot)

    return skip, flagged_overlap


def _gt_to_list(gt):
    """Coefficient vector (mp.matrix, tmax x 1) as a plain list of floats, or None if not yet computed."""
    if gt is None:
        return None
    return [float(gt[i]) for i in range(gt.rows)]


def save_stability_output(wrapper, method, path=None):
    """
    Writes the full stability-analysis scan (arXiv:2605.14652 Fig. 6: rho and
    its statistical/Bayesian error, scanned over lambda for every alpha
    channel) plus the flagged plateau/NLL-minimum results, to a single JSON
    file (see ioutils.py for the shared schema). Works for both
    InverseProblemWrapper ("HLT") and GaussianProcessWrapper ("GP") since they
    share the same channel/result attributes.
    """
    channels = [wrapper.channelA, *wrapper.secondary_channels]

    metadata = ioutils.base_metadata(wrapper.par, method)
    metadata["alphas"] = {ch.label: ch.alpha for ch in channels}
    metadata["algorithm"] = {
        "lambdaMax": wrapper.algorithmPar.lambdaMax,
        "lambdaStep": wrapper.algorithmPar.lambdaStep,
        "lambdaMin": wrapper.algorithmPar.lambdaMin,
        "lambdaScanCap": wrapper.algorithmPar.lambdaScanCap,
        "comparisonRatio": wrapper.algorithmPar.comparisonRatio,
        "kfactor": wrapper.algorithmPar.kfactor,
        "resize": wrapper.algorithmPar.resize,
        "plateau_id": wrapper.algorithmPar.plateau_id,
    }

    energies = []
    for e_i in range(wrapper.par.Ne):
        scan = {
            ch.label: {
                "lambda": [float(x) for x in wrapper.lambda_list[e_i]],
                "rho": [float(x) for x in ch.rho_list[e_i]],
                "errBoot": [float(x) for x in ch.errBoot_list[e_i]],
                "errBayes": [float(x) for x in ch.errBayes_list[e_i]],
                "likelihood": [float(x) for x in ch.likelihood_list[e_i]],
            }
            for ch in channels
        }
        energies.append(
            {
                "energy": float(wrapper.espace[e_i]),
                "scan": scan,
                "result": {
                    "HLT": {
                        "lambda_star": float(wrapper.lambdaResultHLT[e_i]),
                        "rho": float(wrapper.rhoResultHLT[e_i]),
                        "stat_err": float(wrapper.drho_result[e_i]),
                        "sys_err": float(wrapper.rho_sys_err_HLT[e_i]),
                        "quadrature_err": float(wrapper.rho_quadrature_err_HLT[e_i]),
                        "aa0": float(wrapper.aa0[e_i]),
                        "gt": _gt_to_list(wrapper.gt_HLT[e_i]),
                    },
                    "Bayes": {
                        "lambda_star": float(wrapper.lambdaResultBayes[e_i]),
                        "rho": float(wrapper.rhoResultBayes[e_i]),
                        "stat_err": float(wrapper.drho_bayes[e_i]),
                        "sys_err": float(wrapper.rho_sys_err_Bayes[e_i]),
                        "quadrature_err": float(wrapper.rho_quadrature_err_Bayes[e_i]),
                        "NLL": float(wrapper.minNLL[e_i]),
                        "gt": _gt_to_list(wrapper.gt_Bayes[e_i]),
                    },
                },
            }
        )

    path = path or ioutils.default_output_path(wrapper.par, method)
    return ioutils.write_json(path, metadata, energies)
