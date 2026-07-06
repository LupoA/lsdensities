"""
Standalone plotting for lsdensities output files.

InverseProblemWrapper.save() / GaussianProcessWrapper.save() / HilbertEigTruncWrapper.save()
all write one JSON file per run (see src/lsdensities/ioutils.py). This script reads such a file back and reproduces the relevant plot.
Options are:

  stability          arXiv:2605.14652 Fig. 6, rho panel only: rho scanned over
                     lambda, for every alpha channel, at one energy.
  nll                NLL scanned over lambda, for every alpha channel, at one
                     energy, plus the flagged minimum.
  stability-and-nll  arXiv:2605.14652 Fig. 6 in full: both panels above,
                     stacked, at one energy.
  eigenspace         arXiv:2605.14652 Fig. 7: the individual and cumulative
                     contribution to rho as eigenmodes of A_N are added, for
                     every alpha channel, at one energy.
  spectrum           the full rho_sigma(E) vs E, from one or more files.
  kernel             the reconstructed smearing kernel Delta(E, E'), built
                     from the flagged result's (HLT/GP) or chosen channel's
                     (EigenspaceAnalysis) saved g_t coefficients, vs the
                     exact target kernel, at one energy.

Examples:
    python3 plot_output.py stability          --file test_hlt_wnoise/Logs/HLT_....json --energy 0.9
    python3 plot_output.py nll                --file test_hlt_wnoise/Logs/HLT_....json --energy 0.9
    python3 plot_output.py stability-and-nll   --file test_hlt_wnoise/Logs/HLT_....json --energy 0.9
    python3 plot_output.py eigenspace         --file HET_....json --energy 0.9
    python3 plot_output.py eigenspace         --file HET_....json --energy 0.9 --kmin 0 --kmax 13
    python3 plot_output.py spectrum           --file test_hlt_wnoise/Logs/HLT_....json test_gp_wnoise/Logs/GP_....json
    python3 plot_output.py kernel             --file test_hlt_wnoise/Logs/HLT_....json --energy 0.9
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from lsdensities.ioutils import closest_energy_entry, load_json
from lsdensities.plotutils import plotNoErr, plotwErr, setPlotOpt
from lsdensities.utils.rhoMath import cauchy, gauss_fp
from lsdensities.utils.rhoUtils import CB_colors

PRD_SINGLE_WIDTH = 3.375
PRD_DOUBLE_WIDTH = 6.9
ASPECT_RATIO = 1.6

def figsize_single(scale=1.0):
    w = PRD_SINGLE_WIDTH * scale
    h = w / ASPECT_RATIO
    return (w, h)

def figsize_double(scale=1.0):
    w = PRD_DOUBLE_WIDTH * scale
    h = w / ASPECT_RATIO
    return (w, h)

def apply_style():
    mpl.rcParams.update({
        "figure.figsize": figsize_double(),
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 18,

        "lines.linewidth": 1.,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,

        "xtick.major.size": 4,
        "ytick.major.size": 4,

        "font.family": "serif",
        "mathtext.fontset": "cm",
    })

def _energy_entry(data, energy):
    entry = closest_energy_entry(data, energy)
    if abs(entry["energy"] - energy) > 1e-9:
        print(
            f"Note: no entry at exactly E={energy}; using the closest one, E={entry['energy']:.4g}"
        )
    return entry


def _save_or_show(fig, outdir, name, show):
    if show:
        plt.show()
    else:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=300)
        print("Wrote", path)
    plt.close(fig)


def _save_or_show_multi(figs_and_names, outdir, show):
    """Like _save_or_show, but for several independent figures at once (shown together if --show)."""
    if show:
        plt.show()
    else:
        os.makedirs(outdir, exist_ok=True)
        for fig, name in figs_and_names:
            path = os.path.join(outdir, name)
            fig.savefig(path, dpi=300)
            print("Wrote", path)
    for fig, _name in figs_and_names:
        plt.close(fig)


def _stability_channels(entry, alphas):
    """[(color_id, label, scan-for-that-channel, alpha-legend-label), ...], sorted by label."""
    channels = []
    for color_id, label in enumerate(sorted(entry["scan"].keys())):
        ch = entry["scan"][label]
        alpha_label = r"$\alpha = {:1.2f}$".format(alphas[label]) if label in alphas else label
        channels.append((color_id, label, ch, alpha_label))
    return channels


def _stability_title(entry, data):
    return (
        r"$E$" + "= {:2.2f}  ".format(entry["energy"])
        + r"$\sigma$" + " = {:2.2f} ".format(data["metadata"]["sigma"])
        + f"[{data['metadata']['method']}]"
    )


def plot_stability(data, energy, outdir, show):
    """arXiv:2605.14652 Fig. 6, rho panel only: rho scanned over lambda, for every alpha channel."""
    entry = _energy_entry(data, energy)
    alphas = data["metadata"].get("alphas", {})
    channels = _stability_channels(entry, alphas)

    setPlotOpt(plt)
    fig, ax = plt.subplots(figsize=(8, 6))

    for color_id, _label, ch, alpha_label in channels:
        plotwErr(
            ax, ch["lambda"], ch["rho"], ch["errBoot"],
            label=alpha_label, markerId=color_id, colorID=color_id,
        )

    hlt = entry["result"]["HLT"]
    ax.axhspan(
        ymin=hlt["rho"] - hlt["stat_err"], ymax=hlt["rho"] + hlt["stat_err"],
        alpha=0.3, color=CB_colors[4],
    )

    ax.set_xlabel(r"$\lambda$", fontsize=32)
    ax.set_ylabel(r"$\rho_\sigma$", fontsize=32)
    ax.legend()
    ax.set_xscale("log")
    ax.set_title(_stability_title(entry, data))
    plt.tight_layout()
    _save_or_show(fig, outdir, f"stability_E{entry['energy']:.3f}.png", show)


def plot_nll(data, energy, outdir, show):
    """NLL scanned over lambda, for every alpha channel, plus the flagged (Bayesian) minimum."""
    entry = _energy_entry(data, energy)
    alphas = data["metadata"].get("alphas", {})
    channels = _stability_channels(entry, alphas)

    setPlotOpt(plt)
    fig, ax = plt.subplots(figsize=(8, 6))

    for color_id, _label, ch, alpha_label in channels:
        plotNoErr(
            ax, ch["lambda"], ch["likelihood"],
            label="NLL " + alpha_label, markerId=color_id, colorID=color_id,
        )

    bayes = entry["result"]["Bayes"]
    ax.plot(
        bayes["lambda_star"], bayes["NLL"],
        marker="*", markersize=25, markerfacecolor="red", markeredgecolor="black",
        color="red", ls="--", label="Min NLL",
    )

    ax.set_xlabel(r"$\lambda$", fontsize=32)
    ax.set_ylabel("NLL", fontsize=32)
    ax.legend()
    ax.set_xscale("log")
    ax.set_title(_stability_title(entry, data))
    plt.tight_layout()
    _save_or_show(fig, outdir, f"nll_E{entry['energy']:.3f}.png", show)


def plot_stability_and_nll(data, energy, outdir, show):
    """arXiv:2605.14652 Fig. 6 in full: rho and the NLL scanned over lambda, for every alpha channel."""
    entry = _energy_entry(data, energy)
    alphas = data["metadata"].get("alphas", {})
    channels = _stability_channels(entry, alphas)

    setPlotOpt(plt)
    fig, (ax, ax2) = plt.subplots(
        nrows=2, sharex=True, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 2]}
    )

    for color_id, _label, ch, alpha_label in channels:
        plotwErr(
            ax, ch["lambda"], ch["rho"], ch["errBoot"],
            label=alpha_label, markerId=color_id, colorID=color_id,
        )
        plotNoErr(
            ax2, ch["lambda"], ch["likelihood"],
            label="NLL " + alpha_label, markerId=color_id, colorID=color_id,
        )

    hlt = entry["result"]["HLT"]
    bayes = entry["result"]["Bayes"]
    ax.axhspan(
        ymin=hlt["rho"] - hlt["stat_err"], ymax=hlt["rho"] + hlt["stat_err"],
        alpha=0.3, color=CB_colors[4],
    )
    ax2.plot(
        bayes["lambda_star"], bayes["NLL"],
        marker="*", markersize=25, markerfacecolor="red", markeredgecolor="black",
        color="red", ls="--", label="Min NLL",
    )

    ax.set_ylabel(r"$\rho_\sigma$", fontsize=32)
    ax.legend()
    ax.set_xscale("log")
    ax2.set_xlabel(r"$\lambda$", fontsize=32)
    ax2.set_ylabel("NLL", fontsize=32)
    ax.set_title(_stability_title(entry, data))
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    _save_or_show(fig, outdir, f"stability_and_nll_E{entry['energy']:.3f}.png", show)


def _windowed_by_k(ch, kmin, kmax):
    """ch restricted to entries whose k is within [kmin, kmax] (either bound optional)."""
    k = ch["k"]
    lo = kmin if kmin is not None else k[0]
    hi = kmax if kmax is not None else k[-1]
    idx = [i for i, kv in enumerate(k) if lo <= kv <= hi]
    return {key: [v[i] for i in idx] for key, v in ch.items()}


def plot_eigenspace(data, energy, kmin, kmax, outdir, show):
    """arXiv:2605.14652 Fig. 7: individual and cumulative eigenmode contributions to rho, as two separate figures."""
    entry = _energy_entry(data, energy)
    labels = sorted(entry["scan"].keys())
    alphas = data["metadata"].get("alphas", {})
    title = (
        r"$E$" + "= {:2.2f}  ".format(entry["energy"])
        + r"$\sigma$" + " = {:2.2f} ".format(data["metadata"]["sigma"])
    )

    setPlotOpt(plt)
    fig_cum, ax_cum = plt.subplots(figsize=(8, 6))
    fig_contrib, ax_contrib = plt.subplots(figsize=(8, 6))

    for color_id, label in enumerate(labels):
        ch = _windowed_by_k(entry["scan"][label], kmin, kmax)
        result = entry["result"][label]
        alpha_label = r"$\alpha = {:1.2f}$".format(alphas[label]) if label in alphas else label
        plotwErr(
            ax_cum, ch["k"], ch["cumulative_mean"], ch["cumulative_err"],
            label=alpha_label, markerId=color_id, colorID=color_id,
        )
        plotwErr(
            ax_contrib, ch["k"], ch["contrib_mean"], ch["contrib_err"],
            label=alpha_label, markerId=color_id, colorID=color_id,
        )
        if (kmin is None or kmin <= result["kstar"]) and (kmax is None or result["kstar"] <= kmax):
            ax_cum.axvline(result["kstar"], color=CB_colors[color_id], ls="--", alpha=0.5)
            ax_contrib.axvline(result["kstar"], color=CB_colors[color_id], ls="--", alpha=0.5)

    ax_cum.set_xlabel(r"$k$", fontsize=32)
    ax_cum.set_ylabel(r"$\sum_{k'\leq k} \hat g(k')\hat C(k')$", fontsize=22)
    ax_cum.legend()
    ax_cum.set_title(title)
    fig_cum.tight_layout()

    ax_contrib.set_xlabel(r"$k$", fontsize=32)
    ax_contrib.set_ylabel(r"$\hat g(k)\hat C(k)$", fontsize=22)
    ax_contrib.legend()
    ax_contrib.set_title(title)
    fig_contrib.tight_layout()

    _save_or_show_multi(
        [
            (fig_cum, f"eigenspace_cumulative_E{entry['energy']:.3f}.png"),
            (fig_contrib, f"eigenspace_contrib_E{entry['energy']:.3f}.png"),
        ],
        outdir, show,
    )


def _reconstructed_kernel(gt, time_extent, periodicity, e_prime):
    """Delta(E, E') = sum_t g_t(E) * exp(-t E') (or its COSH periodic image sum). gt[i] pairs with t = i+1."""
    kernel = np.zeros_like(e_prime, dtype=float)
    for i, g in enumerate(gt):
        t = i + 1
        if periodicity == "COSH":
            basis = np.exp(-(time_extent - t) * e_prime) + np.exp(-t * e_prime)
        else:
            basis = np.exp(-t * e_prime)
        kernel += g * basis
    return kernel


def _exact_kernel(kerneltype, e_prime, e_star, sigma):
    if kerneltype == "FULLNORMGAUSS":
        return gauss_fp(e_prime, e_star, sigma, norm="full")
    if kerneltype == "HALFNORMGAUSS":
        return gauss_fp(e_prime, e_star, sigma, norm="half")
    if kerneltype == "CAUCHY":
        return cauchy(e_prime, sigma, e_star)
    raise ValueError(f"No exact-kernel formula available for kerneltype '{kerneltype}'")


def plot_kernel(data, energy, result, channel, outdir, show):
    """
    Reconstructed smearing kernel Delta(E, E') vs the exact target kernel, at
    one energy. For HLT/GP output, built from the flagged (--result HLT/Bayes)
    result's saved g_t. For EigenspaceAnalysis output, built from the chosen
    (--channel A/B/C) channel's kstar-truncated effective g_t.
    """
    entry = _energy_entry(data, energy)
    meta = data["metadata"]
    if meta["method"] == "EigenspaceAnalysis":
        if channel not in entry["result"]:
            raise ValueError(
                f"No channel '{channel}' in this file's results (available: {sorted(entry['result'].keys())})"
            )
        gt = entry["result"][channel]["gt"]
        tag = f"channel {channel}"
        file_tag = f"EA-{channel}"
    else:
        gt = entry["result"][result]["gt"]
        tag = result
        file_tag = result
    if gt is None:
        raise ValueError(f"No 'gt' coefficients stored for {tag} at this energy")

    e_star = entry["energy"]
    sigma = meta["sigma"]
    e_prime = np.linspace(max(1e-6, e_star - 6 * sigma), e_star + 6 * sigma, 400)

    reconstructed = _reconstructed_kernel(gt, meta["time_extent"], meta["periodicity"], e_prime)
    exact = _exact_kernel(meta["kerneltype"], e_prime, e_star, sigma)

    setPlotOpt(plt)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(e_prime, exact, color="black", ls="--", label="Exact kernel")
    ax.plot(e_prime, reconstructed, color=CB_colors[0], label=f"Reconstructed ({tag})")
    ax.axvline(e_star, color="gray", ls=":", alpha=0.6)

    ax.set_xlabel(r"$E'$ [GeV]")
    ax.set_ylabel(r"$\Delta_\sigma(E, E')$")
    ax.legend()
    ax.set_title(
        r"$E$" + "= {:2.2f}  ".format(e_star)
        + r"$\sigma$" + " = {:2.2f} ".format(sigma)
        + f"[{meta['method']}, {tag}]"
    )
    plt.tight_layout()
    _save_or_show(fig, outdir, f"kernel_E{e_star:.3f}_{file_tag}.png", show)


def plot_spectrum(datasets, labels, outdir, show):
    """The full rho_sigma(E) vs E, from one or more stability-analysis output files."""
    setPlotOpt(plt)
    fig, ax = plt.subplots(figsize=(8, 6))

    next_color = 0
    for data, run_label in zip(datasets, labels):
        energies = [e["energy"] for e in data["energies"]]
        if data["metadata"]["method"] == "EigenspaceAnalysis":
            res = [e["result"]["A"]["res"] for e in data["energies"]]
            err = [e["result"]["A"]["err"] for e in data["energies"]]
            plotwErr(ax, energies, res, err, label=run_label, markerId=next_color, colorID=next_color)
            next_color += 1
            continue
        hlt_rho = [e["result"]["HLT"]["rho"] for e in data["energies"]]
        hlt_err = [e["result"]["HLT"]["quadrature_err"] for e in data["energies"]]
        bayes_rho = [e["result"]["Bayes"]["rho"] for e in data["energies"]]
        bayes_err = [e["result"]["Bayes"]["quadrature_err"] for e in data["energies"]]
        plotwErr(
            ax, energies, hlt_rho, hlt_err,
            label=f"{run_label} (plateau)", markerId=next_color, colorID=next_color,
        )
        plotwErr(
            ax, energies, bayes_rho, bayes_err,
            label=f"{run_label} (Bayesian NLL)", markerId=next_color + 1, colorID=next_color + 1,
        )
        next_color += 2

    ax.set_xlabel(r"$E$ [GeV]")
    ax.set_ylabel(r"$\rho_\sigma(E)$")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, outdir, "spectrum.png", show)


def main():
    apply_style()
    # Shared so --outdir/--show are accepted both before and after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--outdir", type=str, default=".", help="Where to save plots (ignored if --show). Default: current directory")
    common.add_argument("--show", action="store_true", help="Display interactively instead of saving to file")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[common]
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_stab = sub.add_parser("stability", parents=[common], help="Fig. 6-style rho-vs-lambda plot at one energy (no NLL panel)")
    p_stab.add_argument("--file", required=True, help="Output JSON from InverseProblemWrapper.save() or GaussianProcessWrapper.save()")
    p_stab.add_argument("--energy", type=float, required=True, help="Energy to plot (GeV); closest available is used")

    p_nll = sub.add_parser("nll", parents=[common], help="NLL-vs-lambda plot at one energy")
    p_nll.add_argument("--file", required=True, help="Output JSON from InverseProblemWrapper.save() or GaussianProcessWrapper.save()")
    p_nll.add_argument("--energy", type=float, required=True, help="Energy to plot (GeV); closest available is used")

    p_stab_nll = sub.add_parser("stability-and-nll", parents=[common], help="Fig. 6-style plot with both the rho and NLL panels, at one energy")
    p_stab_nll.add_argument("--file", required=True, help="Output JSON from InverseProblemWrapper.save() or GaussianProcessWrapper.save()")
    p_stab_nll.add_argument("--energy", type=float, required=True, help="Energy to plot (GeV); closest available is used")

    p_eig = sub.add_parser("eigenspace", parents=[common], help="Fig. 7-style eigen-space analysis plot at one energy (two figures: cumulative and per-mode)")
    p_eig.add_argument("--file", required=True, help="Output JSON from HilbertEigTruncWrapper.save()")
    p_eig.add_argument("--energy", type=float, required=True, help="Energy to plot (GeV); closest available is used")
    p_eig.add_argument("--kmin", type=int, default=None, help="Restrict the plotted eigenmode index to k >= kmin. Default: no lower bound")
    p_eig.add_argument("--kmax", type=int, default=None, help="Restrict the plotted eigenmode index to k <= kmax. Default: no upper bound")

    p_spec = sub.add_parser("spectrum", parents=[common], help="Full rho_sigma(E) vs E, optionally overlaying several runs")
    p_spec.add_argument("--file", required=True, nargs="+", help="One or more output JSON files")
    p_spec.add_argument("--labels", nargs="+", help="Legend label per --file (defaults to each file's 'method')")

    p_kernel = sub.add_parser("kernel", parents=[common], help="Reconstructed vs exact smearing kernel at one energy")
    p_kernel.add_argument("--file", required=True, help="Output JSON from InverseProblemWrapper.save(), GaussianProcessWrapper.save() or HilbertEigTruncWrapper.save()")
    p_kernel.add_argument("--energy", type=float, required=True, help="Energy to plot (GeV); closest available is used")
    p_kernel.add_argument("--result", choices=["HLT", "Bayes"], default="HLT", help="For HLT/GP output: which flagged result's g_t to use. Ignored for EigenspaceAnalysis output. Default=HLT")
    p_kernel.add_argument("--channel", default="A", help="For EigenspaceAnalysis output: which alpha channel's g_t to use (A, B or C). Ignored for HLT/GP output. Default=A")

    args = parser.parse_args()

    if args.mode == "stability":
        plot_stability(load_json(args.file), args.energy, args.outdir, args.show)
    elif args.mode == "nll":
        plot_nll(load_json(args.file), args.energy, args.outdir, args.show)
    elif args.mode == "stability-and-nll":
        plot_stability_and_nll(load_json(args.file), args.energy, args.outdir, args.show)
    elif args.mode == "eigenspace":
        plot_eigenspace(load_json(args.file), args.energy, args.kmin, args.kmax, args.outdir, args.show)
    elif args.mode == "kernel":
        plot_kernel(load_json(args.file), args.energy, args.result, args.channel, args.outdir, args.show)
    elif args.mode == "spectrum":
        datasets = [load_json(f) for f in args.file]
        labels = args.labels or [d["metadata"]["method"] for d in datasets]
        if len(labels) != len(datasets):
            parser.error("--labels must have the same length as --file")
        plot_spectrum(datasets, labels, args.outdir, args.show)


if __name__ == "__main__":
    main()
