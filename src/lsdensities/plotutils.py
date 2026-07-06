import numpy as np
from .utils.rhoUtils import CB_colors, plot_markers


def setPlotOpt(plt):
    plt.rcParams["figure.figsize"] = 5, 2
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rc("xtick", labelsize=22)
    plt.rc("ytick", labelsize=22)
    plt.rcParams.update({"font.size": 22})


def plotwErr(ax, x, y, yerr, label="", markerId=0, colorID=0):
    ax.errorbar(
        x=np.array(x, dtype=float),
        y=np.array(y, dtype=float),
        yerr=np.array(yerr, dtype=float),
        marker=plot_markers[markerId],
        markersize=4.8,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=label,
        color="black",
        ecolor=CB_colors[colorID],
        markerfacecolor=CB_colors[colorID],
    )


def plotNoErr(ax, x, y, label="", markerId=0, colorID=0):
    ax.errorbar(
        x=np.array(x, dtype=float),
        y=np.array(y, dtype=float),
        marker=plot_markers[markerId],
        markersize=4.8,
        elinewidth=1.3,
        capsize=2,
        ls="",
        label=label,
        color="black",
        ecolor=CB_colors[colorID],
        markerfacecolor=CB_colors[colorID],
    )
