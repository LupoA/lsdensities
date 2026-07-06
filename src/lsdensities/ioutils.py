"""
Shared JSON output format for lsdensities results.

Every wrapper (InverseProblemWrapper, GaussianProcessWrapper,
HilbertEigTruncWrapper) writes one JSON file per run with the same top-level
shape:

    {"metadata": {...}, "energies": [{"energy": ..., "scan": {...}, "result": {...}}, ...]}

so that a single script (examples/plot_output.py) can read any of them back
and reproduce the relevant plots -- arXiv:2605.14652 Fig. 6 (stability
analysis: rho, its statistical/Bayesian error and the NLL, scanned over
lambda for each alpha) or Fig. 7 (eigen-space analysis: the cumulative
contribution to rho as eigenmodes are added, for each alpha) -- without
needing the live Python objects that produced them.

"scan" holds one entry per alpha channel ("A", "B", "C"); its exact contents
differ between the stability analysis (arrays over lambda) and the
eigen-space analysis (arrays over the eigenmode index k), but the file shape
and the metadata block are shared.
"""

import json
import os


def default_output_path(par, method):
    name = (
        f"{method}_tmax{par.tmax}_sigma{par.sigma}_Ne{par.Ne}"
        f"_Na{par.Na}_kernel{par.kerneltype}.json"
    )
    return os.path.join(par.logpath, name)


def base_metadata(par, method):
    """Fields common to every method's output; callers add an "alphas" dict and an "algorithm" block."""
    return {
        "method": method,
        "time_extent": par.time_extent,
        "tmax": par.tmax,
        "periodicity": par.periodicity,
        "kerneltype": par.kerneltype,
        "sigma": par.sigma,
        "emin": par.emin,
        "emax": par.emax,
        "Ne": par.Ne,
        "Na": par.Na,
        "e0": par.e0,
        "prec": par.prec,
        "A0cut": par.A0cut,
    }


def write_json(path, metadata, energies):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"metadata": metadata, "energies": energies}, f, indent=2)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def closest_energy_entry(data, energy):
    """Returns the entry in data["energies"] whose "energy" is closest to the requested value."""
    return min(data["energies"], key=lambda entry: abs(entry["energy"] - energy))
