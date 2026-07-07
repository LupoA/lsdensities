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

Every metadata block also carries a "provenance" sub-dict (script name,
package version/git commit, UTC timestamp, user, hostname) recording what
produced the file -- see _provenance_metadata.
"""

import getpass
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone


def default_output_path(par, method):
    name = (
        f"{method}_tmax{par.tmax}_sigma{par.sigma}_Ne{par.Ne}"
        f"_Na{par.Na}_kernel{par.kerneltype}.json"
    )
    return os.path.join(par.logpath, name)


def _package_version_info():
    """
    The installed package version, plus the git commit (and its own
    commit timestamp) code is running from, if any
    """
    try:
        package_version = importlib.metadata.version("lsdensities")
    except importlib.metadata.PackageNotFoundError:
        package_version = None

    git_commit = None
    git_commit_timestamp = None
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_commit_timestamp = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return {
        "package_version": package_version,
        "git_commit": git_commit,
        "git_commit_timestamp": git_commit_timestamp,
    }


def _provenance_metadata():
    """Who/what/when/where produced this output file, for reproducibility."""
    try:
        user = getpass.getuser()
    except OSError:
        user = None
    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = None

    info = {
        "script": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "user": user,
        "hostname": hostname,
    }
    info.update(_package_version_info())
    return info


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
        "provenance": _provenance_metadata(),
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
