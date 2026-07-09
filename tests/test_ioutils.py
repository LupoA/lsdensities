"""
Tests for src/lsdensities/io_utils.py's provenance metadata.
"""

import re
from datetime import datetime

from lsdensities.io_utils import _provenance_metadata, base_metadata
from lsdensities.utils.common import Inputs


def test_provenance_metadata_has_expected_keys():
    info = _provenance_metadata()
    assert set(info.keys()) == {
        "script",
        "timestamp_utc",
        "user",
        "hostname",
        "package_version",
        "git_commit",
        "git_commit_timestamp",
    }


def test_provenance_timestamp_is_utc_and_parseable():
    info = _provenance_metadata()
    parsed = datetime.fromisoformat(info["timestamp_utc"])
    assert parsed.utcoffset().total_seconds() == 0


def test_provenance_user_and_hostname_are_nonempty_strings():
    info = _provenance_metadata()
    assert isinstance(info["user"], str) and info["user"]
    assert isinstance(info["hostname"], str) and info["hostname"]


def test_provenance_git_commit_is_a_valid_hash_when_present():
    info = _provenance_metadata()
    # Running from a git checkout in this repo, so this should be populated.
    assert re.fullmatch(r"[0-9a-f]{40}", info["git_commit"])
    datetime.fromisoformat(info["git_commit_timestamp"])


def test_base_metadata_includes_provenance():
    par = Inputs()
    par.time_extent = 16
    par.tmax = 15
    par.periodicity = "EXP"
    par.kerneltype = "FULLNORMGAUSS"
    par.sigma = 0.4
    par.emin = 0.3
    par.emax = 1.2
    par.Ne = 4
    par.Na = 1
    par.e0 = 0
    par.prec = 50
    par.A0cut = 0.2

    meta = base_metadata(par, "HLT")
    assert "provenance" in meta
    assert meta["provenance"]["script"]
