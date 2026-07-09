"""
Tests for common.read_datafile's sample_type handling.

read_datafile used to take a `resampled: bool` flag, which could only ever
distinguish "montecarlo" vs "bootstrap" -- there was no way to say a file
already holds jackknife replicates. It now takes `sample_type` directly, like
every other entry point that builds an Obs.
"""

import pytest

from lsdensities.utils.common import read_datafile


def _write_datafile(path, nms, T):
    with open(path, "w") as f:
        f.write(f"{nms} {T}\n")
        for n in range(nms):
            for t in range(T):
                f.write(f"{t} {n * T + t}\n")


@pytest.mark.parametrize("sample_type", ["montecarlo", "bootstrap", "jackknife"])
def test_read_datafile_honours_sample_type(tmp_path, sample_type):
    datapath = tmp_path / "corr.txt"
    _write_datafile(datapath, nms=5, T=4)

    corr, header_T, header_nms = read_datafile(str(datapath), sample_type=sample_type)

    assert corr.sample_type == sample_type
    assert header_T == 4
    assert header_nms == 5


def test_read_datafile_defaults_to_montecarlo(tmp_path):
    datapath = tmp_path / "corr.txt"
    _write_datafile(datapath, nms=3, T=2)

    corr, _, _ = read_datafile(str(datapath))

    assert corr.sample_type == "montecarlo"


def test_read_datafile_rejects_invalid_sample_type(tmp_path):
    datapath = tmp_path / "corr.txt"
    _write_datafile(datapath, nms=3, T=2)

    with pytest.raises(ValueError):
        read_datafile(str(datapath), sample_type="bogus")
