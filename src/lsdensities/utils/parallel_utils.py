import numpy as np
from .common import LogMessage, Inputs
from .stat_utils import parallel_bootstrap_compact_fp
import multiprocessing
from typing import List
import random


class ParallelBootstrapLoop:
    def __init__(self, par: Inputs, in_: np.ndarray, is_folded=False):
        self.par = par
        self.looplen = par.num_boot
        self.vlen = par.time_extent
        self.inputsample = in_
        self.num_processes = max(1, min(multiprocessing.cpu_count(), self.looplen))
        self.chunk_size = self.looplen // self.num_processes
        if self.looplen % self.num_processes != 0:
            self.chunk_size += 1
        self.out_array = multiprocessing.Array("d", self.looplen * self.vlen)
        self.out_ = np.frombuffer(self.out_array.get_obj()).reshape(
            (self.looplen, self.vlen)
        )
        self.processes: List[multiprocessing.Process] = []
        self.is_folded = is_folded

    def run(self) -> np.ndarray:
        for i in range(self.num_processes):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, self.looplen)
            if self.is_folded is False:
                process = multiprocessing.Process(
                    target=parallel_bootstrap_compact_fp,
                    args=(
                        self.par,
                        self.inputsample,
                        self.out_,
                        start,
                        end,
                        random.randint(0, 2 ** (32) - 1),
                    ),
                )
            if self.is_folded is True:
                process = multiprocessing.Process(
                    target=parallel_bootstrap_compact_fp,
                    args=(
                        self.par,
                        self.inputsample,
                        self.out_,
                        start,
                        end,
                        random.randint(0, 2 ** (32) - 1),
                        self.is_folded,
                    ),
                )
            try:
                process.start()
            except Exception as e:
                print(f"Failed to start process {i}: {e}")
                self.terminate_all_processes()
                raise
            self.processes.append(process)
        print(LogMessage(), "Bootstrap ::: Running parallel loop")
        for process in self.processes:
            try:
                process.join()
            except Exception as e:
                print(f"Failed to join process {process}: {e}")
                self.terminate_all_processes()
                raise
        print(LogMessage(), "Bootstrap ::: End loop, joining processes")
        return self.out_

    def terminate_all_processes(self) -> None:
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()


class JackknifeLoop:
    """
    Builds the delete-1 jackknife replicates (leave-one-out means) of a raw
    sample. Unlike bootstrap, jackknife has no free "how many replicates"
    choice -- there are always exactly par.num_samples replicates, one per
    raw configuration -- and no randomness is involved, so (unlike
    ParallelBootstrapLoop) this does not need multiprocessing: each replicate
    is O(1) given the precomputed total sum, for O(n) total instead of the
    naive O(n^2).

    Mirrors ParallelBootstrapLoop's constructor/`.run()` interface so it can
    be used as a drop-in alternative at the same call sites; use together
    with Obs(..., sample_type="jackknife").
    """

    def __init__(self, par: Inputs, in_: np.ndarray, is_folded=False):
        self.par = par
        self.inputsample = in_
        self.is_folded = is_folded

    def run(self) -> np.ndarray:
        n = self.par.num_samples
        vlen = (
            int(self.par.time_extent / 2) + 1 if self.is_folded else self.par.time_extent
        )
        out_ = np.zeros((n, self.par.time_extent))
        total = np.sum(self.inputsample[:, :vlen], axis=0)
        out_[:, :vlen] = (total[None, :] - self.inputsample[:, :vlen]) / (n - 1)
        return out_
