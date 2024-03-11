import numpy as np
from .rhoUtils import LogMessage, Inputs
from .rhoStat import parallel_bootstrap_compact_fp
import multiprocessing as multiprocessing
from typing import List
import random


class ParallelBootstrapLoop:
    def __init__(self, par: Inputs, in_: np.ndarray, is_folded=False):
        self.par = par
        self.looplen = par.num_boot
        self.vlen = par.time_extent
        self.inputsample = in_
        self.num_processes = multiprocessing.cpu_count()
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
