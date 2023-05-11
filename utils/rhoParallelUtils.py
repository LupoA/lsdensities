import numpy as np
import matplotlib.pyplot as plt
import random as rd
import os
from rhoUtils import LogMessage
from rhoStat import parallel_bootstrap_compact_fp
import multiprocessing as multiprocessing

class parallel_bootstrap_loop:
    def __init__(self, par_, in_):
        self.par = par_
        self.looplen = par_.num_boot
        self.vlen = par_.time_extent
        self.inputsample = np.zeros((par_.num_boot,par_.time_extent))
        self.inputsample = in_
        self.num_processes = multiprocessing.cpu_count()
        self.chunk_size = par_.num_boot // self.num_processes
        self.out_array = multiprocessing.Array('d', self.looplen * self.vlen)
        self.out_ = np.frombuffer(self.out_array.get_obj()).reshape((self.looplen, self.vlen))
        self.processes = []

    def run(self):
        for i in range(self.num_processes):
            start = i * self.chunk_size
            end = start + self.chunk_size if i != self.num_processes - 1 else self.looplen
            process = multiprocessing.Process(target=parallel_bootstrap_compact_fp,
                                              args=(self.par, self.inputsample, self.out_, start, end))
            process.start()
            self.processes.append(process)
        print(LogMessage(), "Bootstrap ::: Running parallel loop ")
        for process in self.processes:
            process.join()
        print(LogMessage(), "Bootstrap ::: End loop, joining processes ")
        return self.out_


def init_bootstrap_loop(loop_len, vector_len):
    print(LogMessage(), "Create a multiprocessing Array to store the results")
    out_array = multiprocessing.Array('d', loop_len * vector_len)
    out_ = np.frombuffer(out_array.get_obj()).reshape((loop_len, vector_len))
    print(LogMessage(), "Create a list of processes, one for each CPU core")
    processes = []
    num_processes = multiprocessing.cpu_count()
    chunk_size = loop_len // num_processes

def run_bootstrap_loop():
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i != num_processes - 1 else Nb
        process = multiprocessing.Process(target=parallel_bootstrap_compact_fp, args=(par, rawcorr.sample, out_, start, end))
        process.start()
        processes.append(process)
    print(LogMessage(), " Wait for the processes to finish")
    for process in processes:
        process.join()