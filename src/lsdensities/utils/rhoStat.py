import numpy as np
import random
from mpmath import mp
import lsdensities.utils.rhoUtils as u
from .rhoUtils import LogMessage, Inputs
import multiprocessing as multiprocessing
from typing import List


def bootstrap_obstype_fp(par_, in_obs_type):
    """
    bootstrap the sample attribute of in_obs_type
    """
    from .rhoUtils import ranvec

    out_ = np.zeros((par_.num_boot, par_.time_extent))
    randv = np.zeros(par_.num_samples)
    for b in range(0, par_.num_boot):
        randv = ranvec(randv, par_.num_samples, 0, par_.num_samples)
        for i in range(0, par_.time_extent):
            out_[b][i] = 0
            for j in range(0, par_.num_samples):
                out_[b][i] += in_obs_type.sample[int(randv[j])][i]
            out_[b][i] /= par_.num_samples
    return out_


def bootstrap_sample_fp(par_, in_sample):
    """
    bootstrap in_sample
    """
    from .rhoUtils import ranvec

    out_ = np.zeros((par_.num_boot, par_.time_extent))
    randv = np.zeros(par_.num_samples)
    for b in range(0, par_.num_boot):
        randv = ranvec(randv, par_.num_samples, 0, par_.num_samples)
        for i in range(0, par_.time_extent):
            out_[b][i] = 0
            for j in range(0, par_.num_samples):
                out_[b][i] += in_sample[int(randv[j])][i]
            out_[b][i] /= par_.num_samples
    return out_


def parallel_bootstrap_sample_fp(par_, in_, out_, start, end, seed, is_folded=False):
    """
    parallelisable version of bootstrap_sample_fp
    """
    import lsdensities.utils.rhoUtils

    random.seed(seed)
    randv = np.zeros(par_.num_samples)
    if is_folded is False:
        for b in range(start, end):
            randv = lsdensities.utils.rhoUtils.ranvec(
                randv, par_.num_samples, 0, par_.num_samples
            ).astype(int)
            for i in range(par_.time_extent):
                out_[b][i] = np.mean(in_[randv[:], i])
    if is_folded is True:
        for b in range(start, end):
            randv = lsdensities.utils.rhoUtils.ranvec(
                randv, par_.num_samples, 0, par_.num_samples
            ).astype(int)
            for i in range(int(par_.time_extent / 2) + 1):
                out_[b][i] = np.mean(in_[randv[:], i])


def average_2d_mpmatrix(in_, bootstrap=True):
    """
    average 2D mpmatrix on the second axis, for each value of the first axis
    returns a 2D mpmatrix to be interpreted as two 1D mpmatrix:
    return: mp_array_mean, mp_array_std
    """
    xlen_ = in_.rows
    samplesize_ = in_.cols
    out_ = mp.matrix(xlen_, 2)
    for x in range(xlen_):
        out_[x, 0] = 0
        out_[x, 1] = 0
        for b in range(samplesize_):
            out_[x, 0] = mp.fadd(out_[x, 0], in_[x, b])
        out_[x, 0] = mp.fdiv(out_[x, 0], samplesize_)
        for b in range(samplesize_):
            aux_ = mp.fsub(in_[x, b], out_[x, 0])
            aux_ = mp.fmul(aux_, aux_)
            out_[x, 1] = mp.fadd(out_[x, 1], aux_)
        out_[x, 1] = mp.fdiv(out_[x, 1], samplesize_)
        out_[x, 1] = mp.sqrt(out_[x, 1])
        if bootstrap is False:
            aux_ = mp.sqrt(samplesize_)
            out_[x, 1] = mp.fdiv(out_[x, 1], aux_)
    return out_


def average_1d_mpmatrix(in_, bootstrap=True):
    """
    average 1D mpmatrix
    """
    if in_.rows == 1:
        samplesize_ = in_.cols
    if in_.cols == 1:
        samplesize_ = in_.rows
    out_ = mp.matrix(2, 1)
    out_[0] = 0
    out_[1] = 0
    for b in range(samplesize_):
        out_[0] = mp.fadd(out_[0], in_[b])
    out_[0] = mp.fdiv(out_[0], samplesize_)
    for b in range(samplesize_):
        aux_ = mp.fsub(in_[b], out_[0])
        aux_ = mp.fmul(aux_, aux_)
        out_[1] = mp.fadd(out_[1], aux_)
    out_[1] = mp.fdiv(out_[1], samplesize_)
    out_[1] = mp.sqrt(out_[1])
    if bootstrap is False:
        aux_ = mp.sqrt(samplesize_)
        out_[1] = mp.fdiv(out_[1], aux_)
    return out_


def resample(input_corr, par, parallelise=False):
    if parallelise is True:
        if par.periodicity == "EXP":
            output_corr = u.Obs(
                T=par.time_extent, tmax=par.tmax, nms=par.num_boot, is_resampled=True
            )
            resample = ParallelBootstrapLoop(par, input_corr.sample, is_folded=False)
        if par.periodicity == "COSH":
            output_corr = u.Obs(
                T=input_corr.T,
                tmax=input_corr.tmax,
                nms=par.num_boot,
                is_resampled=True,
            )
            resample = ParallelBootstrapLoop(par, input_corr.sample, is_folded=True)
        output_corr.sample = resample.run()

    else:
        output_corr = u.Obs(
            T=input_corr.T,
            tmax=input_corr.tmax,
            nms=par.num_boot,
            is_resampled=True,
        )
        output_corr.sample = bootstrap_obstype_fp(par, input_corr)

    output_corr.evaluate()
    return output_corr


class ParallelBootstrapLoop:
    def __init__(self, par: Inputs, input_sample: np.ndarray, is_folded=False):
        self.par = par
        self.looplen = par.num_boot
        self.vlen = par.time_extent
        self.inputsample = input_sample
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
                    target=parallel_bootstrap_sample_fp,
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
                    target=parallel_bootstrap_sample_fp,
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
