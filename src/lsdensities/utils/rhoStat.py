import numpy as np
import matplotlib.pyplot as plt
import math
import random
from mpmath import mp
import lsdensities.utils.rhoUtils as u
from .rhoUtils import LogMessage, Inputs
import multiprocessing as multiprocessing
from typing import List


def averageVector_fp(vector, get_error=True, get_var=False):
    sum = 0
    stdv = 0
    for i in range(len(vector)):
        sum += vector[i]
    sum /= len(vector)
    for i in range(len(vector)):
        stdv += (vector[i] - sum) ** 2
    stdv /= len(vector) - 1
    stdv = math.sqrt(stdv)
    var = stdv
    err = stdv / math.sqrt(len(vector))
    if get_error is True:
        if get_var is True:
            return sum, var
        if get_var is False:
            return sum, err
    if get_error is False:
        return sum


def bootstrap_compact_fp(par_, in_):
    from .rhoUtils import ranvec

    out_ = np.zeros((par_.num_boot, par_.time_extent))
    randv = np.zeros(par_.num_samples)
    for b in range(0, par_.num_boot):
        randv = ranvec(randv, par_.num_samples, 0, par_.num_samples)
        for i in range(0, par_.time_extent):
            out_[b][i] = 0
            for j in range(0, par_.num_samples):
                out_[b][i] += in_[int(randv[j])][i]
            out_[b][i] /= par_.num_samples
    return out_


def parallel_bootstrap_compact_fp_DEPRECATED(
    par_, in_, out_, start, end
):  #   old slower version but more explicit
    print("Function is deprecated.")
    exit(1)
    from .rhoUtils import ranvec

    randv = np.zeros(par_.num_samples)
    for b in range(start, end):
        randv = ranvec(randv, par_.num_samples, 0, par_.num_samples)
        for i in range(0, par_.time_extent):
            out_[b][i] = 0
            for j in range(0, par_.num_samples):
                out_[b][i] += in_[int(randv[j])][i]
            out_[b][i] /= par_.num_samples


def parallel_bootstrap_compact_fp(par_, in_, out_, start, end, seed, is_folded=False):
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


def bootstrap_fp(T_, nms_, Nb_, in_, out_):
    from .rhoUtils import ranvec

    randv = np.zeros(nms_)
    for b in range(0, Nb_):
        randv = ranvec(randv, nms_, 0, nms_)
        for i in range(0, T_):
            for j in range(0, nms_):
                out_[b][i] += in_[int(randv[j])][i]
            out_[b][i] /= nms_
    return out_


def getCovMatrix_fp(sample, central, nbins, vmax, showplot=False):
    cov_ = np.zeros((vmax, vmax))
    for vi in range(vmax):
        for vj in range(vmax):
            cov_[vi][vj] = 0
            for n in range(nbins):
                cov_[vi][vj] += (sample[n][vi] - central[vi]) * (
                    sample[n][vj] - central[vj]
                )
            cov_[vi][vj] /= nbins - 1
    if showplot is True:
        plt.imshow(cov_, cmap="viridis")
        plt.colorbar()
        plt.show()
    return cov_


def covToCorr_fp(in_, cov_, vmax, showplot=False):
    corrmat_ = np.zeros((vmax, vmax))
    for vi in range(vmax):
        for vj in range(vmax):
            corrmat_[vi][vj] = cov_[vi][vj] / (in_.sigma[vi] * in_.sigma[vj])
    if showplot is True:
        plt.imshow(corrmat_)
        plt.colorbar()
        plt.show()
    return corrmat_


def averageVector_mp(in_, bootstrap=True):
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


def averageScalar_mp(in_, bootstrap=True):
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


def resample(input_corr, par):
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
        resample = ParallelBootstrapLoop(par, input_corr.sample, is_folded=False)

    output_corr.sample = resample.run()
    output_corr.evaluate()
    return output_corr


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
