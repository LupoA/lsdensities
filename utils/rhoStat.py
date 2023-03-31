import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math


def averageVector_fp(vector, get_error=True, get_var=False):
    sum=0
    stdv=0
    for i in range(len(vector)):
        sum+=vector[i]
    sum /= len(vector)
    for i in range(len(vector)):
        stdv += (vector[i]-sum)**2
    stdv /= len(vector)-1
    stdv = math.sqrt(stdv)
    var = stdv
    err = stdv / math.sqrt(len(vector))
    if get_error==True:
        if get_var == True:
            return sum, var
        if get_var == False:
            return sum, err
    if get_error==False:
        return sum
        
def bootstrap_compact_fp(par_, in_):
    from rhoUtils import ranvec
    out_ = np.zeros((par_.num_boot,par_.time_extent))
    randv = np.zeros(par_.num_samples)
    for b in range(0, par_.num_boot):
        randv = ranvec(randv, par_.num_samples, 0, par_.num_samples)
        for i in range(0, par_.time_extent):
            out_[b][i] = 0
            for j in range(0, par_.num_samples):
                out_[b][i] += in_[int(randv[j])][i]
            out_[b][i] /= par_.num_samples
    return out_
    
def bootstrap_fp(T_, nms_, Nb_, in_, out_):
    from rhoUtils import ranvec
    randv = np.zeros(nms_)
    for b in range(0, Nb_):
        randv = ranvec(randv, nms_, 0, nms_)
        for i in range(0, T_):
            for j in range(0, nms_):
                out_[b][i] += in_[int(randv[j])][i]
            out_[b][i] /= nms_
    return out_
    
def getCovMatrix_fp(in_, nbins, vmax, showplot=False):
    cov_ = np.zeros((vmax,vmax))
    for vi in range(vmax):
        for vj in range(vmax):
            cov_[vi][vj] = 0
            for n in range(nbins):
                cov_[vi][vj] += ((in_.sample[n][vi] - in_.central[vi])*(in_.sample[n][vj] - in_.central[vj]))
            cov_[vi][vj] /= (nbins-1)
    if showplot==True:
        plt.imshow(cov_,  cmap='viridis')
        plt.colorbar()
        plt.show()
        plt.clf()
    return cov_

def covToCorr_fp(in_, cov_, vmax, showplot=False):
    corrmat_ = np.zeros((vmax, vmax))
    for vi in range(vmax):
        for vj in range(vmax):
            corrmat_[vi][vj] = cov_[vi][vj] /  (in_.sigma[vi]*in_.sigma[vj])
    if showplot==True:
        plt.imshow(corrmat_)
        plt.colorbar()
        plt.show()
        plt.clf()
    return corrmat_
    
    
from mpmath import mp, mpf

def averageVector_mp(in_, bootstrap=True):
    xlen_ = in_.rows
    samplesize_ = in_.cols
    out_ = mp.matrix(xlen_, 2)
    for x in range(xlen_):
        out_[x, 0] = 0
        out_[x, 1] = 0
        for b in range(samplesize_):
            out_[x,0] = mp.fadd(out_[x,0], in_[x,b])
        out_[x,0] = mp.fdiv(out_[x,0], samplesize_)
        for b in range(samplesize_):
            aux_ = mp.fsub(in_[x,b], out_[x,0])
            aux_ = mp.fmul(aux_, aux_)
            out_[x, 1] = mp.fadd(out_[x, 1], aux_)
        out_[x, 1] = mp.fdiv(out_[x, 1], samplesize_)
        out_[x, 1] = mp.sqrt(out_[x, 1])
        if bootstrap==False:
            aux_ = mp.sqrt(samplesize_)
            out_[x, 1] = mp.fdiv(out_[x, 1], aux_)
    return out_
    

def averageScalar_mp(in_, bootstrap=True):
    samplesize_ = in_.rows
    out_ = mp.matrix(2,1)
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
    if bootstrap==False:
        aux_ = mp.sqrt(samplesize_)
        out_[1] = mp.fdiv(out_[1], aux_)
    return out_

