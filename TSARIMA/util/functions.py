# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
import pandas as pd
import tensorly.backend as T
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.base import unfold, fold
import tensorly.tenalg
import tensorly.random
from scipy import linalg
from .svd import svd_fun


def svd_init(tensor, modes, ranks):
    factors = []
    for index, mode in enumerate(modes):
        eigenvecs, _, _ = svd_fun(unfold(tensor, mode), n_eigenvecs=ranks[index])
        factors.append(eigenvecs)
        #print("factor mode: ", index)
    return factors

def init(dims, ranks):
    factors = []
    for index, rank in enumerate(ranks):
        M_i = np.zeros((rank, dims[index]))
        mindim = min(dims[index], rank)
        for i in range(mindim):
            M_i[i][i] = 1
        factors.append(M_i)
    return factors

def autocorr(Y, lag=10,s=1):  #lag为阶数
    """
    计算<Y(t), Y(t-0)>, ..., <Y(t), Y(t-lag)>
    :param Y: list [tensor1, tensor2, ..., tensorT]
    :param lag: int
    :return: array(k+1)
    """
    T = len(Y)
    r = []
#     print("Y")
#     print(Y)
    if s==1:
        for l in range(lag + 1):
            product = 0
            for t in range(T):
                tl = l - t if t < l else t - l
                product += np.sum(Y[t] * Y[tl])
            r.append(product)
    if s!=1:
        for l in range(lag + 1):
            product = 0
            for t in range(T):
                tl = s*l - t if t < s*l else t - s*l
                product += np.sum(Y[t] * Y[tl])
            r.append(product)
    return r

def fit_ar(Y, p,P,s=1):
    r = autocorr(Y, p,1)
    #print("auto-corr:",r)
    R = linalg.toeplitz(r[:p])
    r = r[1:]
    A = linalg.pinv(R).dot(r)

    r = autocorr(Y, P,s)
    # print("auto-corr:",r)
    R = linalg.toeplitz(r[:P])
    r = r[1:]
    A2 = linalg.pinv(R).dot(r)


    return A,A2

def fit_ar_ma(Y,p,P,q,Q,s=1):

    N = len(Y)
    A,A2 = fit_ar(Y, p,P,s)
    B = [0.]
    if q>0:
        Res = []
        for i in range(p,N):
            res = Y[i] - np.sum([ a * Y[i-j] for a, j in zip(A, range(1, p+1))], axis=0)
            Res.append(res)
        #Res = np.array(Res)
        B,B2 = fit_ar(Res, q,Q)
    return A, B, A2, B2


