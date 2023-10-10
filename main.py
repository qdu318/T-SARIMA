#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import libraries

from TSARIMA import TSARIMA
from TSARIMA.util.utility import get_index
import h5py
import numpy as np

if __name__ == "__main__":
    # prepare data
    # the data should be arranged as (ITEM, TIME) pattern
    # import traffic dataset

    Manhattan = np.load("data/Taxi-Manhattan.npy")[:,:,:]
    Manhattan = Manhattan.reshape(-1, 15, 5)

    for i in range(48):

        data = Manhattan[-400 -50:-50+i, :, :]
        data = np.array(data)
        data=np.moveaxis(data, 0, -1)
        # print("shape of data: {}".format(data.shape))
        # print("This dataset have {}*{} series, and each serie have {} time step".format(
        #     data.shape[0],data.shape[1], data.shape[2]
        # ))
        # parameters setting
        ts = data[..., :-1] # training data,
        label = data[:,:, -1] # label, take the last time step as label
        p = 1 # p-order
        d = 1 # d-order
        q = 1 # q-order
        s=48
        P=1
        Q=1
        Rs = [5,5] # tucker decomposition ranks
        k =  10 # iterations
        tol = 0.001 # stop criterion
        Ms_mode = 4 # orthogonality mode
        # Run program

        model = TSARIMA(ts, p, d, q,s,P,Q, Rs, k, tol, verbose=0, Ms_mode=Ms_mode)
        result, _ = model.run()
        pred = result

        index_d=get_index(pred, label)
        print("Evaluation index: \n{}".format(get_index(pred, label)))




