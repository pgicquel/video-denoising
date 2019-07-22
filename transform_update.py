# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:12:29 2019

@author: pauls
"""

import numpy as np

def trans_update(sparsed_code_array, data_array):
    if len(sparsed_code_array.shape) == 1:
        sparsed_code_array = sparsed_code_array[:,np.newaxis]
    if len(data_array.shape) == 1:
        data_array = data_array[:,np.newaxis]
    Tau = data_array @ sparsed_code_array.T
    u, s, vh = np.linalg.svd(Tau)
    return vh.T @ u.T
    