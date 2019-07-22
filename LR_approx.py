# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:15:18 2019

@author: pauls
"""

import numpy as np
from hard_tresholding import hard_treshold

def LR_approx(D, Param):
    treshold = Param.LR_treshold
    u, s, vh  = np.linalg.svd(D, full_matrices=False)
    s = np.diag(s)
    return u.dot(hard_treshold(s, treshold).dot(vh))