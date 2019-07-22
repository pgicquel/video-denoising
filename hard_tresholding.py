# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:19:34 2019

@author: pauls
"""

import numpy as np

def hard_treshold(X, treshold):
    return X * (np.abs(X) >= treshold)