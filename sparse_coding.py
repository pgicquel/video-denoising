# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:02:10 2019

@author: pauls
"""

from hard_tresholding import hard_treshold

def sparse(data, Param, W):
    # data is a vector to be sparsed or a matrix with the vectors on the 
    # rows
    # W is the transform matrix, DCT by default
    treshold = Param.sparse_treshold 
    
    code = W @ data
    sparsed_code = hard_treshold(code, treshold)
    
    return sparsed_code