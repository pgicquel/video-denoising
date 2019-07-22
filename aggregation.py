# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:57:08 2019

@author: pauls
"""

import numpy as np

def aggregation(Patches, Param, t):
    xpsize = Param.xpsize
    ypsize = Param.ypsize
    stride = Param.stride
    x_patches_num = Param.x_patches_num
    y_patches_num = Param.y_patches_num
    patches_num = Param.patches_num
    height = Param.height
    width = Param.width
    k = 0
    Agg_block = np.zeros((height, width))
    Weights = np.zeros((height, width))
    for i in range(ypsize):
        for j in range(xpsize):
            Block = Patches[k, t*patches_num:(t+1)*patches_num]
            Block = Block.reshape((y_patches_num, x_patches_num))
            Agg_block[i:height-ypsize+i+1:stride, j:width-xpsize+j+1:stride]\
            += Block
            Weights[i:height-ypsize+i+1:stride, j:width-xpsize+j+1:stride]\
            += 1
            k += 1
    
    Image = Agg_block / Weights
    return Image


            