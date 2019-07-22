# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:27:23 2019

@author: pauls
"""
import numpy as np

def overlapping_patches(Video, Param):
    # Decompose all the frame of the video into 2D patches
    # Video is a 3D array and Param a dataclass containing the following
    # parameters
    height , width, frame_num = Video.shape
    stride = Param.stride
    xpsize = Param.xpsize
    ypsize = Param.ypsize
    x_patches_num = int((width - xpsize) // stride) + 1
    y_patches_num = int((height - ypsize) // stride) + 1
    patches_num = x_patches_num * y_patches_num
    Patches = np.zeros((xpsize*ypsize, patches_num*frame_num))
    # Patches are stored in columns
    for idx_frame in range(frame_num):
        k = 0
        for i in range(ypsize):
            for j in range(xpsize):
                
                Block = Video[i:height-ypsize+i+1:stride,
                              j:width-xpsize+j+1:stride, idx_frame]
                Patches[k, idx_frame*patches_num:(idx_frame+1)*patches_num]\
                = np.ravel(Block).T
                k += 1
    
    return Patches